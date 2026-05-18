from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from train.mappo_model import mode_aux_loss_and_accuracy, target_selection_aux_loss_and_accuracy
from train.ppo_recurrent.losses import action_logprob_and_entropy, compute_ppo_loss


@dataclass(frozen=True)
class MappoBatchTensors:
    """Normalized flattened batch tensors used by PPO objectives.

    Shapes:
        new_logprob: [N, A, L]
        entropy: [N, A, L]
        value: [N, A, L] when value_per_agent else [N, L]
        advantage: [N, A, L]
        valid_agent_mask: [N, A, L]
        value_mask: [N, A, L] when value_per_agent else [N, L]
    """

    new_logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    advantage: torch.Tensor
    valid_agent_mask: torch.Tensor
    value_mask: torch.Tensor


def action_logprob_entropy(cfg, mean, log_std, binary_logits, target_logits, action):
    base_end = cfg.continuous_action_dim + cfg.binary_action_dim
    logp, ent = action_logprob_and_entropy(mean, log_std, binary_logits, action[:, :base_end])
    if cfg.target_action_dim > 0:
        if target_logits is None:
            raise RuntimeError("target_action_dim requires target logits")
        target = action[:, base_end].long().clamp(0, cfg.target_action_dim - 1)
        dist = torch.distributions.Categorical(logits=target_logits)
        logp = logp + dist.log_prob(target)
        ent = ent + dist.entropy()
    return logp, ent


def compute_policy_trajectory_logprob_entropy(
    trainer,
    rollout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent policy over rollout and return trajectory log-probabilities/entropy.

    Args:
        trainer: trainer providing cfg and model.policy_outputs.
        rollout: rollout buffers with actor_obs [N, A, L, obs_dim], action [N, A, L, action_dim],
            done [N, L], and h_init [N, A, 1, H].

    Returns:
        new_logprob: [N, A, L]
        entropy: [N, A, L]
    """

    cfg = trainer.cfg
    N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
    flat_h = rollout.h_init[:, :, 0].reshape(N * A, cfg.gru_hidden)
    logprobs, entropies = [], []
    h = flat_h

    for t in range(L):
        obs_t = rollout.actor_obs[:, :, t].reshape(N * A, cfg.obs_dim)
        mean, log_std, logits, target_logits, h = trainer.model.policy_outputs(obs_t, h)
        target_logits = trainer.model._masked_target_logits(
            target_logits,
            trainer.model._target_mask(obs_t),
        )
        action_t = rollout.action[:, :, t].reshape(N * A, cfg.action_dim)
        logp, ent = action_logprob_entropy(cfg, mean, log_std, logits, target_logits, action_t)
        logprobs.append(logp.view(N, A))
        entropies.append(ent.view(N, A))
        done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
        h = h.view(N, A, cfg.gru_hidden)
        h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)

    return torch.stack(logprobs, dim=2), torch.stack(entropies, dim=2)


def apply_optimizer_step(trainer, total_loss: torch.Tensor) -> tuple[float, float, float]:
    """Apply backward + optimizer step and capture grad norms.

    Args:
        total_loss: scalar tensor.

    Returns:
        actor_grad_norm, critic_grad_norm, trunk_grad_norm (pre-clipping).
    """

    cfg = trainer.cfg
    trainer.optimizer.zero_grad()
    total_loss.backward()
    actor_grad_norm = trainer._group_grad_norm(trainer._actor_params)
    critic_grad_norm = trainer._group_grad_norm(trainer._critic_params)
    trunk_grad_norm = trainer._group_grad_norm(trainer._trunk_params)
    nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
    trainer.optimizer.step()
    return actor_grad_norm, critic_grad_norm, trunk_grad_norm


def _build_batch_tensors(trainer, rollout) -> MappoBatchTensors:
    cfg = trainer.cfg
    N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len

    new_logprob, entropy = compute_policy_trajectory_logprob_entropy(trainer, rollout)
    valid_agent = rollout.agent_loss_mask.expand(N, A, L)
    if cfg.value_per_agent:
        value = (
            trainer.model.value(
                rollout.critic_obs.permute(0, 2, 1, 3).reshape(
                    N * L * A,
                    cfg.critic_obs_dim,
                )
            )
            .view(N, L, A)
            .permute(0, 2, 1)
        )
        advantage = rollout.advantages
        value_mask = valid_agent
    else:
        value = trainer.model.value(
            rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)
        ).view(N, L)
        advantage = rollout.advantages[:, None, :].expand(N, A, L)
        value_mask = (valid_agent.sum(dim=1) > 0.0).to(valid_agent.dtype)

    return MappoBatchTensors(
        new_logprob=new_logprob,
        entropy=entropy,
        value=value,
        advantage=advantage,
        valid_agent_mask=valid_agent,
        value_mask=value_mask,
    )


def update_full_rollout(
    trainer,
    rollout,
    return_mean: float,
    return_std: float,
) -> dict[str, float]:
    # PPO Eq. (1-2): evaluate recurrent policy over trajectory.
    batch = _build_batch_tensors(trainer, rollout)

    cfg = trainer.cfg
    loss = compute_ppo_loss(
        new_logprob=batch.new_logprob,
        old_logprob=rollout.logprob,
        advantage=batch.advantage,
        value=batch.value,
        old_value=rollout.value,
        return_=rollout.returns,
        valid_mask=batch.valid_agent_mask,
        clip_ratio=cfg.clip_ratio,
        value_clip_ratio=cfg.value_clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        entropy=batch.entropy,
        return_mean=return_mean,
        return_std=return_std,
        value_mask=batch.value_mask,
    )

    target_aux_losses, target_aux_accs, target_aux_counts = [], [], []
    mode_aux_losses, mode_aux_accs, mode_aux_counts = [], [], []
    p_combat_parts = []
    if cfg.target_selection_aux_coef > 0.0 or cfg.mode_gated_combat:
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        h = rollout.h_init[:, :, 0].reshape(N * A, cfg.gru_hidden)
        for t in range(L):
            obs_t = rollout.actor_obs[:, :, t].reshape(N * A, cfg.obs_dim)
            features, h = trainer.model.actor_head_features(obs_t, h)
            _mean, _logits, target_selection_logits = trainer.model.policy_heads_from_features(
                obs_t, features
            )
            mode_logits = trainer.model.mode_logits_from_features(features)
            p_combat = trainer.model.combat_probability(mode_logits)
            if p_combat is not None:
                p_combat_parts.append(p_combat.mean())
            flat_mask = rollout.agent_loss_mask[:, :, t].reshape(N * A)
            mode_loss, mode_acc, mode_count = mode_aux_loss_and_accuracy(
                mode_logits, obs_t, cfg, mask=flat_mask
            )
            mode_aux_losses.append(mode_loss)
            mode_aux_accs.append(mode_acc)
            mode_aux_counts.append(mode_count)
            aux_loss, aux_acc, aux_count = target_selection_aux_loss_and_accuracy(
                target_selection_logits, obs_t, cfg, mask=flat_mask
            )
            target_aux_losses.append(aux_loss)
            target_aux_accs.append(aux_acc)
            target_aux_counts.append(aux_count)
            done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
            h = h.view(N, A, cfg.gru_hidden)
            h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)
    if target_aux_losses:
        target_aux_loss = torch.stack(target_aux_losses).mean()
        target_aux_count = torch.stack(target_aux_counts).sum()
        target_aux_acc = (
            torch.stack(target_aux_accs).mean()
            if float(target_aux_count.item()) > 0.0
            else rollout.actor_obs.new_tensor(0.0)
        )
    else:
        target_aux_loss = rollout.actor_obs.new_tensor(0.0)
        target_aux_acc = rollout.actor_obs.new_tensor(0.0)
        target_aux_count = rollout.actor_obs.new_tensor(0.0)
    if mode_aux_losses:
        mode_aux_loss = torch.stack(mode_aux_losses).mean()
        mode_aux_count = torch.stack(mode_aux_counts).sum()
        mode_aux_acc = (
            torch.stack(mode_aux_accs).mean()
            if float(mode_aux_count.item()) > 0.0
            else rollout.actor_obs.new_tensor(0.0)
        )
        mean_p_combat = (
            torch.stack(p_combat_parts).mean()
            if p_combat_parts
            else rollout.actor_obs.new_tensor(0.0)
        )
    else:
        mode_aux_loss = rollout.actor_obs.new_tensor(0.0)
        mode_aux_acc = rollout.actor_obs.new_tensor(0.0)
        mode_aux_count = rollout.actor_obs.new_tensor(0.0)
        mean_p_combat = rollout.actor_obs.new_tensor(0.0)
    total_loss = (
        loss.total_loss
        + cfg.target_selection_aux_coef * target_aux_loss
        + (cfg.mode_aux_coef * mode_aux_loss if cfg.mode_gated_combat else 0.0)
    )

    # Optimization: backward, grad metrics, global grad clip, optimizer step.
    actor_grad_norm, critic_grad_norm, trunk_grad_norm = apply_optimizer_step(trainer, total_loss)

    scalar_keys = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "total_loss",
    )
    scalars = torch.stack(
        [
            loss.policy_loss,
            loss.value_loss,
            loss.entropy,
            loss.approx_kl,
            loss.clip_fraction,
            total_loss,
        ]
    ).detach().cpu().tolist()
    out = {k: float(v) for k, v in zip(scalar_keys, scalars, strict=False)}
    out["actor_grad_norm"] = actor_grad_norm
    out["critic_grad_norm"] = critic_grad_norm
    out["trunk_grad_norm"] = trunk_grad_norm
    out["target_selection_aux_loss"] = float(target_aux_loss.detach().cpu().item())
    out["target_selection_aux_accuracy"] = float(target_aux_acc.detach().cpu().item())
    out["target_selection_aux_count"] = float(target_aux_count.detach().cpu().item())
    out["mode_aux_loss"] = float(mode_aux_loss.detach().cpu().item())
    out["mode_accuracy"] = float(mode_aux_acc.detach().cpu().item())
    out["mode_aux_count"] = float(mode_aux_count.detach().cpu().item())
    out["mean_p_combat"] = float(mean_p_combat.detach().cpu().item())
    out["lr"] = trainer.current_learning_rate
    return out
