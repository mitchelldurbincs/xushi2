from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

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


def compute_policy_trajectory_logprob_entropy(trainer, rollout) -> tuple[torch.Tensor, torch.Tensor]:
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
        target_logits = trainer.model._masked_target_logits(target_logits, trainer.model._target_mask(obs_t))
        action_t = rollout.action[:, :, t].reshape(N * A, cfg.action_dim)
        logp, ent = action_logprob_entropy(cfg, mean, log_std, logits, target_logits, action_t)
        logprobs.append(logp.view(N, A))
        entropies.append(ent.view(N, A))
        done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
        h = h.view(N, A, cfg.gru_hidden)
        h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)

    return torch.stack(logprobs, dim=2), torch.stack(entropies, dim=2)


def compute_policy_loss(cfg, rollout, batch: MappoBatchTensors, *, return_mean: float, return_std: float):
    """Compute PPO clipped policy objective.

    Inputs:
        rollout.logprob: [N, A, L]
        batch.new_logprob: [N, A, L]
        batch.advantage: [N, A, L]
        batch.valid_agent_mask: [N, A, L]
        batch.entropy: [N, A, L]

    Returns:
        PpoLoss namedtuple with policy/value/entropy/kl/clip stats and total loss scalar.
    """

    return compute_ppo_loss(
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


def compute_value_loss(cfg, rollout, batch: MappoBatchTensors, *, return_mean: float, return_std: float):
    """Compute full PPO objective including value regression terms.

    Inputs:
        batch.value: [N, A, L] or [N, L]
        rollout.value: same leading shape semantics as batch.value
        rollout.returns: same leading shape semantics as batch.value
        batch.value_mask: [N, A, L] or [N, L]

    Returns:
        PpoLoss namedtuple with total_loss for backward pass.
    """

    return compute_ppo_loss(
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
            trainer.model.value(rollout.critic_obs.permute(0, 2, 1, 3).reshape(N * L * A, cfg.critic_obs_dim))
            .view(N, L, A)
            .permute(0, 2, 1)
        )
        advantage = rollout.advantages
        value_mask = valid_agent
    else:
        value = trainer.model.value(rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)).view(N, L)
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


def update_full_rollout(trainer, rollout, return_mean: float, return_std: float) -> dict[str, float]:
    # PPO Eq. (1-2): evaluate recurrent policy over trajectory, collect pi_theta(a_t|s_t) and entropy.
    batch = _build_batch_tensors(trainer, rollout)

    # PPO Eq. (3): clipped surrogate policy objective with entropy regularization.
    policy_loss_terms = compute_policy_loss(trainer.cfg, rollout, batch, return_mean=return_mean, return_std=return_std)

    # PPO Eq. (4): clipped value objective + combined total objective.
    loss = compute_value_loss(trainer.cfg, rollout, batch, return_mean=return_mean, return_std=return_std)
    # Keep policy-objective computation explicit in staged pipeline; both objectives should agree on policy stats.
    _ = policy_loss_terms.policy_loss

    # Optimization: backward, grad metrics, global grad clip, optimizer step.
    actor_grad_norm, critic_grad_norm, trunk_grad_norm = apply_optimizer_step(trainer, loss.total_loss)

    return {
        "policy_loss": float(loss.policy_loss.item()),
        "value_loss": float(loss.value_loss.item()),
        "entropy": float(loss.entropy.item()),
        "approx_kl": float(loss.approx_kl.item()),
        "clip_fraction": float(loss.clip_fraction.item()),
        "total_loss": float(loss.total_loss.item()),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "trunk_grad_norm": trunk_grad_norm,
        "lr": trainer.current_learning_rate,
    }
