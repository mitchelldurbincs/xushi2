from __future__ import annotations

import torch
import torch.nn as nn

from train.ppo_recurrent.losses import action_logprob_and_entropy, compute_ppo_loss


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


def update_full_rollout(trainer, rollout, return_mean: float, return_std: float) -> dict[str, float]:
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
    new_logprob = torch.stack(logprobs, dim=2)
    entropy = torch.stack(entropies, dim=2)
    valid_agent = rollout.agent_loss_mask.expand(N, A, L)
    if cfg.value_per_agent:
        value = (
            trainer.model.value(rollout.critic_obs.permute(0, 2, 1, 3).reshape(N * L * A, cfg.critic_obs_dim))
            .view(N, L, A)
            .permute(0, 2, 1)
        )
        advantage = rollout.advantages
    else:
        value = trainer.model.value(rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)).view(N, L)
        advantage = rollout.advantages[:, None, :].expand(N, A, L)
    value_mask = valid_agent if cfg.value_per_agent else (valid_agent.sum(dim=1) > 0.0).to(valid_agent.dtype)

    loss = compute_ppo_loss(
        new_logprob=new_logprob,
        old_logprob=rollout.logprob,
        advantage=advantage,
        value=value,
        old_value=rollout.value,
        return_=rollout.returns,
        valid_mask=valid_agent,
        clip_ratio=cfg.clip_ratio,
        value_clip_ratio=cfg.value_clip_ratio,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        entropy=entropy,
        return_mean=return_mean,
        return_std=return_std,
        value_mask=value_mask,
    )

    trainer.optimizer.zero_grad()
    loss.total_loss.backward()
    actor_grad_norm = trainer._group_grad_norm(trainer._actor_params)
    critic_grad_norm = trainer._group_grad_norm(trainer._critic_params)
    trunk_grad_norm = trainer._group_grad_norm(trainer._trunk_params)
    nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
    trainer.optimizer.step()

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
            loss.total_loss,
        ]
    ).detach().cpu().tolist()
    out = {k: float(v) for k, v in zip(scalar_keys, scalars, strict=False)}
    out["actor_grad_norm"] = actor_grad_norm
    out["critic_grad_norm"] = critic_grad_norm
    out["trunk_grad_norm"] = trunk_grad_norm
    out["lr"] = trainer.current_learning_rate
    return out
