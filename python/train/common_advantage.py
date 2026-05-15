from __future__ import annotations

"""Shared deterministic GAE recursion helpers.

Mask semantics (single source of truth):
- ``dones[..., t]`` gates the transition *from step t to t+1*.
- ``next_nonterminal = 1 - dones[..., t]`` multiplies both:
  1) bootstrap term into ``V_{t+1}``, and
  2) recursive carry from ``A_{t+1}`` into ``A_t``.
- For the final rollout step, ``last_done`` applies the same gating against
  ``last_value`` bootstrap.

For centralized MAPPO reward aggregation, ``reduce_agents='mean'`` with an
``agent_mask`` computes per-(env,t) weighted means:
``sum_a(reward[a] * mask[a]) / clamp(sum_a(mask[a]), min=1)``.
"""

import torch


def _reduce_agents_mean(rewards: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 3:
        raise ValueError(f"expected rewards with [N, A, T], got shape {tuple(rewards.shape)}")
    if tuple(agent_mask.shape) != tuple(rewards.shape):
        raise ValueError(
            f"agent_mask must match rewards shape {tuple(rewards.shape)}, got {tuple(agent_mask.shape)}"
        )
    active = agent_mask.sum(dim=1).clamp(min=1.0)
    return (rewards * agent_mask).sum(dim=1) / active


def compute_gae_core(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    agent_mask: torch.Tensor | None = None,
    reduce_agents: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic GAE for [N,T] or [N,A,T] layouts."""
    if reduce_agents is None:
        reward_used = rewards
    elif reduce_agents == "mean":
        if agent_mask is None:
            raise ValueError("agent_mask is required when reduce_agents='mean'")
        reward_used = _reduce_agents_mean(rewards, agent_mask)
    else:
        raise ValueError(f"unsupported reduce_agents='{reduce_agents}'")

    if tuple(reward_used.shape) != tuple(values.shape):
        raise ValueError(
            f"reward/value shape mismatch: reward {tuple(reward_used.shape)} vs value {tuple(values.shape)}"
        )
    if reward_used.ndim not in (2, 3):
        raise ValueError(f"expected [N,T] or [N,A,T], got shape {tuple(reward_used.shape)}")

    N = reward_used.shape[0]
    T = reward_used.shape[-1]
    trailing = reward_used.shape[1:-1]

    expected_done = (N, T)
    if tuple(dones.shape) != expected_done:
        raise ValueError(f"dones must have shape {expected_done}, got {tuple(dones.shape)}")
    expected_last = (N,) + trailing
    if tuple(last_value.shape) != expected_last:
        raise ValueError(f"last_value must have shape {expected_last}, got {tuple(last_value.shape)}")
    if tuple(last_done.shape) != (N,):
        raise ValueError(f"last_done must have shape {(N,)}, got {tuple(last_done.shape)}")

    advantages = torch.zeros_like(values)
    last_gae = torch.zeros_like(last_value)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_nonterminal = (1.0 - last_done).view((N,) + (1,) * len(trailing))
        else:
            next_value = values[..., t + 1]
            next_nonterminal = (1.0 - dones[:, t]).view((N,) + (1,) * len(trailing))
        delta = reward_used[..., t] + gamma * next_value * next_nonterminal - values[..., t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[..., t] = last_gae

    returns = advantages + values
    return advantages, returns
