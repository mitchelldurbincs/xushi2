from __future__ import annotations

"""Shared deterministic GAE recursion helpers.

Mask semantics (single source of truth):
- ``dones[..., t]`` marks the end of an episode between step t and t+1, and
  gates the recursive carry from ``A_{t+1}`` into ``A_t``: the next episode's
  advantage must never flow backwards into this one.
- ``terminateds[..., t]`` marks a *true* MDP terminal and gates the bootstrap
  into ``V_{t+1}``. A time-limit truncation is an episode boundary but not a
  terminal state, so its value must still be bootstrapped -- otherwise the
  critic is taught that every timeout is worth zero. In this game the round
  timer expiring is the common ending, so conflating the two biased V(s) low
  across the end of nearly every episode.
- When ``terminateds`` is omitted it defaults to ``dones``, which reproduces
  the older conflated behavior.
- For the final rollout step, ``last_done`` / ``last_terminated`` apply the
  same gating against the ``last_value`` bootstrap.

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
    terminateds: torch.Tensor | None = None,
    last_terminated: torch.Tensor | None = None,
    truncated_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deterministic GAE for [N,T] or [N,A,T] layouts.

    ``truncated_values`` optionally supplies ``V(s_T)`` for the *pre-reset*
    state at each step. Where an episode was truncated rather than terminated,
    that value is what the bootstrap should use: ``values[..., t+1]`` describes
    the freshly reset next episode, not the state the agent actually left.
    """
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

    if terminateds is None:
        terminateds = dones
    elif tuple(terminateds.shape) != expected_done:
        raise ValueError(
            f"terminateds must have shape {expected_done}, got {tuple(terminateds.shape)}"
        )
    if last_terminated is None:
        last_terminated = last_done
    elif tuple(last_terminated.shape) != (N,):
        raise ValueError(
            f"last_terminated must have shape {(N,)}, got {tuple(last_terminated.shape)}"
        )
    if truncated_values is not None and tuple(truncated_values.shape) != tuple(values.shape):
        raise ValueError(
            f"truncated_values must match values shape {tuple(values.shape)}, "
            f"got {tuple(truncated_values.shape)}"
        )

    advantages = torch.zeros_like(values)
    last_gae = torch.zeros_like(last_value)
    view_shape = (N,) + (1,) * len(trailing)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            episode_continues = (1.0 - last_done).view(view_shape)
            not_terminal = (1.0 - last_terminated).view(view_shape)
            truncated = ((1.0 - last_terminated) * last_done).view(view_shape)
        else:
            next_value = values[..., t + 1]
            episode_continues = (1.0 - dones[:, t]).view(view_shape)
            not_terminal = (1.0 - terminateds[:, t]).view(view_shape)
            truncated = ((1.0 - terminateds[:, t]) * dones[:, t]).view(view_shape)
        # On a truncated step `next_value` describes the reset episode, so swap
        # in V(s_T) for the state the agent actually left, when available.
        if truncated_values is not None:
            next_value = torch.where(
                truncated.expand_as(next_value) > 0.0, truncated_values[..., t], next_value
            )
        # Bootstrap survives truncation; only a real terminal zeroes it. The
        # recursive carry is cut at every episode boundary, terminal or not.
        delta = reward_used[..., t] + gamma * next_value * not_terminal - values[..., t]
        last_gae = delta + gamma * gae_lambda * episode_continues * last_gae
        advantages[..., t] = last_gae

    returns = advantages + values
    return advantages, returns
