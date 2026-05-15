from __future__ import annotations

import torch

from train.mappo_model import MappoConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.mappo_rollout_trainer import MappoRollout


def _validate_agent_loss_mask(mask: torch.Tensor, cfg: MappoConfig) -> None:
    expected_shape = (cfg.num_envs, cfg.n_agents, cfg.rollout_len)
    if tuple(mask.shape) != expected_shape:
        raise AssertionError(
            f"agent_loss_mask must have shape {expected_shape}, got {tuple(mask.shape)}"
        )
    if torch.any(mask < 0.0):
        raise AssertionError("agent_loss_mask entries must be non-negative")
    if torch.any(mask.sum(dim=1) <= 0.0):
        raise AssertionError("agent_loss_mask must leave at least one active agent per env-step")


def compute_gae(rollout: "MappoRollout", cfg: MappoConfig) -> None:
    device = rollout.advantages.device
    if cfg.value_per_agent:
        last_gae = torch.zeros(cfg.num_envs, cfg.n_agents, device=device)
        for t in reversed(range(cfg.rollout_len)):
            if t == cfg.rollout_len - 1:
                next_value = rollout.last_value
                next_nonterminal = (1.0 - rollout.last_done).view(cfg.num_envs, 1)
            else:
                next_value = rollout.value[:, :, t + 1]
                next_nonterminal = (1.0 - rollout.done[:, t]).view(cfg.num_envs, 1)
            delta = (
                rollout.reward[:, :, t]
                + cfg.gamma * next_value * next_nonterminal
                - rollout.value[:, :, t]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            rollout.advantages[:, :, t] = last_gae
        rollout.returns = rollout.advantages + rollout.value
        return

    _validate_agent_loss_mask(rollout.agent_loss_mask, cfg)
    active_count = rollout.agent_loss_mask.sum(dim=1).clamp(min=1.0)
    reward = (rollout.reward * rollout.agent_loss_mask).sum(dim=1) / active_count
    last_gae = torch.zeros(cfg.num_envs, device=device)
    for t in reversed(range(cfg.rollout_len)):
        if t == cfg.rollout_len - 1:
            next_value = rollout.last_value
            next_nonterminal = 1.0 - rollout.last_done
        else:
            next_value = rollout.value[:, t + 1]
            next_nonterminal = 1.0 - rollout.done[:, t]
        delta = reward[:, t] + cfg.gamma * next_value * next_nonterminal - rollout.value[:, t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        rollout.advantages[:, t] = last_gae
    rollout.returns = rollout.advantages + rollout.value
