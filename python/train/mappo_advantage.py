from __future__ import annotations

import torch

from train.common_advantage import compute_gae_core
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
    if cfg.value_per_agent:
        rollout.advantages, rollout.returns = compute_gae_core(
            rewards=rollout.reward,
            values=rollout.value,
            dones=rollout.done,
            last_value=rollout.last_value,
            last_done=rollout.last_done,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        return

    _validate_agent_loss_mask(rollout.agent_loss_mask, cfg)
    rollout.advantages, rollout.returns = compute_gae_core(
        rewards=rollout.reward,
        values=rollout.value,
        dones=rollout.done,
        last_value=rollout.last_value,
        last_done=rollout.last_done,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        agent_mask=rollout.agent_loss_mask,
        reduce_agents="mean",
    )
