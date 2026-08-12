from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train.common_advantage import compute_gae_core
from train.mappo_model import MappoConfig

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


def compute_gae(rollout: MappoRollout, cfg: MappoConfig) -> None:
    # Validate before the branch: "at least one active agent per env-step" is
    # exactly the invariant _rollout_metrics relies on, and skipping the check
    # on the value_per_agent path left it unverified there.
    _validate_agent_loss_mask(rollout.agent_loss_mask, cfg)
    shared = {
        "rewards": rollout.reward,
        "values": rollout.value,
        "dones": rollout.done,
        "last_value": rollout.last_value,
        "last_done": rollout.last_done,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        # Time-limit truncations end the episode but are not MDP terminals, so
        # they keep their bootstrap -- taken from the pre-reset state's value.
        "terminateds": rollout.terminated,
        "last_terminated": rollout.last_terminated,
        "truncated_values": rollout.truncated_value,
    }
    if cfg.value_per_agent:
        rollout.advantages, rollout.returns = compute_gae_core(**shared)
        return

    rollout.advantages, rollout.returns = compute_gae_core(
        **shared,
        agent_mask=rollout.agent_loss_mask,
        reduce_agents="mean",
    )
