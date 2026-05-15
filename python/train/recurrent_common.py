"""Shared helpers for recurrent PPO/MAPPO trainers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def apply_global_seeds(seed: int) -> None:
    """Seed torch + numpy with a normalized integer seed."""
    normalized_seed = int(seed)
    torch.manual_seed(normalized_seed)
    np.random.seed(normalized_seed)


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> float:
    """Apply ``lr`` to every optimizer group and return normalized value."""
    normalized_lr = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = normalized_lr
    return normalized_lr


def get_optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Read learning rate from the first optimizer group."""
    return float(optimizer.param_groups[0]["lr"])


def grad_group_norm(params: list[torch.nn.Parameter]) -> float:
    """L2 norm across gradients for a parameter group."""
    total_sq = 0.0
    for param in params:
        if param.grad is not None:
            total_sq += float(param.grad.detach().pow(2).sum().item())
    return float(total_sq**0.5)


@dataclass(frozen=True)
class UpdateSamplingState:
    """Utility state for deterministic update scheduling and minibatch RNG."""

    update_counter: int
    minibatch_seed: int


def next_update_sampling_state(seed: int, update_counter: int) -> UpdateSamplingState:
    """Return deterministic minibatch seed and next update counter input."""
    normalized_counter = int(update_counter)
    return UpdateSamplingState(
        update_counter=normalized_counter + 1,
        minibatch_seed=int(seed) * 1_000_003 + (normalized_counter + 1),
    )
