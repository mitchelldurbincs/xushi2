"""Per-update learning-rate schedule for PPO.

Pure function — takes the current update index and hyperparameters, returns
the LR to apply. Supported schedules: constant, linear, cosine, all with
optional linear warmup.
"""

from __future__ import annotations

import math


def lr_for_update(
    update_idx: int,
    total_updates: int,
    *,
    base_lr: float,
    schedule: str = "constant",
    lr_final_ratio: float = 1.0,
    warmup_updates: int = 0,
) -> float:
    """Return the learning rate to use for a 1-based PPO update index."""
    if total_updates <= 0:
        raise ValueError(f"total_updates must be > 0, got {total_updates}")
    if not (1 <= update_idx <= total_updates):
        raise ValueError(
            f"update_idx must be in [1, {total_updates}], got {update_idx}"
        )
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if lr_final_ratio <= 0.0:
        raise ValueError(
            f"lr_final_ratio must be > 0, got {lr_final_ratio}"
        )
    if warmup_updates < 0:
        raise ValueError(
            f"warmup_updates must be >= 0, got {warmup_updates}"
        )
    if warmup_updates >= total_updates:
        raise ValueError(
            "warmup_updates must be < total_updates so the schedule phase "
            f"still has at least one update; got warmup_updates="
            f"{warmup_updates}, total_updates={total_updates}"
        )

    schedule_name = schedule.strip().lower()

    if warmup_updates > 0 and update_idx <= warmup_updates:
        return float(base_lr) * (float(update_idx) / float(warmup_updates))

    if schedule_name == "constant":
        return float(base_lr)

    schedule_updates = total_updates - warmup_updates
    if schedule_updates <= 1:
        return float(base_lr) * float(lr_final_ratio)

    progress = float(update_idx - warmup_updates - 1) / float(schedule_updates - 1)
    if schedule_name == "linear":
        ratio = 1.0 + progress * (float(lr_final_ratio) - 1.0)
    elif schedule_name == "cosine":
        ratio = float(lr_final_ratio) + 0.5 * (1.0 - float(lr_final_ratio)) * (
            1.0 + math.cos(math.pi * progress)
        )
    else:
        raise ValueError(f"unknown lr schedule: {schedule!r}")

    return float(base_lr) * ratio
