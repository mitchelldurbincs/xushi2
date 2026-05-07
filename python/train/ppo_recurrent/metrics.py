from __future__ import annotations

import torch


def init_metrics_sum() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "total_loss": 0.0,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "trunk_grad_norm": 0.0,
    }


def accumulate(metrics_sum: dict[str, float], mb_stats: dict[str, float], n_valid: float) -> None:
    if n_valid <= 0:
        return
    for key, val in mb_stats.items():
        metrics_sum[key] += val * n_valid


def reduce_metrics(metrics_sum: dict[str, float], *, total_valid: float, num_minibatches: int, lr: float) -> dict[str, float]:
    denom = max(total_valid, 1.0)
    out = {k: v / denom for k, v in metrics_sum.items()}
    out["num_minibatches"] = float(num_minibatches)
    out["total_valid"] = float(total_valid)
    out["lr"] = float(lr)
    return out


def add_post_update_diagnostics(metrics: dict[str, float], *, rollout, model) -> None:
    with torch.no_grad():
        done_mask = rollout.done > 0.5
        if bool(done_mask.any()):
            metrics["terminal_adv_std"] = float(rollout.advantages[done_mask].std().item())
        else:
            metrics["terminal_adv_std"] = 0.0
        metrics["mean_log_std"] = float(model.log_std.detach().mean().item())
