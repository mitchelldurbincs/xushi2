"""Structured logging helpers for recurrent PPO orchestration."""

from __future__ import annotations

from typing import Mapping


LogEvent = dict[str, object]


def _event_base(*, event: str, phase: str, variant: str, update: int, total_updates: int) -> LogEvent:
    return {
        "event": event,
        "phase": phase,
        "variant": variant,
        "update": int(update),
        "total_updates": int(total_updates),
    }


def format_human_event(record: Mapping[str, object]) -> str:
    event = str(record["event"])
    phase = str(record["phase"])
    variant = str(record["variant"])

    if event == "update":
        return (
            f"[{phase}/{variant}] update={int(record['update']):4d}/{int(record['total_updates'])} "
            f"policy_loss={float(record['policy_loss']):+.3f} "
            f"value_loss={float(record['value_loss']):.3f} "
            f"entropy={float(record['entropy']):+.3f} "
            f"approx_kl={float(record['approx_kl']):+.4f} "
            f"actor_gn={float(record['actor_grad_norm']):.3f} "
            f"critic_gn={float(record['critic_grad_norm']):.3f} "
            f"trunk_gn={float(record['trunk_grad_norm']):.3f} "
            f"term_adv_std={float(record['terminal_adv_std']):.3f} "
            f"mean_log_std={float(record['mean_log_std']):+.3f} "
            f"lr={float(record['lr']):.3e}"
        )

    if event == "eval":
        return (
            f"[{phase}/{variant}] eval@{int(record['update'])}={float(record['mean_reward']):+.3f} "
            f"win={int(record['wins'])}/{int(record['episodes'])} "
            f"loss={int(record['losses'])}/{int(record['episodes'])} "
            f"draw={int(record['draws'])}/{int(record['episodes'])} "
            f"term={int(record['terminated'])}/{int(record['episodes'])} "
            f"trunc={int(record['truncated'])}/{int(record['episodes'])} "
            f"tick={float(record['mean_final_tick']):.1f} "
            f"score=A{float(record['mean_team_a_score']):.2f}/B{float(record['mean_team_b_score']):.2f} "
            f"kills=A{float(record['mean_team_a_kills']):.2f}/B{float(record['mean_team_b_kills']):.2f} "
            f"lr={float(record['lr']):.3e}"
        )

    if event == "checkpoint":
        return f"[{phase}/{variant}] checkpoint@{int(record['update'])}: {record['path']}"

    if event == "early_stop":
        return f"[{phase}/{variant}] early-stop: {record['reason']}"

    raise ValueError(f"unknown event type: {event}")


def log_update(*, phase: str, variant: str, update: int, total_updates: int, metrics: Mapping[str, float]) -> LogEvent:
    return {
        **_event_base(event="update", phase=phase, variant=variant, update=update, total_updates=total_updates),
        "policy_loss": float(metrics["policy_loss"]),
        "value_loss": float(metrics["value_loss"]),
        "entropy": float(metrics["entropy"]),
        "approx_kl": float(metrics["approx_kl"]),
        "actor_grad_norm": float(metrics["actor_grad_norm"]),
        "critic_grad_norm": float(metrics["critic_grad_norm"]),
        "trunk_grad_norm": float(metrics["trunk_grad_norm"]),
        "terminal_adv_std": float(metrics["terminal_adv_std"]),
        "mean_log_std": float(metrics["mean_log_std"]),
        "lr": float(metrics["lr"]),
    }


def log_eval(*, phase: str, variant: str, update: int, total_updates: int, lr: float, eval_stats: object) -> LogEvent:
    return {
        **_event_base(event="eval", phase=phase, variant=variant, update=update, total_updates=total_updates),
        "mean_reward": float(eval_stats.mean_reward),
        "wins": int(eval_stats.wins),
        "losses": int(eval_stats.losses),
        "draws": int(eval_stats.draws),
        "terminated": int(eval_stats.terminated),
        "truncated": int(eval_stats.truncated),
        "episodes": int(eval_stats.episodes),
        "mean_final_tick": float(eval_stats.mean_final_tick),
        "mean_team_a_score": float(eval_stats.mean_team_a_score),
        "mean_team_b_score": float(eval_stats.mean_team_b_score),
        "mean_team_a_kills": float(eval_stats.mean_team_a_kills),
        "mean_team_b_kills": float(eval_stats.mean_team_b_kills),
        "lr": float(lr),
    }


def log_checkpoint(*, phase: str, variant: str, update: int, total_updates: int, path: str) -> LogEvent:
    return {
        **_event_base(event="checkpoint", phase=phase, variant=variant, update=update, total_updates=total_updates),
        "path": path,
    }


def log_early_stop(*, phase: str, variant: str, update: int, total_updates: int, reason: str) -> LogEvent:
    return {
        **_event_base(event="early_stop", phase=phase, variant=variant, update=update, total_updates=total_updates),
        "reason": reason,
    }
