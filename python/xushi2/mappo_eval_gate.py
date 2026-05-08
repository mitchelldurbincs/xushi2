"""Gate helpers for compact MAPPO eval diagnostics."""

from __future__ import annotations

from typing import Any


def check_eval_gate(metrics: dict[str, Any], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a pass/fail summary for one MAPPO eval-stat dictionary."""
    episodes = max(1, int(metrics.get("episodes", 0)))
    wins = int(metrics.get("wins", 0))
    losses = int(metrics.get("losses", 0))
    draws = int(metrics.get("draws", 0))
    win_rate = float(metrics.get("win_rate", wins / float(episodes)))
    loss_rate = float(metrics.get("loss_rate", losses / float(episodes)))
    draw_rate = float(metrics.get("draw_rate", draws / float(episodes)))
    mean_reward = float(metrics.get("mean_reward", 0.0))
    mean_score_a = float(metrics.get("mean_score_a", 0.0))
    mean_score_b = float(metrics.get("mean_score_b", 0.0))
    mean_final_tick = float(metrics.get("mean_final_tick", 0.0))

    failures: list[str] = []
    criteria: dict[str, float | int] = {}

    def min_check(name: str, value: float) -> None:
        if name in gate_cfg:
            threshold = float(gate_cfg[name])
            criteria[name] = threshold
            if value < threshold:
                failures.append(f"{name} {value:.3f} < {threshold:.3f}")

    def max_check(name: str, value: float) -> None:
        if name in gate_cfg:
            threshold = float(gate_cfg[name])
            criteria[name] = threshold
            if value > threshold:
                failures.append(f"{name} {value:.3f} > {threshold:.3f}")

    min_check("min_win_rate", win_rate)
    max_check("max_loss_rate", loss_rate)
    max_check("max_draw_rate", draw_rate)
    min_check("min_mean_reward", mean_reward)
    min_check("min_mean_score_a", mean_score_a)
    max_check("max_mean_score_b", mean_score_b)
    max_check("max_mean_final_tick", mean_final_tick)
    if "min_episodes" in gate_cfg:
        threshold = int(gate_cfg["min_episodes"])
        criteria["min_episodes"] = threshold
        if episodes < threshold:
            failures.append(f"episodes {episodes} < {threshold}")

    return {
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "mean_reward": mean_reward,
            "mean_score_a": mean_score_a,
            "mean_score_b": mean_score_b,
            "mean_final_tick": mean_final_tick,
        },
        "criteria": criteria,
    }
