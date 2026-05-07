from __future__ import annotations

from dataclasses import dataclass

from train.ppo_recurrent.logging import (
    format_human_event,
    log_checkpoint,
    log_early_stop,
    log_eval,
    log_update,
)


@dataclass
class _EvalStats:
    mean_reward: float = 1.0
    episodes: int = 8
    wins: int = 3
    losses: int = 2
    draws: int = 3
    terminated: int = 4
    truncated: int = 4
    mean_final_tick: float = 10.0
    mean_team_a_score: float = 1.2
    mean_team_b_score: float = 1.1
    mean_team_a_kills: float = 0.3
    mean_team_b_kills: float = 0.2


def test_log_update_schema_has_expected_keys() -> None:
    record = log_update(
        phase="phase2",
        variant="recurrent",
        update=5,
        total_updates=100,
        metrics={
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "approx_kl": 0.4,
            "actor_grad_norm": 0.5,
            "critic_grad_norm": 0.6,
            "trunk_grad_norm": 0.7,
            "terminal_adv_std": 0.8,
            "mean_log_std": 0.9,
            "lr": 1e-3,
        },
    )
    assert record["event"] == "update"
    assert {"phase", "variant", "update", "total_updates", "policy_loss", "lr"}.issubset(record)
    assert "update=" in format_human_event(record)


def test_log_eval_schema_has_expected_keys() -> None:
    record = log_eval(
        phase="phase2",
        variant="feedforward",
        update=7,
        total_updates=100,
        lr=3e-4,
        eval_stats=_EvalStats(),
    )
    assert record["event"] == "eval"
    assert {"mean_reward", "episodes", "wins", "losses", "draws", "lr"}.issubset(record)
    assert "eval@7" in format_human_event(record)


def test_log_checkpoint_and_early_stop_shape() -> None:
    checkpoint = log_checkpoint(
        phase="phase3",
        variant="recurrent",
        update=9,
        total_updates=100,
        path="runs/x/ckpt_0009.pt",
    )
    early_stop = log_early_stop(
        phase="phase3",
        variant="recurrent",
        update=9,
        total_updates=100,
        reason="stagnated",
    )

    assert checkpoint["event"] == "checkpoint"
    assert {"path", "phase", "variant", "update", "total_updates"}.issubset(checkpoint)
    assert "checkpoint@9" in format_human_event(checkpoint)

    assert early_stop["event"] == "early_stop"
    assert {"reason", "phase", "variant", "update", "total_updates"}.issubset(early_stop)
    assert "early-stop" in format_human_event(early_stop)
