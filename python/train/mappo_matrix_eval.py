from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from train.checkpoint_runtime import checkpoint_runtime
from train.runtime_adapter import resolve_runtime_env_factory
from train.mappo_eval_gate_io import read_json_artifact as read_json_artifact
from train.mappo_eval_gate_io import write_json_artifact
from train.mappo_evaluate import evaluate_mappo
from train.mappo_model import MappoActorCritic, MappoEvalStats
from xushi2.mappo_matrix_gate import check_matrix_gate

_CURRENT_SELFPLAY_SEED_OFFSET = 720_000
_ANCHOR_BOT_SEED_OFFSET = 700_000
_SNAPSHOT_SEED_OFFSET = 710_000
_MATCHUP_SEED_STRIDE = 100
_SELFPLAY_FIELDS = (
    "self_play",
    "match_type",
    "snapshot_paths",
    "snapshot_league",
    "self_play_schedule",
)


@dataclass(frozen=True)
class CheckpointEnvConfig:
    values: dict


@dataclass(frozen=True)
class MatrixEvalConfig:
    episodes: int = 1
    anchor_bots: tuple[str, ...] = ()
    opponent_checkpoints: tuple[str, ...] = ()
    current_selfplay: bool = True
    output: str = "matrix_eval.json"
    gate: dict = field(default_factory=dict)
    gate_output: str = "matrix_gate.json"

    @classmethod
    def from_dict(cls, payload: dict) -> MatrixEvalConfig:
        return cls(
            episodes=int(payload.get("episodes", 1)),
            anchor_bots=tuple(str(bot) for bot in payload.get("anchor_bots", ())),
            opponent_checkpoints=tuple(
                str(p) for p in payload.get("opponent_checkpoints", ())
            ),
            current_selfplay=bool(payload.get("current_selfplay", True)),
            output=str(payload.get("output", "matrix_eval.json")),
            gate=dict(payload.get("gate", {})),
            gate_output=str(payload.get("gate_output", "matrix_gate.json")),
        )


def _without_selfplay_fields(env_cfg: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(env_cfg)
    for field_name in _SELFPLAY_FIELDS:
        cleaned.pop(field_name, None)
    return cleaned


def _native_bot_env_cfg(
    ckpt_env_cfg: CheckpointEnvConfig,
    bot: str,
    *,
    learner_team: str = "A",
) -> dict[str, Any]:
    env_cfg = _without_selfplay_fields(ckpt_env_cfg.values)
    env_cfg.update(
        {
            "opponent_bot": str(bot),
            "learner_team": learner_team,
            "n_agents": 3,
        }
    )
    return env_cfg


def _snapshot_env_cfg(
    ckpt_env_cfg: CheckpointEnvConfig,
    snapshot_path: str,
) -> dict[str, Any]:
    env_cfg = dict(ckpt_env_cfg.values)
    env_cfg.update(
        {
            "opponent_bot": "snapshot",
            "learner_team": "A",
            "snapshot_paths": [snapshot_path],
            "snapshot_league": {
                "latest": [snapshot_path],
                "weights": {"latest": 1.0},
            },
        }
    )
    return env_cfg


def _current_selfplay_env_cfg(ckpt_env_cfg: CheckpointEnvConfig) -> dict[str, Any]:
    env_cfg = dict(ckpt_env_cfg.values)
    env_cfg["self_play_schedule"] = {
        "weights": {
            "current": 1.0,
            "snapshot": 0.0,
            "anchor": 0.0,
        }
    }
    env_cfg["n_agents"] = 6
    return env_cfg


def _checkpoint_runtime_env_fn(
    env_cfg: dict[str, Any],
    mappo_cfg: dict | None,
    *,
    error_message: str,
):
    env_cfg = dict(env_cfg)
    env_cfg.setdefault("kind", "mappo_match")
    _runtime, env_fn, _seed_base = resolve_runtime_env_factory(
        {
            "env": env_cfg,
            "mappo": dict(mappo_cfg or {}),
            "learner": {"kind": "mappo"},
        },
        require_learner="mappo",
        context=error_message,
    )
    return env_fn


def matrix_native_bot_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    bot: str,
    mappo_cfg: dict | None = None,
):
    env_cfg = _native_bot_env_cfg(ckpt_env_cfg, bot)
    return _checkpoint_runtime_env_fn(
        env_cfg,
        mappo_cfg,
        error_message="matrix native-bot eval requires an environment runtime",
    )


def matrix_snapshot_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    snapshot_path: str,
    mappo_cfg: dict | None = None,
):
    env_cfg = _snapshot_env_cfg(ckpt_env_cfg, snapshot_path)
    return _checkpoint_runtime_env_fn(
        env_cfg,
        mappo_cfg,
        error_message="matrix snapshot eval requires an environment runtime",
    )


def mappo_matrix_row(
    *,
    learner: str,
    opponent: str,
    opponent_type: str,
    stats: MappoEvalStats,
) -> dict:
    episodes = int(stats.episodes)
    episode_denominator = max(1, episodes)
    return {
        "learner": learner,
        "opponent": opponent,
        "opponent_type": opponent_type,
        "episodes": episodes,
        "win_rate": float(stats.wins) / episode_denominator,
        "loss_rate": float(stats.losses) / episode_denominator,
        "draw_rate": float(stats.draws) / episode_denominator,
        "mean_reward": float(stats.mean_reward),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
        "mean_kills_a": float(stats.mean_team_a_kills),
        "mean_kills_b": float(stats.mean_team_b_kills),
        "mean_final_tick": float(stats.mean_final_tick),
    }


def matrix_retention_summary(
    rows: list[dict],
    gate: dict | None = None,
) -> dict[str, float | int | bool | None]:
    if not rows:
        return {
            "matrix_score": 0.0,
            "matrix_rows": 0,
            "matrix_gate_passed": False if gate is not None else None,
        }
    scores = [
        float(row.get("win_rate", 0.0)) - float(row.get("loss_rate", 0.0))
        for row in rows
    ]
    return {
        "matrix_score": float(np.mean(scores)),
        "matrix_rows": len(rows),
        "matrix_gate_passed": (
            bool(gate.get("passed", False)) if gate is not None else None
        ),
    }


def matrix_gate_label(value: bool | None) -> str:
    return "ungated" if value is None else ("pass" if value else "fail")


def matrix_current_selfplay_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    mappo_cfg: dict | None = None,
):
    env_cfg = _current_selfplay_env_cfg(ckpt_env_cfg)
    return _checkpoint_runtime_env_fn(
        env_cfg,
        mappo_cfg,
        error_message="matrix current-selfplay eval requires an environment runtime",
    )


def _evaluate_matrix_row(
    *,
    model: MappoActorCritic,
    env_fn,
    episodes: int,
    seed: int,
    learner: str,
    opponent: str,
    opponent_type: str,
) -> dict:
    stats = evaluate_mappo(
        model,
        env_fn,
        episodes=episodes,
        seed=seed,
    )
    return mappo_matrix_row(
        learner=learner,
        opponent=opponent,
        opponent_type=opponent_type,
        stats=stats,
    )


def _write_matrix_outputs(
    *,
    output_dir: Path,
    matrix_cfg: MatrixEvalConfig,
    rows: list[dict],
) -> None:
    if not rows:
        return

    write_json_artifact(output_dir / matrix_cfg.output, rows)
    if not matrix_cfg.gate:
        return

    gate = check_matrix_gate(rows, dict(matrix_cfg.gate))
    write_json_artifact(output_dir / matrix_cfg.gate_output, gate)
    if not gate["passed"]:
        raise RuntimeError("MAPPO matrix gate failed: " + "; ".join(gate["failures"]))


def run_mappo_matrix_eval(
    *,
    model: MappoActorCritic,
    phase: int,
    ckpt_env_cfg: CheckpointEnvConfig,
    matrix_cfg: MatrixEvalConfig,
    output_dir: Path,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    mappo_cfg = dict(model.cfg.__dict__)

    if model.cfg.n_agents == 6 and matrix_cfg.current_selfplay:
        rows.append(
            _evaluate_matrix_row(
                model=model,
                env_fn=matrix_current_selfplay_env_fn(ckpt_env_cfg, mappo_cfg),
                episodes=matrix_cfg.episodes,
                seed=seed + _CURRENT_SELFPLAY_SEED_OFFSET,
                learner="ckpt_final.pt",
                opponent="current",
                opponent_type="selfplay",
            )
        )

    for index, bot in enumerate(matrix_cfg.anchor_bots):
        rows.append(
            _evaluate_matrix_row(
                model=model,
                env_fn=matrix_native_bot_env_fn(ckpt_env_cfg, bot, mappo_cfg),
                episodes=matrix_cfg.episodes,
                seed=seed + _ANCHOR_BOT_SEED_OFFSET + _MATCHUP_SEED_STRIDE * index,
                learner="ckpt_final.pt",
                opponent=bot,
                opponent_type="bot",
            )
        )

    for index, opponent_checkpoint in enumerate(matrix_cfg.opponent_checkpoints):
        rows.append(
            _evaluate_matrix_row(
                model=model,
                env_fn=matrix_snapshot_env_fn(
                    ckpt_env_cfg,
                    opponent_checkpoint,
                    mappo_cfg,
                ),
                episodes=matrix_cfg.episodes,
                seed=seed + _SNAPSHOT_SEED_OFFSET + _MATCHUP_SEED_STRIDE * index,
                learner="ckpt_final.pt",
                opponent=Path(opponent_checkpoint).name,
                opponent_type="snapshot",
            )
        )

    _write_matrix_outputs(
        output_dir=output_dir,
        matrix_cfg=matrix_cfg,
        rows=rows,
    )
    return rows
