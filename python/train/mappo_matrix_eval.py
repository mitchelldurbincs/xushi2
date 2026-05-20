from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from train.mappo_eval_gate_io import read_json_artifact, write_json_artifact
from train.mappo_evaluate import evaluate_mappo
from train.mappo_model import MappoActorCritic, MappoEvalStats
from train.checkpoint_runtime import checkpoint_runtime
from xushi2.mappo_matrix_gate import check_matrix_gate


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
    def from_dict(cls, payload: dict) -> "MatrixEvalConfig":
        return cls(
            episodes=int(payload.get("episodes", 1)),
            anchor_bots=tuple(str(bot) for bot in payload.get("anchor_bots", ())),
            opponent_checkpoints=tuple(str(p) for p in payload.get("opponent_checkpoints", ())),
            current_selfplay=bool(payload.get("current_selfplay", True)),
            output=str(payload.get("output", "matrix_eval.json")),
            gate=dict(payload.get("gate", {})),
            gate_output=str(payload.get("gate_output", "matrix_gate.json")),
        )


def matrix_native_bot_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    bot: str,
    mappo_cfg: dict | None = None,
):
    env_cfg = dict(ckpt_env_cfg.values)
    env_cfg.pop("self_play", None)
    env_cfg.pop("match_type", None)
    env_cfg.pop("snapshot_paths", None)
    env_cfg.pop("snapshot_league", None)
    env_cfg.pop("self_play_schedule", None)
    env_cfg.update({"opponent_bot": str(bot), "learner_team": "A"})
    env_cfg["n_agents"] = 3
    runtime = checkpoint_runtime({"env": env_cfg, "mappo": dict(mappo_cfg or {})}).runtime
    if runtime.env_fn is None:
        raise ValueError("matrix native-bot eval requires an environment runtime")
    return runtime.env_fn


def matrix_snapshot_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    snapshot_path: str,
    mappo_cfg: dict | None = None,
):
    env_cfg = dict(ckpt_env_cfg.values)
    env_cfg.update({"opponent_bot": "snapshot", "learner_team": "A", "snapshot_paths": [snapshot_path], "snapshot_league": {"latest": [snapshot_path], "weights": {"latest": 1.0}}})
    runtime = checkpoint_runtime({"env": env_cfg, "mappo": dict(mappo_cfg or {})}).runtime
    if runtime.env_fn is None:
        raise ValueError("matrix snapshot eval requires an environment runtime")
    return runtime.env_fn


def mappo_matrix_row(*, learner: str, opponent: str, opponent_type: str, stats: MappoEvalStats) -> dict:
    e = max(1, int(stats.episodes))
    return {"learner": learner, "opponent": opponent, "opponent_type": opponent_type, "episodes": int(stats.episodes), "win_rate": float(stats.wins)/e, "loss_rate": float(stats.losses)/e, "draw_rate": float(stats.draws)/e, "mean_reward": float(stats.mean_reward), "mean_score_a": float(stats.mean_team_a_score), "mean_score_b": float(stats.mean_team_b_score), "mean_kills_a": float(stats.mean_team_a_kills), "mean_kills_b": float(stats.mean_team_b_kills), "mean_final_tick": float(stats.mean_final_tick)}


def matrix_retention_summary(rows: list[dict], gate: dict | None = None) -> dict[str, float | int | bool | None]:
    if not rows:
        return {"matrix_score": 0.0, "matrix_rows": 0, "matrix_gate_passed": False if gate is not None else None}
    scores = [float(r.get("win_rate", 0.0)) - float(r.get("loss_rate", 0.0)) for r in rows]
    return {"matrix_score": float(np.mean(scores)), "matrix_rows": len(rows), "matrix_gate_passed": bool(gate.get("passed", False)) if gate is not None else None}


def matrix_gate_label(value: bool | None) -> str:
    return "ungated" if value is None else ("pass" if value else "fail")


def matrix_current_selfplay_env_fn(
    ckpt_env_cfg: CheckpointEnvConfig,
    mappo_cfg: dict | None = None,
):
    env_cfg = dict(ckpt_env_cfg.values)
    env_cfg["self_play_schedule"] = {"weights": {"current": 1.0, "snapshot": 0.0, "anchor": 0.0}}
    env_cfg["n_agents"] = 6
    runtime = checkpoint_runtime({"env": env_cfg, "mappo": dict(mappo_cfg or {})}).runtime
    if runtime.env_fn is None:
        raise ValueError("matrix current-selfplay eval requires an environment runtime")
    return runtime.env_fn


def run_mappo_matrix_eval(*, model: MappoActorCritic, phase: int, ckpt_env_cfg: CheckpointEnvConfig, matrix_cfg: MatrixEvalConfig, output_dir: Path, seed: int) -> list[dict]:
    rows: list[dict] = []
    mappo_cfg = dict(model.cfg.__dict__)
    if model.cfg.n_agents == 6 and matrix_cfg.current_selfplay:
        stats = evaluate_mappo(model, matrix_current_selfplay_env_fn(ckpt_env_cfg, mappo_cfg), episodes=matrix_cfg.episodes, seed=seed + 720_000)
        rows.append(mappo_matrix_row(learner="ckpt_final.pt", opponent="current", opponent_type="selfplay", stats=stats))
    for i, bot in enumerate(matrix_cfg.anchor_bots):
        stats = evaluate_mappo(model, matrix_native_bot_env_fn(ckpt_env_cfg, bot, mappo_cfg), episodes=matrix_cfg.episodes, seed=seed + 700_000 + 100*i)
        rows.append(mappo_matrix_row(learner="ckpt_final.pt", opponent=bot, opponent_type="bot", stats=stats))
    for i, opp in enumerate(matrix_cfg.opponent_checkpoints):
        stats = evaluate_mappo(model, matrix_snapshot_env_fn(ckpt_env_cfg, opp, mappo_cfg), episodes=matrix_cfg.episodes, seed=seed + 710_000 + 100*i)
        rows.append(mappo_matrix_row(learner="ckpt_final.pt", opponent=Path(opp).name, opponent_type="snapshot", stats=stats))
    if rows:
        write_json_artifact(output_dir / matrix_cfg.output, rows)
        if matrix_cfg.gate:
            gate = check_matrix_gate(rows, dict(matrix_cfg.gate))
            write_json_artifact(output_dir / matrix_cfg.gate_output, gate)
            if not gate["passed"]:
                raise RuntimeError("MAPPO matrix gate failed: " + "; ".join(gate["failures"]))
    return rows
