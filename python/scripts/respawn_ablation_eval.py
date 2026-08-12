"""Respawn-time ablation eval for Phase 4 MAPPO checkpoints.

Step 0 of docs/reports/2026-07-09-phase4-3v3-review-recommendations.md: run a
no-training matrix over ``mechanics.respawn_ticks`` values to test whether the
respawn treadmill (respawn 8s vs capture 8s) is what blocks conversion. If the
bridge checkpoint scores at respawn 2400 (no respawn inside a 60s round) but
not at 240, the respawn curriculum is the right lever.

Example (from python/):

    .venv/bin/python -m scripts.respawn_ablation_eval \
        --checkpoint ../data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt \
        --episodes 50 --output ../runs/respawn_ablation/respawn_ablation.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.mappo import MappoActorCritic, evaluate_mappo, mappo_config_from_checkpoint
from train.mappo_evaluate import eval_stats_dict
from train.runtime_adapter import resolve_runtime_env_factory

DEFAULT_CHECKPOINT = "data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt"
DEFAULT_RESPAWN_TICKS = (240, 720, 2400)


def _load_checkpoint(path: str | Path) -> tuple[MappoActorCritic, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    ckpt_config = ckpt.get("config", {})
    cfg = mappo_config_from_checkpoint(ckpt_config["mappo"])
    model = MappoActorCritic(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_config


def _env_fn(ckpt_config: dict, *, opponent_bot: str, respawn_ticks: int):
    env_cfg = copy.deepcopy(dict(ckpt_config.get("env", {})))
    env_cfg.setdefault("kind", "mappo_match")
    env_cfg["opponent_bot"] = str(opponent_bot)
    sim_cfg = dict(env_cfg.get("sim", {}))
    mechanics = dict(sim_cfg.get("mechanics", {}))
    mechanics["respawn_ticks"] = int(respawn_ticks)
    sim_cfg["mechanics"] = mechanics
    env_cfg["sim"] = sim_cfg
    _runtime, env_fn, _seed_base = resolve_runtime_env_factory(
        {
            "env": env_cfg,
            "mappo": dict(ckpt_config.get("mappo", {})),
            "learner": {"kind": "mappo"},
        },
        require_learner="mappo",
        context="respawn ablation eval",
    )
    return env_fn


def run_respawn_ablation(
    checkpoint: str,
    *,
    opponents: list[str],
    respawn_ticks: list[int],
    episodes: int,
    seed: int,
    objective_timing_seconds: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    model, ckpt_config = _load_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    for opp_idx, opponent in enumerate(opponents):
        for tick_idx, ticks in enumerate(respawn_ticks):
            env_fn = _env_fn(ckpt_config, opponent_bot=opponent, respawn_ticks=int(ticks))
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=int(episodes),
                seed=int(seed) + 10_000 * opp_idx + 100 * tick_idx,
                objective_timing_seconds=objective_timing_seconds,
            )
            rows.append(
                {
                    "checkpoint": Path(checkpoint).name,
                    "opponent": str(opponent),
                    "respawn_ticks": int(ticks),
                    "respawn_seconds": float(ticks) / 30.0,
                    **eval_stats_dict(stats),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Phase 4 MAPPO checkpoint across respawn_ticks values"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--respawn-ticks", action="append", type=int, default=[])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0x9E5F0A47)
    parser.add_argument(
        "--unlock-seconds",
        type=float,
        default=None,
        help="Optional objective unlock override; requires --capture-seconds",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=None,
        help="Optional objective capture override; requires --unlock-seconds",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if (args.unlock_seconds is None) != (args.capture_seconds is None):
        raise ValueError("--unlock-seconds and --capture-seconds must be set together")
    timing = (
        None
        if args.unlock_seconds is None
        else (float(args.unlock_seconds), float(args.capture_seconds))
    )
    opponents = [str(b) for b in args.opponent] or ["weak_basic_v2"]
    respawn_ticks = [int(t) for t in args.respawn_ticks] or list(DEFAULT_RESPAWN_TICKS)

    rows = run_respawn_ablation(
        str(args.checkpoint),
        opponents=opponents,
        respawn_ticks=respawn_ticks,
        episodes=int(args.episodes),
        seed=int(args.seed),
        objective_timing_seconds=timing,
    )
    for row in rows:
        print(
            "[respawn_ablation] "
            f"opponent={row['opponent']} "
            f"respawn={row['respawn_ticks']}t({row['respawn_seconds']:.0f}s) "
            f"wins={row['wins']}/{row['episodes']} "
            f"losses={row['losses']}/{row['episodes']} "
            f"draws={row['draws']}/{row['episodes']} "
            f"score={row['mean_score_a']:.2f}/{row['mean_score_b']:.2f} "
            f"kills={row['mean_kills_a']:.1f}/{row['mean_kills_b']:.1f} "
            f"uncont={row['mean_uncontested_on_point_seconds_a']:.2f}/"
            f"{row['mean_uncontested_on_point_seconds_b']:.2f} "
            f"cap_gain={row['mean_cap_progress_gain_ticks']:.1f} "
            f"cap_loss={row['mean_cap_progress_loss_ticks']:.1f} "
            f"hit_fire={row['team_a_hit_fire']:.4f}",
            flush=True,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"[respawn_ablation] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
