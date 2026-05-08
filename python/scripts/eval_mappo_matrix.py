"""Evaluate MAPPO checkpoints against anchor bots and frozen snapshots.

This is a compact league diagnostic for snapshot/self-play phases. It keeps the
surface deliberately small: 3-agent MAPPO checkpoints can be evaluated as Team A
against scripted anchors via their native env stack, and against frozen snapshot
checkpoints via the Phase-9 snapshot-opponent env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.mappo import MappoActorCritic, MappoConfig, evaluate_mappo
from train.phases import resolve_phase


def _load_checkpoint(path: str | Path) -> tuple[MappoActorCritic, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    ckpt_config = ckpt.get("config", {})
    cfg = MappoConfig(**ckpt_config["mappo"])
    if cfg.n_agents != 3:
        raise ValueError(
            "eval_mappo_matrix currently supports 3-agent learner checkpoints; "
            f"got n_agents={cfg.n_agents}"
        )
    model = MappoActorCritic(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_config


def _native_bot_env_fn(ckpt_config: dict, bot: str):
    phase = int(ckpt_config.get("phase", 4))
    if phase == 9:
        phase = 8
    if phase not in (4, 5, 6, 7, 8, 10):
        raise ValueError(f"unsupported bot-eval phase {phase}")
    env_cfg = dict(ckpt_config.get("env", {}))
    env_cfg["opponent_bot"] = str(bot)
    env_cfg["learner_team"] = "A"
    _phase, spec = resolve_phase({"phase": phase, "env": env_cfg})
    env_fn, _meta, _seed = spec["env_bundle"]({"phase": phase, "env": env_cfg})
    return env_fn


def _snapshot_env_fn(ckpt_config: dict, snapshot_path: str):
    env_cfg = dict(ckpt_config.get("env", {}))
    env_cfg["opponent_bot"] = "snapshot"
    env_cfg["learner_team"] = "A"
    env_cfg["snapshot_paths"] = [snapshot_path]
    env_cfg["snapshot_league"] = {
        "latest": [snapshot_path],
        "weights": {"latest": 1.0},
    }
    _phase, spec = resolve_phase({"phase": 9, "env": env_cfg})
    env_fn, _meta, _seed = spec["env_bundle"]({"phase": 9, "env": env_cfg})
    return env_fn


def _result_row(
    *,
    learner: str,
    opponent: str,
    opponent_type: str,
    stats,
) -> dict[str, Any]:
    episodes = max(1, int(stats.episodes))
    return {
        "learner": learner,
        "opponent": opponent,
        "opponent_type": opponent_type,
        "episodes": int(stats.episodes),
        "win_rate": float(stats.wins) / float(episodes),
        "loss_rate": float(stats.losses) / float(episodes),
        "draw_rate": float(stats.draws) / float(episodes),
        "mean_reward": float(stats.mean_reward),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
        "mean_kills_a": float(stats.mean_team_a_kills),
        "mean_kills_b": float(stats.mean_team_b_kills),
        "mean_final_tick": float(stats.mean_final_tick),
    }


def evaluate_matrix(
    checkpoints: list[str],
    *,
    anchor_bots: list[str],
    opponent_checkpoints: list[str],
    episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learner_idx, checkpoint in enumerate(checkpoints):
        model, ckpt_config = _load_checkpoint(checkpoint)
        label = Path(checkpoint).name
        for bot_idx, bot in enumerate(anchor_bots):
            env_fn = _native_bot_env_fn(ckpt_config, bot)
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=int(episodes),
                seed=int(seed) + 10_000 * learner_idx + 100 * bot_idx,
            )
            rows.append(
                _result_row(
                    learner=label,
                    opponent=str(bot),
                    opponent_type="bot",
                    stats=stats,
                )
            )
        for opp_idx, opponent in enumerate(opponent_checkpoints):
            env_fn = _snapshot_env_fn(ckpt_config, opponent)
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=int(episodes),
                seed=int(seed) + 10_000 * learner_idx + 1_000 + 100 * opp_idx,
            )
            rows.append(
                _result_row(
                    learner=label,
                    opponent=Path(opponent).name,
                    opponent_type="snapshot",
                    stats=stats,
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO matchup matrix")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--anchor-bot", action="append", default=[])
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xE0A17)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = evaluate_matrix(
        [str(p) for p in args.checkpoint],
        anchor_bots=[str(b) for b in args.anchor_bot],
        opponent_checkpoints=[str(p) for p in args.opponent_checkpoint],
        episodes=int(args.episodes),
        seed=int(args.seed),
    )
    if not rows:
        raise ValueError("no matchups requested; pass --anchor-bot or --opponent-checkpoint")
    for row in rows:
        print(
            "[mappo_matrix] "
            f"learner={row['learner']} "
            f"opponent={row['opponent_type']}:{row['opponent']} "
            f"win={row['win_rate']:.3f} draw={row['draw_rate']:.3f} "
            f"reward={row['mean_reward']:+.3f} "
            f"score={row['mean_score_a']:.2f}/{row['mean_score_b']:.2f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
