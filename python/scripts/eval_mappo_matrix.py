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
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.mappo import MappoActorCritic, MappoConfig, evaluate_mappo
from train.mappo_evaluate import eval_stats_dict
from train.runtime_specs import resolve_runtime_spec


def _load_checkpoint(path: str | Path) -> tuple[MappoActorCritic, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    ckpt_config = ckpt.get("config", {})
    cfg = MappoConfig(**ckpt_config["mappo"])
    model = MappoActorCritic(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_config


def _phase4_current_selfplay_checkpoint(model: MappoActorCritic, ckpt_config: dict) -> bool:
    env_cfg = dict(ckpt_config.get("env", {}))
    self_play_cfg = dict(env_cfg.get("self_play", {}))
    return (
        int(ckpt_config.get("phase", 4)) == 4
        and int(model.cfg.n_agents) == 6
        and bool(self_play_cfg.get("enabled", False))
    )


def _native_bot_eval_model(
    model: MappoActorCritic,
    ckpt_config: dict,
) -> MappoActorCritic:
    if int(model.cfg.n_agents) == 3:
        return model
    if not _phase4_current_selfplay_checkpoint(model, ckpt_config):
        raise ValueError(
            "native bot matrix eval supports 3-agent checkpoints and Phase 4 "
            f"current-self-play checkpoints; got n_agents={model.cfg.n_agents}"
        )

    adapted_cfg = replace(
        model.cfg,
        n_agents=3,
        value_per_agent=False,
        agent_loss_mask=(),
    )
    adapted = MappoActorCritic(adapted_cfg)
    adapted.load_state_dict(model.state_dict())
    adapted.eval()
    return adapted


def _native_bot_env_fn(ckpt_config: dict, bot: str, *, learner_team: str = "A"):
    phase = int(ckpt_config.get("phase", 4))
    if phase == 9:
        phase = 8
    if learner_team not in ("A", "B"):
        raise ValueError(f"learner_team must be A or B, got {learner_team!r}")
    env_cfg = dict(ckpt_config.get("env", {}))
    env_cfg.pop("self_play", None)
    env_cfg.pop("match_type", None)
    env_cfg["opponent_bot"] = str(bot)
    env_cfg["learner_team"] = learner_team
    runtime = resolve_runtime_spec({"phase": phase, "env": env_cfg})
    if runtime.learner.kind != "mappo" or runtime.env_fn is None:
        raise ValueError(
            "native bot matrix eval requires a MAPPO runtime, "
            f"got learner={runtime.learner.kind!r} env={runtime.env.kind!r}"
        )
    return runtime.env_fn


def _snapshot_env_fn(ckpt_config: dict, snapshot_path: str):
    env_cfg = dict(ckpt_config.get("env", {}))
    env_cfg["opponent_bot"] = "snapshot"
    env_cfg["learner_team"] = "A"
    env_cfg["snapshot_paths"] = [snapshot_path]
    env_cfg["snapshot_league"] = {
        "latest": [snapshot_path],
        "weights": {"latest": 1.0},
    }
    runtime = resolve_runtime_spec({"phase": 9, "env": env_cfg})
    if runtime.learner.kind != "mappo" or runtime.env_fn is None:
        raise ValueError(
            "snapshot matrix eval requires a MAPPO runtime, "
            f"got learner={runtime.learner.kind!r} env={runtime.env.kind!r}"
        )
    return runtime.env_fn


def _result_row(
    *,
    learner: str,
    opponent: str,
    opponent_type: str,
    learner_team: str,
    stats,
) -> dict[str, Any]:
    metrics = eval_stats_dict(stats)
    return {
        "learner": learner,
        "learner_team": learner_team,
        "opponent": opponent,
        "opponent_type": opponent_type,
        **metrics,
    }


def evaluate_matrix(
    checkpoints: list[str],
    *,
    anchor_bots: list[str],
    opponent_checkpoints: list[str],
    episodes: int,
    seed: int,
    learner_team: str = "A",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learner_idx, checkpoint in enumerate(checkpoints):
        model, ckpt_config = _load_checkpoint(checkpoint)
        label = Path(checkpoint).name
        for bot_idx, bot in enumerate(anchor_bots):
            eval_model = _native_bot_eval_model(model, ckpt_config)
            env_fn = _native_bot_env_fn(ckpt_config, bot, learner_team=learner_team)
            stats = evaluate_mappo(
                eval_model,
                env_fn,
                episodes=int(episodes),
                seed=int(seed) + 10_000 * learner_idx + 100 * bot_idx,
            )
            rows.append(
                _result_row(
                    learner=label,
                    opponent=str(bot),
                    opponent_type="bot",
                    learner_team=learner_team,
                    stats=stats,
                )
            )
        for opp_idx, opponent in enumerate(opponent_checkpoints):
            if int(model.cfg.n_agents) != 3:
                raise ValueError(
                    "snapshot matrix eval currently requires a 3-agent learner checkpoint; "
                    f"got n_agents={model.cfg.n_agents}"
                )
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
                    learner_team="A",
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
    parser.add_argument("--learner-team", choices=["A", "B"], default="A")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = evaluate_matrix(
        [str(p) for p in args.checkpoint],
        anchor_bots=[str(b) for b in args.anchor_bot],
        opponent_checkpoints=[str(p) for p in args.opponent_checkpoint],
        episodes=int(args.episodes),
        seed=int(args.seed),
        learner_team=str(args.learner_team),
    )
    if not rows:
        raise ValueError("no matchups requested; pass --anchor-bot or --opponent-checkpoint")
    for row in rows:
        print(
            "[mappo_matrix] "
            f"learner={row['learner']} "
            f"team={row['learner_team']} "
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
