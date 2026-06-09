"""Dump a greedy eval episode to a text replay file the viewer can replay.

Usage:
    python -m scripts.dump_replay \
        --checkpoint runs/.../ckpt_0600.pt \
        --output ../data/replays/phase4_eval.replay \
        --seed 0xD1CEDA7A

Replay format (ASCII, line-delimited):
    Line 1: header — space-separated ``key=value`` pairs. Required keys:
        format, seed, round_seconds, action_repeat,
        mech_dmg, mech_fcd, mech_hbr, mech_resp
    MAPPO lines: one decision per line, 37 numeric fields:
        tick, then six action slots of
        mx my aim_delta_rad primary_fire ability_1 ability_2.
    Phase 10+ lines append target_slot per action slot, for 43 fields total.
    Phase 4 replay dumping currently requires a noop scripted opponent so the
    enemy-team slots are exact zero actions.

The viewer reads the header to construct an identical ``MatchConfig`` and
then drives a fresh ``Sim`` with the per-decision actions; the replay
relies on Phase-0 determinism rather than dumping full state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_dump.rollout import dump_mappo, load_mappo_checkpoint
from train.runtime_adapter import resolve_runtime_env_factory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump a greedy eval episode for the viewer to replay"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of consecutive episodes to dump"
    )
    parser.add_argument(
        "--max-decisions", type=int, default=None, help="Optional cap for quick smoke dumps"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy distribution "
        "instead of greedy. Reflects training-time behavior.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    ckpt_config = dict(raw_ckpt.get("config", {}))
    if "mappo" not in ckpt_config:
        raise ValueError("replay dumping only supports MAPPO checkpoints")

    model, ckpt_config = load_mappo_checkpoint(args.checkpoint)
    resolve_runtime_env_factory(
        ckpt_config,
        require_learner="mappo",
        context="MAPPO replay dump",
    )
    n_decisions = dump_mappo(
        model,
        ckpt_config,
        seed=int(args.seed),
        episodes=int(args.episodes),
        max_decisions=args.max_decisions,
        output_path=output_path,
        stochastic=bool(args.stochastic),
    )

    print(f"[dump_replay] wrote {n_decisions} decisions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
