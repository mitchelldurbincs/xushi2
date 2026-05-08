"""Drive Phase4MappoEnv with a hardcoded walk-to-objective policy.

This is a Phase-4 sanity check independent of learning: Team A derives
movement from actor obs `own_position` and walks to the objective against a
noop Team B. A healthy env should produce Team A score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.obs_manifest import actor_field_slice


def _sim_cfg(round_length_seconds: int) -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def _write_replay_header(fh: TextIO, *, seed: int, round_length_seconds: int) -> None:
    fh.write(
        "xushi2_text_replay "
        f"seed={seed} "
        f"round_seconds={round_length_seconds} "
        "action_repeat=3 "
        "team_size=3 "
        "mech_dmg=7500 "
        "mech_fcd=15 "
        "mech_hbr=0.75 "
        "mech_resp=240\n"
    )


def _write_replay_decision(fh: TextIO, *, tick: int, learner_action: np.ndarray) -> None:
    # Text viewer format: tick followed by six slots, each
    # move_x move_y aim_delta primary_fire ability_1 ability_2.
    slots = np.zeros((6, 6), dtype=np.float32)
    slots[:3, :] = learner_action
    fields = " ".join(f"{float(v):.6f}" for v in slots.reshape(-1))
    fh.write(f"{tick} {fields}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round-length", type=int, default=30)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument(
        "--dump-replay",
        type=Path,
        default=None,
        help="write a text replay consumable by src/viewer --replay",
    )
    args = parser.parse_args()

    env = Phase4MappoEnv(_sim_cfg(args.round_length), opponent_bot="noop")
    pos_slice = actor_field_slice("own_position")
    obs, info = env.reset(seed=args.seed)
    replay_fh = None
    try:
        if args.dump_replay is not None:
            args.dump_replay.parent.mkdir(parents=True, exist_ok=True)
            replay_fh = args.dump_replay.open("w", encoding="utf-8")
            _write_replay_header(
                replay_fh,
                seed=args.seed,
                round_length_seconds=args.round_length,
            )
        for _ in range(args.max_decisions):
            own_pos = obs[:, pos_slice]
            move = -own_pos.copy()
            norm = np.linalg.norm(move, axis=1, keepdims=True)
            move = np.where(norm > 0.02, move / np.maximum(norm, 1e-6), 0.0)
            action = np.zeros((3, 6), dtype=np.float32)
            action[:, :2] = move.astype(np.float32)
            if replay_fh is not None:
                _write_replay_decision(
                    replay_fh,
                    tick=int(info["tick"]),
                    learner_action=action,
                )
            obs, _reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
    finally:
        if replay_fh is not None:
            replay_fh.close()
        env.close()

    print(
        "phase4_walk_objective "
        f"tick={info['tick']} winner={info['winner']} "
        f"score={info['team_a_score']:.2f}/{info['team_b_score']:.2f} "
        f"kills={info['team_a_kills']}/{info['team_b_kills']}"
    )
    if args.dump_replay is not None:
        print(f"replay={args.dump_replay}")
    return 0 if float(info["team_a_score"]) > 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
