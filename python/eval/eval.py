"""Evaluation entrypoint.

Phase 0: runs one scripted-vs-scripted match and prints a short summary.
With --dump-golden, writes the per-decision hash trajectory to stdout (one
hex hash per line) for committing as a golden-replay artifact.

Phase 1b: with --dump-obs / --dump-reward, runs the Gymnasium env with a
scripted opponent and writes obs / reward trajectories to CSV. These are
debugging tools — they do not change the Phase-0 hash dump behavior.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from xushi2.env import VALID_OPPONENT_BOTS, XushiEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM
from xushi2.runner import run_episode


def _zero_action() -> dict:
    return {
        "move_x": 0.0,
        "move_y": 0.0,
        "aim_delta": 0.0,
        "primary_fire": 0,
        "ability_1": 0,
        "ability_2": 0,
    }


def _dump_env_trajectory(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    seed: int,
    obs_path: str | None,
    reward_path: str | None,
) -> None:
    env = XushiEnv(sim_cfg, opponent_bot=opponent_bot, learner_team=learner_team)
    obs, _ = env.reset(seed=seed)

    obs_writer = None
    reward_writer = None
    obs_file = None
    reward_file = None
    if obs_path is not None:
        obs_file = open(obs_path, "w", newline="")
        obs_writer = csv.writer(obs_file)
        obs_writer.writerow(["tick"] + [f"f{i}" for i in range(ACTOR_PHASE1_DIM)])
    if reward_path is not None:
        reward_file = open(reward_path, "w", newline="")
        reward_writer = csv.writer(reward_file)
        reward_writer.writerow(
            ["tick", "step_reward_learner", "reward_team_a", "reward_team_b"])

    try:
        # Act as a zero-action learner — this is a trajectory-recording
        # smoke test, not a policy eval.
        while True:
            action = _zero_action()
            obs, reward, terminated, truncated, info = env.step(action)
            if obs_writer is not None:
                obs_writer.writerow([info["tick"]] + obs.tolist())
            if reward_writer is not None:
                reward_writer.writerow([
                    info["tick"], reward,
                    info["reward_team_a"], info["reward_team_b"],
                ])
            if terminated or truncated:
                break
    finally:
        if obs_file is not None:
            obs_file.close()
        if reward_file is not None:
            reward_file.close()
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 evaluation entrypoint")
    parser.add_argument("--policy", type=str, default=None,
                        help="Policy checkpoint path (unused in Phase 0)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--bot-a", type=str, default="basic")
    parser.add_argument("--bot-b", type=str, default="basic")
    parser.add_argument("--round-length-seconds", type=int, default=30)
    parser.add_argument("--dump-golden", action="store_true",
                        help="Print hash trajectory (one hex per line) to stdout")

    # Phase-1b env-mode flags. Enable obs/reward CSV dumps by running one
    # episode of the Gymnasium env against a scripted opponent.
    parser.add_argument("--dump-obs", type=str, default=None,
                        help="Path for per-decision actor-obs CSV (Phase 1b)")
    parser.add_argument("--dump-reward", type=str, default=None,
                        help="Path for per-decision reward CSV (Phase 1b)")
    parser.add_argument("--opponent-bot", type=str, default=None,
                        choices=sorted(VALID_OPPONENT_BOTS),
                        help="Scripted opponent for the env dump (Phase 1b)")
    parser.add_argument("--learner-team", type=str, default="A",
                        choices=("A", "B"),
                        help="Team controlled by the learner in the env dump "
                             "(Phase 1b)")

    # Phase-1 mechanics. Required — no defaults. Match the values in
    # experiments/configs/phase0_determinism.yaml to reproduce its golden.
    parser.add_argument("--revolver-damage-centi-hp", type=int, required=True)
    parser.add_argument("--revolver-fire-cooldown-ticks", type=int, required=True)
    parser.add_argument("--revolver-hitbox-radius", type=float, required=True)
    parser.add_argument("--respawn-ticks", type=int, required=True)
    args = parser.parse_args()

    sim_cfg = {
        "seed": args.seed,
        "round_length_seconds": args.round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "mechanics": {
            "revolver_damage_centi_hp": args.revolver_damage_centi_hp,
            "revolver_fire_cooldown_ticks": args.revolver_fire_cooldown_ticks,
            "revolver_hitbox_radius": args.revolver_hitbox_radius,
            "respawn_ticks": args.respawn_ticks,
        },
    }

    # Env-mode CSV dump path (Phase 1b). Opponent-bot is required here.
    if args.dump_obs is not None or args.dump_reward is not None:
        if args.opponent_bot is None:
            parser.error(
                "--opponent-bot is required when --dump-obs or --dump-reward "
                "is set")
        _dump_env_trajectory(
            sim_cfg=sim_cfg,
            opponent_bot=args.opponent_bot,
            learner_team=args.learner_team,
            seed=args.seed,
            obs_path=args.dump_obs,
            reward_path=args.dump_reward,
        )
        return 0

    # Phase-0 scripted-vs-scripted mode.
    for ep_idx in range(args.episodes):
        r = run_episode(sim_cfg, args.bot_a, args.bot_b,
                        seed_override=args.seed + ep_idx)
        if args.dump_golden:
            for h in r.decision_hashes:
                print(f"{h:016x}")
        else:
            winner_str = {0: "draw", 1: "A", 2: "B"}.get(r.winner, "?")
            print(f"episode={ep_idx} seed=0x{args.seed + ep_idx:x} "
                  f"decisions={len(r.decision_hashes)} final_tick={r.final_tick} "
                  f"kills=A{r.team_a_kills}/B{r.team_b_kills} winner={winner_str} "
                  f"first_hash=0x{r.decision_hashes[0]:016x} "
                  f"last_hash=0x{r.decision_hashes[-1]:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
