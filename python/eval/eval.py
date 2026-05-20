"""Evaluation entrypoint.

Phase 0: runs scripted-vs-scripted matches and prints summary.
With --dump-golden, writes per-decision hash trajectory to stdout.

Compatibility: legacy --dump-obs / --dump-reward flags now delegate to
`eval.dump_env_trajectory`.
"""

from __future__ import annotations

import argparse

from eval.dump_env_trajectory import dump_env_trajectory
from eval.sim_cfg import add_sim_cfg_args, build_sim_cfg_from_args
from xushi2.env import VALID_OPPONENT_BOTS
from xushi2.runner import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 evaluation entrypoint")
    parser.add_argument(
        "--policy", type=str, default=None, help="Policy checkpoint path (unused in Phase 0)"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=None)
    parser.add_argument("--bot-a", type=str, default="basic")
    parser.add_argument("--bot-b", type=str, default="basic")
    parser.add_argument("--dump-golden", action="store_true", help="Print hash trajectory")

    # Backward-compatibility flags: env dump moved to eval.dump_env_trajectory.
    parser.add_argument("--dump-obs", type=str, default=None)
    parser.add_argument("--dump-reward", type=str, default=None)
    parser.add_argument("--opponent-bot", type=str, default=None, choices=sorted(VALID_OPPONENT_BOTS))
    parser.add_argument("--learner-team", type=str, default="A", choices=("A", "B"))

    add_sim_cfg_args(parser)
    args = parser.parse_args()

    sim_cfg = build_sim_cfg_from_args(args)
    seed_base = sim_cfg["seed"]

    if args.dump_obs is not None or args.dump_reward is not None:
        if args.opponent_bot is None:
            parser.error("--opponent-bot is required when --dump-obs or --dump-reward is set")
        dump_env_trajectory(
            sim_cfg=sim_cfg,
            opponent_bot=args.opponent_bot,
            learner_team=args.learner_team,
            seed=seed_base,
            obs_path=args.dump_obs,
            reward_path=args.dump_reward,
        )
        return 0

    for ep_idx in range(args.episodes):
        run_seed = seed_base + ep_idx
        r = run_episode(sim_cfg, args.bot_a, args.bot_b, seed_override=run_seed)
        if args.dump_golden:
            for h in r.decision_hashes:
                print(f"{h:016x}")
        else:
            winner_str = {0: "draw", 1: "A", 2: "B"}.get(r.winner, "?")
            print(
                f"episode={ep_idx} seed=0x{run_seed:x} "
                f"decisions={len(r.decision_hashes)} final_tick={r.final_tick} "
                f"kills=A{r.team_a_kills}/B{r.team_b_kills} winner={winner_str} "
                f"first_hash=0x{r.decision_hashes[0]:016x} "
                f"last_hash=0x{r.decision_hashes[-1]:016x}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
