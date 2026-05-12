"""Env trajectory CSV dump CLI.

Owns env-mode dumping previously hosted in eval.py via --dump-obs/--dump-reward.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack

from eval.sim_cfg import add_mechanics_args, build_sim_cfg_from_args
from xushi2.env import VALID_OPPONENT_BOTS, XushiEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM


def _zero_action() -> dict:
    return {
        "move_x": 0.0,
        "move_y": 0.0,
        "aim_delta": 0.0,
        "primary_fire": 0,
        "ability_1": 0,
        "ability_2": 0,
    }


def dump_env_trajectory(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    seed: int,
    obs_path: str | None,
    reward_path: str | None,
) -> None:
    env = XushiEnv(sim_cfg, opponent_bot=opponent_bot, learner_team=learner_team)
    obs, _ = env.reset(seed=seed)

    with ExitStack() as stack:
        obs_writer = None
        reward_writer = None
        if obs_path is not None:
            obs_file = stack.enter_context(open(obs_path, "w", newline=""))
            obs_writer = csv.writer(obs_file)
            obs_writer.writerow(["tick", *(f"f{i}" for i in range(ACTOR_PHASE1_DIM))])
        if reward_path is not None:
            reward_file = stack.enter_context(open(reward_path, "w", newline=""))
            reward_writer = csv.writer(reward_file)
            reward_writer.writerow(
                ["tick", "step_reward_learner", "reward_team_a", "reward_team_b"]
            )

        try:
            while True:
                action = _zero_action()
                obs, reward, terminated, truncated, info = env.step(action)
                if obs_writer is not None:
                    obs_writer.writerow([info["tick"], *obs.tolist()])
                if reward_writer is not None:
                    reward_writer.writerow(
                        [
                            info["tick"],
                            reward,
                            info["reward_team_a"],
                            info["reward_team_b"],
                        ]
                    )
                if terminated or truncated:
                    break
        finally:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 env trajectory dump")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--round-length-seconds", type=int, default=30)
    parser.add_argument("--dump-obs", type=str, default=None)
    parser.add_argument("--dump-reward", type=str, default=None)
    parser.add_argument("--opponent-bot", type=str, required=True, choices=sorted(VALID_OPPONENT_BOTS))
    parser.add_argument("--learner-team", type=str, default="A", choices=("A", "B"))
    add_mechanics_args(parser)
    args = parser.parse_args()

    if args.dump_obs is None and args.dump_reward is None:
        parser.error("at least one of --dump-obs or --dump-reward is required")

    dump_env_trajectory(
        sim_cfg=build_sim_cfg_from_args(args),
        opponent_bot=args.opponent_bot,
        learner_team=args.learner_team,
        seed=args.seed,
        obs_path=args.dump_obs,
        reward_path=args.dump_reward,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
