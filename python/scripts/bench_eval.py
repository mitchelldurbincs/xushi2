"""One-shot timing for evaluate_mappo. Not part of the test suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train.mappo_evaluate import evaluate_mappo
from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import make_mappo_config
from train.runtime_specs import resolve_runtime_spec
from train.train import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend", choices=["sync", "async", "serial-legacy"], default="async"
    )
    parser.add_argument("--num-envs", type=int, default=0, help="0 = default")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = resolve_runtime_spec(config)
    if runtime.learner.kind != "mappo" or runtime.env_fn is None:
        raise ValueError(
            f"eval benchmark requires MAPPO runtime, got learner={runtime.learner.kind!r}"
        )
    env_fn = runtime.env_fn
    seed_base = runtime.seed_base
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    model.eval()

    timings: list[float] = []
    for r in range(args.repeat):
        t0 = time.perf_counter()
        if args.backend == "serial-legacy":
            from train.mappo_model import MappoEvalStats, _eval_outcome_counts
            import numpy as np

            rewards = []
            final_ticks = []
            wins = losses = draws = terminated_count = truncated_count = 0
            for i in range(int(args.episodes)):
                env = env_fn()
                try:
                    obs, _info = env.reset(seed=int(seed_base) + int(args.seed) + r + i)
                    h = model.init_hidden(model.cfg.n_agents)
                    done = term = trunc = False
                    ep_reward = 0.0
                    info = {}
                    while not done:
                        obs_t = torch.as_tensor(obs, dtype=torch.float32)
                        with torch.no_grad():
                            action, h = model.greedy_action(obs_t, h)
                        obs, reward, term, trunc, info = env.step(action.cpu().numpy())
                        ep_reward += float(np.mean(reward))
                        done = bool(term or trunc)
                    rewards.append(ep_reward)
                    final_ticks.append(int(info.get("tick", 0)))
                    terminated_count += int(bool(term))
                    truncated_count += int(bool(trunc))
                finally:
                    env.close()
            stats = MappoEvalStats(
                mean_reward=float(np.mean(rewards)) if rewards else 0.0,
                episodes=len(rewards),
                wins=wins,
                losses=losses,
                draws=draws,
                terminated=terminated_count,
                truncated=truncated_count,
                mean_final_tick=float(np.mean(final_ticks)) if final_ticks else 0.0,
                mean_team_a_score=0.0,
                mean_team_b_score=0.0,
                mean_team_a_kills=0.0,
                mean_team_b_kills=0.0,
            )
        else:
            kwargs = {"backend": args.backend}
            if args.num_envs > 0:
                kwargs["num_envs"] = int(args.num_envs)
            stats = evaluate_mappo(
                model=model,
                env_fn=env_fn,
                episodes=int(args.episodes),
                seed=int(seed_base) + int(args.seed) + r,
                **kwargs,
            )
        dt = time.perf_counter() - t0
        timings.append(dt)
        print(
            f"repeat={r} backend={args.backend} episodes={stats.episodes} "
            f"elapsed_s={dt:.3f} per_ep_s={dt / max(1, stats.episodes):.4f}"
        )

    out = {
        "backend": args.backend,
        "episodes": int(args.episodes),
        "num_envs": int(args.num_envs),
        "repeat": int(args.repeat),
        "timings_s": timings,
        "mean_s": sum(timings) / len(timings),
        "min_s": min(timings),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
