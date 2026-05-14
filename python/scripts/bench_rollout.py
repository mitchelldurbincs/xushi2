"""Isolate mappo rollout timing (no update). Not part of the test suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.phases import resolve_phase
from train.train import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--vector-env", choices=["sync", "async"], default="async")
    args = parser.parse_args()

    config = load_config(args.config)
    ppo_cfg = dict(config.get("ppo", {}))
    ppo_cfg["vector_env"] = args.vector_env
    config["ppo"] = ppo_cfg
    phase, phase_spec = resolve_phase(config)
    env_fn, _ckpt_env_cfg, seed_base = phase_spec["env_bundle"](config)
    cfg = make_mappo_config(config)

    out_runs = []
    for r in range(args.repeat):
        trainer = MappoTrainer(env_fn, cfg, seed=int(seed_base) + r)
        try:
            for _ in range(args.warmup):
                trainer.collect_rollout()
            t0 = time.perf_counter()
            for _ in range(args.measured):
                trainer.collect_rollout()
            dt = time.perf_counter() - t0
        finally:
            trainer.close()
        env_steps = cfg.num_envs * cfg.rollout_len * args.measured
        sps = env_steps / dt if dt > 0 else 0.0
        per_iter = dt / max(1, args.measured)
        out_runs.append(
            {
                "repeat": r,
                "elapsed_s": dt,
                "per_iter_s": per_iter,
                "env_sps": sps,
                "env_steps_per_iter": cfg.num_envs * cfg.rollout_len,
            }
        )
        print(
            f"repeat={r} vector_env={args.vector_env} "
            f"per_iter_s={per_iter:.4f} env_sps={sps:.1f}"
        )

    print(json.dumps({"runs": out_runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
