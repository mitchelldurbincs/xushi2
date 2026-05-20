from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.ppo_recurrent.orchestration import make_ppo_config
from train.ppo_recurrent.trainer import PPOTrainer
from train.runtime_specs import resolve_runtime_spec
from train.train import load_config


@dataclass(frozen=True)
class BenchMetadata:
    python_version: str
    torch_version: str
    cpu_count: int
    git_sha: str | None


@dataclass(frozen=True)
class BenchAggregate:
    repeat_index: int
    seed: int
    warmup_iterations: int
    measured_iterations: int
    env_steps_per_iteration: int
    total_samples_processed: int
    rollout_wall_time_sec: float
    update_wall_time_sec: float
    total_wall_time_sec: float
    env_steps_per_sec: float
    learner_steps_per_sec: float


@dataclass(frozen=True)
class BenchResult:
    target: str
    config_path: str
    vector_env: str
    metadata: BenchMetadata
    runs: list[BenchAggregate]


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _metadata() -> BenchMetadata:
    return BenchMetadata(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cpu_count=os.cpu_count() or 0,
        git_sha=_git_sha(),
    )


def _build_trainers(config: dict, target: str, seed: int, vector_env: str):
    runtime = resolve_runtime_spec(config)
    if runtime.env_fn is None:
        raise ValueError(f"benchmark target requires an env runtime, got env={runtime.env.kind!r}")
    env_fn = runtime.env_fn
    seed_base = runtime.seed_base
    run_seed = int(seed_base) + int(seed)

    if target in ("ppo_recurrent", "env_step_only", "update_only"):
        ppo_cfg = dict(config.get("ppo", {}))
        ppo_cfg["vector_env"] = vector_env
        local_cfg = dict(config)
        local_cfg["ppo"] = ppo_cfg
        trainer = PPOTrainer(
            env_fn,
            make_ppo_config(local_cfg, use_recurrence=(target == "ppo_recurrent")),
            seed=run_seed,
        )
        return trainer, "ppo"

    if target == "mappo":
        if runtime.learner.kind != "mappo":
            raise ValueError(
                f"mappo target requires learner.kind='mappo', got {runtime.learner.kind!r}"
            )
        ppo_cfg = dict(config.get("ppo", {}))
        ppo_cfg["vector_env"] = vector_env
        local_cfg = dict(config)
        local_cfg["ppo"] = ppo_cfg
        trainer = MappoTrainer(env_fn, make_mappo_config(local_cfg), seed=run_seed)
        return trainer, "mappo"

    raise ValueError(f"unsupported target: {target}")


def _run_once(
    *,
    config: dict,
    target: str,
    warmup_iterations: int,
    measured_iterations: int,
    seed: int,
    repeat_index: int,
    vector_env: str,
) -> BenchAggregate:
    trainer, trainer_kind = _build_trainers(config, target, seed, vector_env)
    cfg = trainer.config if trainer_kind == "ppo" else trainer.cfg
    env_steps_per_iteration = int(cfg.num_envs) * int(cfg.rollout_len)

    try:
        for _ in range(warmup_iterations):
            if target == "env_step_only":
                trainer.collect_rollout()
            elif target == "update_only":
                rollout = trainer.collect_rollout()
                trainer.update(rollout)
            else:
                rollout = trainer.collect_rollout()
                trainer.update(rollout)

        rollout_time = 0.0
        update_time = 0.0
        total_samples = 0
        for _ in range(measured_iterations):
            t0 = time.perf_counter()
            rollout = trainer.collect_rollout()
            t1 = time.perf_counter()
            if target != "env_step_only":
                trainer.update(rollout)
            t2 = time.perf_counter()
            if target != "update_only":
                rollout_time += t1 - t0
            if target != "env_step_only":
                update_time += t2 - t1
            total_samples += env_steps_per_iteration

        total_time = rollout_time + update_time
        env_sps = total_samples / rollout_time if rollout_time > 0 else 0.0
        learner_sps = total_samples / update_time if update_time > 0 else 0.0
        return BenchAggregate(
            repeat_index=repeat_index,
            seed=seed,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            env_steps_per_iteration=env_steps_per_iteration,
            total_samples_processed=total_samples,
            rollout_wall_time_sec=rollout_time / measured_iterations,
            update_wall_time_sec=(update_time / measured_iterations) if measured_iterations > 0 else 0.0,
            total_wall_time_sec=total_time / measured_iterations,
            env_steps_per_sec=env_sps,
            learner_steps_per_sec=learner_sps,
        )
    finally:
        if hasattr(trainer, "envs"):
            trainer.envs.close()
        if hasattr(trainer, "close"):
            trainer.close()


def _emit(result: BenchResult, output_format: str) -> None:
    payload = asdict(result)
    if output_format == "json":
        print(json.dumps(payload, indent=2))
        return
    if output_format == "csv":
        fields = list(payload["runs"][0].keys()) if payload["runs"] else []
        writer = csv.DictWriter(sys.stdout, fieldnames=["target", "config_path", "vector_env", *fields])
        writer.writeheader()
        for row in payload["runs"]:
            writer.writerow({"target": result.target, "config_path": result.config_path, "vector_env": result.vector_env, **row})
        return
    print(f"benchmark target={result.target} config={result.config_path} vector_env={result.vector_env}")
    print(f"python={result.metadata.python_version} torch={result.metadata.torch_version} cpus={result.metadata.cpu_count} git_sha={result.metadata.git_sha}")
    for run in result.runs:
        print(
            f"repeat={run.repeat_index} seed={run.seed} samples={run.total_samples_processed} "
            f"rollout_s={run.rollout_wall_time_sec:.6f} update_s={run.update_wall_time_sec:.6f} "
            f"env_sps={run.env_steps_per_sec:.2f} learner_sps={run.learner_steps_per_sec:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 benchmark entrypoint")
    parser.add_argument("--target", choices=["ppo_recurrent", "mappo", "env_step_only", "update_only"], required=True)
    parser.add_argument("--config", type=Path, required=True, help="Path to a phase config YAML")
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--vector-env", choices=["sync", "async"], default="sync")
    parser.add_argument("--output", choices=["summary", "json", "csv"], default="summary")
    args = parser.parse_args()

    config = load_config(args.config)
    runs: list[BenchAggregate] = []
    for i in range(int(args.repeat)):
        runs.append(
            _run_once(
                config=config,
                target=str(args.target),
                warmup_iterations=int(args.warmup_iterations),
                measured_iterations=int(args.measured_iterations),
                seed=int(args.seed) + i,
                repeat_index=i,
                vector_env=str(args.vector_env),
            )
        )

    result = BenchResult(
        target=str(args.target),
        config_path=str(args.config),
        vector_env=str(args.vector_env),
        metadata=_metadata(),
        runs=runs,
    )
    _emit(result, str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
