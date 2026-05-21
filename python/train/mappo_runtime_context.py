from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from train.mappo_rollout_trainer import make_mappo_config
from train.runtime_adapter import resolve_runtime_env_factory
from xushi2.snapshot_retention import SnapshotRetention


@dataclass(frozen=True)
class RuntimeContext:
    config: dict[str, Any]
    phase: int | None
    phase_label: str
    env_fn: Any
    ckpt_env_cfg: dict[str, Any]
    seed_base: int
    cfg: Any
    run_cfg: dict[str, Any]
    total_updates: int
    eval_every: int
    eval_episodes: int
    checkpoint_every: int
    output_dir: Path
    retention: SnapshotRetention | None
    objective_timing_enabled: bool
    objective_initial_unlock_seconds: float
    objective_initial_capture_seconds: float
    objective_final_unlock_seconds: float
    objective_final_capture_seconds: float
    objective_timing_anneal_updates: int
    objective_eval_canonical_every: int
    majority_on_point_initial: float
    majority_on_point_anneal_updates: int
    uncontested_on_point_initial: float
    uncontested_on_point_anneal_updates: int


def build_runtime_context(config: dict[str, Any]) -> RuntimeContext:
    runtime, env_fn, _seed_base = resolve_runtime_env_factory(
        config,
        require_learner="mappo",
        context="MAPPO runtime",
    )
    phase = runtime.phase_int
    phase_label = runtime.phase_label
    ckpt_env_cfg = runtime.ckpt_env_cfg
    seed_base = runtime.seed_base
    cfg = make_mappo_config(config)
    env_cfg = dict(config.get("env", {}))
    run_cfg = config.get("run", {})
    reward_cfg = dict(config.get("env", {}).get("reward", {}))
    objective_timing_cfg = dict(env_cfg.get("objective_timing_curriculum", {}))
    objective_timing_enabled = bool(objective_timing_cfg.get("enabled", False))
    sim_cfg = dict(env_cfg.get("sim", {}))

    def _sim_timing_seconds(field: str, default: float) -> float:
        nested = dict(sim_cfg.get("objective_timing", {}))
        seconds_key = f"objective_{field}_seconds"
        ticks_key = f"objective_{field}_ticks"
        if seconds_key in sim_cfg:
            return float(sim_cfg[seconds_key])
        if f"{field}_seconds" in nested:
            return float(nested[f"{field}_seconds"])
        if ticks_key in sim_cfg:
            return float(sim_cfg[ticks_key]) / 30.0
        if f"{field}_ticks" in nested:
            return float(nested[f"{field}_ticks"]) / 30.0
        return float(default)

    objective_initial_unlock_seconds = float(
        objective_timing_cfg.get(
            "initial_unlock_seconds", _sim_timing_seconds("unlock", 15.0)
        )
    )
    objective_initial_capture_seconds = float(
        objective_timing_cfg.get(
            "initial_capture_seconds", _sim_timing_seconds("capture", 8.0)
        )
    )
    objective_final_unlock_seconds = float(
        objective_timing_cfg.get("final_unlock_seconds", 15.0)
    )
    objective_final_capture_seconds = float(
        objective_timing_cfg.get("final_capture_seconds", 8.0)
    )
    objective_timing_anneal_updates = int(
        objective_timing_cfg.get("anneal_updates", 0)
    )
    objective_eval_canonical_every = int(
        objective_timing_cfg.get("eval_canonical_every", 0)
    )
    majority_on_point_initial = float(reward_cfg.get("majority_on_point_coef", 0.0))
    majority_on_point_anneal_updates = int(
        reward_cfg.get("majority_on_point_anneal_updates", 0)
    )
    uncontested_on_point_initial = float(
        reward_cfg.get("uncontested_on_point_coef", 0.0)
    )
    uncontested_on_point_anneal_updates = int(
        reward_cfg.get("uncontested_on_point_anneal_updates", 0)
    )
    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 10))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    output_dir = Path(str(run_cfg.get("output_dir", "runs/phase4_mappo"))) / "mappo"
    output_dir.mkdir(parents=True, exist_ok=True)
    retention: SnapshotRetention | None = None
    if run_cfg.get("snapshot_retention"):
        retention_cfg = dict(run_cfg.get("snapshot_retention", {}))
        env_cfg = config.get("env", {})
        retention = SnapshotRetention(
            output_dir / str(retention_cfg.get("manifest", "snapshot_league.json")),
            max_latest=int(retention_cfg.get("max_latest", 20)),
            preserve_best=int(retention_cfg.get("preserve_best", 3)),
            anchor_paths=tuple(
                retention_cfg.get("anchor_paths", env_cfg.get("snapshot_paths", ()))
            )
            if bool(retention_cfg.get("include_config_anchors", True))
            else (),
            weights=dict(
                retention_cfg.get(
                    "weights",
                    env_cfg.get("snapshot_league", {}).get(
                        "weights",
                        {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
                    ),
                )
            ),
        )
    return RuntimeContext(
        config=config,
        phase=phase,
        phase_label=phase_label,
        env_fn=env_fn,
        ckpt_env_cfg=dict(ckpt_env_cfg),
        seed_base=int(seed_base),
        cfg=cfg,
        run_cfg=dict(run_cfg),
        total_updates=total_updates,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        checkpoint_every=checkpoint_every,
        output_dir=output_dir,
        retention=retention,
        objective_timing_enabled=objective_timing_enabled,
        objective_initial_unlock_seconds=objective_initial_unlock_seconds,
        objective_initial_capture_seconds=objective_initial_capture_seconds,
        objective_final_unlock_seconds=objective_final_unlock_seconds,
        objective_final_capture_seconds=objective_final_capture_seconds,
        objective_timing_anneal_updates=objective_timing_anneal_updates,
        objective_eval_canonical_every=objective_eval_canonical_every,
        majority_on_point_initial=majority_on_point_initial,
        majority_on_point_anneal_updates=majority_on_point_anneal_updates,
        uncontested_on_point_initial=uncontested_on_point_initial,
        uncontested_on_point_anneal_updates=uncontested_on_point_anneal_updates,
    )
