"""Normalize old and new checkpoint runtime config shapes.

Checkpoints may store legacy phase metadata, explicit runtime metadata, or only
the compact env config used to reconstruct an eval environment. This module
keeps that compatibility logic in one place so eval/replay scripts do not need
to branch on numeric phase labels for runtime behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train.runtime_specs import RuntimeSpec, resolve_runtime_spec
from xushi2.entity_obs import ENTITY_OBS_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.grid_obs import ENTITY_GRID_OBS_DIM, MULTI_ENEMY_ENTITY_GRID_OBS_DIM


@dataclass(frozen=True)
class CheckpointRuntime:
    raw_config: dict[str, Any]
    phase_label: str
    phase_int: int | None
    env_cfg: dict[str, Any]
    mappo_cfg: dict[str, Any]
    runtime: RuntimeSpec

    @property
    def is_mappo(self) -> bool:
        return bool(self.mappo_cfg)

    @property
    def n_agents(self) -> int:
        if self.mappo_cfg:
            return int(self.mappo_cfg.get("n_agents", self.runtime.shapes.n_agents))
        return int(self.runtime.shapes.n_agents)

    @property
    def current_selfplay(self) -> bool:
        self_play_cfg = dict(self.env_cfg.get("self_play", {}))
        return bool(self_play_cfg.get("enabled", False))

    @property
    def six_agent_runtime(self) -> bool:
        return self.n_agents == 6

    @property
    def has_map_randomization(self) -> bool:
        return bool(self.env_cfg.get("map_randomization", {}))

    @property
    def has_target_slot(self) -> bool:
        return int(self.mappo_cfg.get("target_action_dim", 0)) > 0

    @property
    def has_fog(self) -> bool:
        fog = self.env_cfg.get("fog_mode", self.env_cfg.get("features", {}).get("fog"))
        return fog not in (None, "", "none")


def parse_phase_metadata(raw_phase: Any, *, has_mappo: bool) -> tuple[int | None, str]:
    if raw_phase is None:
        return (4 if has_mappo else 3), ("phase4" if has_mappo else "phase3")
    raw_text = str(raw_phase)
    try:
        phase_int = int(raw_text.removeprefix("phase"))
    except ValueError:
        phase_int = None
    phase_label = raw_text if raw_text.startswith("phase") else f"phase{raw_text}"
    return phase_int, phase_label


def checkpoint_runtime(ckpt_config: dict[str, Any]) -> CheckpointRuntime:
    has_mappo = "mappo" in ckpt_config
    phase_int, phase_label = parse_phase_metadata(
        ckpt_config.get("phase"), has_mappo=has_mappo
    )
    env_cfg = dict(ckpt_config.get("env", {}))
    mappo_cfg = dict(ckpt_config.get("mappo", {}))
    runtime_env_cfg = _runtime_env_cfg(env_cfg, mappo_cfg)
    runtime_config: dict[str, Any] = {"env": runtime_env_cfg}
    if "learner" in ckpt_config:
        runtime_config["learner"] = dict(ckpt_config["learner"])
    if "experiment" in ckpt_config:
        runtime_config["experiment"] = dict(ckpt_config["experiment"])
    elif phase_int is not None and "kind" not in runtime_env_cfg:
        runtime_config["phase"] = phase_int
    elif phase_label and "kind" not in runtime_env_cfg:
        runtime_config["phase"] = phase_label
    if "kind" in runtime_env_cfg and "learner" not in runtime_config and has_mappo:
        runtime_config["learner"] = {"kind": "mappo"}
    runtime = resolve_runtime_spec(runtime_config)
    return CheckpointRuntime(
        raw_config=dict(ckpt_config),
        phase_label=phase_label,
        phase_int=phase_int,
        env_cfg=env_cfg,
        mappo_cfg=mappo_cfg,
        runtime=runtime,
    )


def _runtime_env_cfg(env_cfg: dict[str, Any], mappo_cfg: dict[str, Any]) -> dict[str, Any]:
    if not mappo_cfg or "kind" in env_cfg:
        return dict(env_cfg)
    out = dict(env_cfg)
    out["kind"] = "mappo_match"
    out.setdefault("actor_obs", _actor_obs_from_mappo_cfg(mappo_cfg))
    out.setdefault("critic_obs", "team_global")
    out.setdefault("n_agents", int(mappo_cfg.get("n_agents", 3)))
    if int(mappo_cfg.get("target_action_dim", 0)) > 0:
        out.setdefault("target_slot", True)
    return out


def _actor_obs_from_mappo_cfg(mappo_cfg: dict[str, Any]) -> str:
    if int(mappo_cfg.get("target_action_dim", 0)) > 0:
        return "multi_enemy_entity_grid"
    obs_dim = int(mappo_cfg.get("obs_dim", 31))
    if obs_dim == 31:
        return "flat"
    if obs_dim == ENTITY_OBS_DIM:
        return "entity"
    if obs_dim == ENTITY_GRID_OBS_DIM:
        return "entity_grid"
    if obs_dim in (
        MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        MULTI_ENEMY_ENTITY_GRID_OBS_DIM + MULTI_ENEMY_TOKEN_COUNT,
    ):
        return "multi_enemy_entity_grid"
    raise ValueError(f"cannot infer mappo actor_obs from checkpoint obs_dim={obs_dim}")
