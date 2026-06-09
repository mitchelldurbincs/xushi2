"""Explicit runtime specs and legacy phase compatibility adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from envs.runtime_factory import mappo_env_fn_from_config
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM


@dataclass(frozen=True)
class ExperimentSpec:
    phase: int | None
    phase_label: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearnerSpec:
    kind: str
    training_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvSpec:
    kind: str
    actor_obs: str
    critic_obs: str | None
    team_size: int
    learner_team: str
    opponent_kind: str
    features: dict[str, Any]


@dataclass(frozen=True)
class ShapeSpec:
    obs_dim: int
    action_dim: int
    continuous_action_dim: int
    binary_action_dim: int
    critic_obs_dim: int | None = None
    n_agents: int = 1
    target_action_dim: int = 0


@dataclass(frozen=True)
class RuntimeSpec:
    experiment: ExperimentSpec
    learner: LearnerSpec
    env: EnvSpec
    shapes: ShapeSpec
    env_fn: Callable[[], gym.Env] | None
    ckpt_env_cfg: dict[str, Any]
    seed_base: int

    @property
    def phase_label(self) -> str:
        return self.experiment.phase_label

    @property
    def phase_int(self) -> int | None:
        return self.experiment.phase


def resolve_runtime_spec(config: dict[str, Any]) -> RuntimeSpec:
    if _has_explicit_runtime_spec(config):
        return _resolve_explicit_runtime_spec(config)
    return _resolve_legacy_phase_runtime_spec(config)


def _has_explicit_runtime_spec(config: dict[str, Any]) -> bool:
    learner = config.get("learner")
    env = config.get("env")
    return (
        isinstance(learner, dict) and "kind" in learner and isinstance(env, dict) and "kind" in env
    )


def _experiment_from_config(config: dict[str, Any], *, default_label: str) -> ExperimentSpec:
    exp_cfg = dict(config.get("experiment", {}))
    raw_phase = exp_cfg.get("phase", config.get("phase"))
    phase_int = _parse_phase_int(raw_phase)
    phase_label = str(raw_phase) if raw_phase is not None else default_label
    if phase_label.isdigit():
        phase_label = f"phase{phase_label}"
    tags = tuple(str(t) for t in exp_cfg.get("tags", ()))
    return ExperimentSpec(phase=phase_int, phase_label=phase_label, tags=tags)


def _parse_phase_int(raw_phase: Any) -> int | None:
    if raw_phase is None:
        return None
    if isinstance(raw_phase, str) and raw_phase.startswith("phase"):
        raw_phase = raw_phase.removeprefix("phase")
    try:
        return int(raw_phase)
    except (TypeError, ValueError):
        return None


def _resolve_seed_base(env_cfg: dict[str, Any], sim_cfg: dict[str, Any]) -> int:
    return int(env_cfg.get("seed_base", sim_cfg.get("seed", 0)))


def _base_env_cfg(env_cfg: dict[str, Any], *, opponent_default: str = "basic") -> dict[str, Any]:
    opponent_cfg = dict(env_cfg.get("opponent", {}))
    return {
        "sim": dict(env_cfg.get("sim", {})),
        "opponent_bot": str(
            env_cfg.get("opponent_bot", opponent_cfg.get("kind", opponent_default))
        ),
        "learner_team": str(env_cfg.get("learner_team", "A")),
        "reward": dict(env_cfg.get("reward", {})),
    }


def _resolve_explicit_runtime_spec(config: dict[str, Any]) -> RuntimeSpec:
    learner_cfg = dict(config.get("learner", {}))
    env_cfg = dict(config.get("env", {}))
    learner_kind = str(learner_cfg.get("kind", "mappo"))
    env_kind = str(env_cfg.get("kind"))
    if env_kind == "mappo_match":
        return _explicit_mappo_match_spec(config, learner_kind, env_cfg)
    raise ValueError(f"unsupported explicit env.kind {env_kind!r}")


def _explicit_mappo_match_spec(
    config: dict[str, Any], learner_kind: str, env_cfg: dict[str, Any]
) -> RuntimeSpec:
    base_cfg = _base_env_cfg(env_cfg)
    features_cfg = dict(env_cfg.get("features", {}))
    actor_obs = str(env_cfg.get("actor_obs", "flat"))
    fog_mode = str(env_cfg.get("fog_mode", features_cfg.get("fog", "none")))
    target_slot = bool(env_cfg.get("target_slot", features_cfg.get("target_slot", False)))
    map_randomization = dict(env_cfg.get("map_randomization", {}))
    snapshot_paths = tuple(str(p) for p in env_cfg.get("snapshot_paths", ()))
    self_play = bool(dict(env_cfg.get("self_play", {})).get("enabled", False))
    mini_game = env_cfg.get("mini_game")
    default_agents = (
        3 if mini_game == "cap_duel" else 6 if self_play else env_cfg.get("team_size", 3)
    )
    n_agents = int(env_cfg.get("n_agents", default_agents))
    shapes = _mappo_shapes(actor_obs=actor_obs, target_slot=target_slot, n_agents=n_agents)
    ckpt_env_cfg = {
        **base_cfg,
        "kind": "mappo_match",
        "actor_obs": actor_obs,
        "n_agents": n_agents,
    }
    if fog_mode != "none":
        ckpt_env_cfg["fog_mode"] = fog_mode
        ckpt_env_cfg["visible_radius"] = float(env_cfg.get("visible_radius", 0.65))
    if map_randomization:
        ckpt_env_cfg["map_randomization"] = map_randomization
    for key in (
        "mini_game",
        "mini_game_config",
        "self_play",
        "self_play_schedule",
        "snapshot_paths",
        "snapshot_league",
        "target_slot",
        "features",
        "team_size",
        "n_agents",
    ):
        if key in env_cfg:
            ckpt_env_cfg[key] = env_cfg[key]
    exp = _experiment_from_config(config, default_label="mappo_match")
    return RuntimeSpec(
        experiment=exp,
        learner=LearnerSpec(kind=learner_kind, training_variants=("mappo",)),
        env=EnvSpec(
            kind="mappo_match",
            actor_obs=actor_obs,
            critic_obs=str(env_cfg.get("critic_obs", "team_global")),
            team_size=int(env_cfg.get("team_size", 3)),
            learner_team=base_cfg["learner_team"],
            opponent_kind=base_cfg["opponent_bot"],
            features={
                **features_cfg,
                "fog": fog_mode,
                "map_randomization": bool(map_randomization),
                "snapshot": bool(snapshot_paths),
                "current_selfplay": self_play,
                "target_slot": target_slot,
            },
        ),
        shapes=shapes,
        env_fn=mappo_env_fn_from_config(ckpt_env_cfg),
        ckpt_env_cfg=ckpt_env_cfg,
        seed_base=_resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _mappo_shapes(*, actor_obs: str, target_slot: bool, n_agents: int) -> ShapeSpec:
    if target_slot:
        raise ValueError("target_slot actor observations were removed with phase10")
    if actor_obs == "flat":
        obs_dim = 31
    elif actor_obs in ("multi_enemy_entity_grid", "partial_entity_grid"):
        obs_dim = MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    else:
        raise ValueError(f"unsupported mappo actor_obs {actor_obs!r}")
    target_dim = 0
    return ShapeSpec(
        obs_dim=obs_dim,
        critic_obs_dim=135,
        n_agents=int(n_agents),
        action_dim=6,
        continuous_action_dim=3,
        binary_action_dim=3,
        target_action_dim=target_dim,
    )


def _resolve_legacy_phase_runtime_spec(config: dict[str, Any]) -> RuntimeSpec:
    from train.phases import resolve_phase

    phase, phase_spec = resolve_phase(config)
    label = str(phase_spec["label"])
    learner_kind = _legacy_learner_kind(phase_spec)
    env_fn = None
    ckpt_env_cfg: dict[str, Any] = {}
    seed_base = 0
    if "env_bundle" in phase_spec:
        env_fn, ckpt_env_cfg, seed_base = phase_spec["env_bundle"](config)
    elif "seed_deriver" in phase_spec:
        seed_base = int(phase_spec["seed_deriver"](config))
    shapes = ShapeSpec(
        obs_dim=int(phase_spec.get("obs_dim", 0)),
        critic_obs_dim=(
            None if "critic_obs_dim" not in phase_spec else int(phase_spec["critic_obs_dim"])
        ),
        n_agents=int(phase_spec.get("n_agents", 1)),
        action_dim=int(phase_spec.get("action_dim", 0)),
        continuous_action_dim=int(phase_spec.get("continuous_action_dim", 0)),
        binary_action_dim=int(phase_spec.get("binary_action_dim", 0)),
        target_action_dim=int(phase_spec.get("target_action_dim", 0)),
    )
    env_cfg = dict(config.get("env", {}))
    legacy_actor_obs = str(ckpt_env_cfg.get("actor_obs", _legacy_actor_obs(phase)))
    return RuntimeSpec(
        experiment=ExperimentSpec(phase=phase, phase_label=label, tags=(label,)),
        learner=LearnerSpec(
            kind=learner_kind,
            training_variants=tuple(str(v) for v in phase_spec.get("training_variants", ())),
        ),
        env=EnvSpec(
            kind=_legacy_env_kind(phase),
            actor_obs=legacy_actor_obs,
            critic_obs=("team_global" if shapes.critic_obs_dim is not None else None),
            team_size=int(env_cfg.get("team_size", shapes.n_agents)),
            learner_team=str(env_cfg.get("learner_team", ckpt_env_cfg.get("learner_team", "A"))),
            opponent_kind=str(
                env_cfg.get("opponent_bot", ckpt_env_cfg.get("opponent_bot", "basic"))
            ),
            features=_legacy_features(phase, env_cfg, ckpt_env_cfg, shapes),
        ),
        shapes=shapes,
        env_fn=env_fn,
        ckpt_env_cfg=dict(ckpt_env_cfg),
        seed_base=int(seed_base),
    )


def _legacy_learner_kind(phase_spec: dict[str, Any]) -> str:
    variants = tuple(str(v) for v in phase_spec.get("training_variants", ()))
    if "mappo" in variants:
        return "mappo"
    return "scripted_determinism"


def _legacy_env_kind(phase: int) -> str:
    if phase >= 4:
        return "mappo_match"
    return "scripted_determinism"


def _legacy_actor_obs(phase: int) -> str:
    if phase == 11:
        return "multi_enemy_entity_grid"
    return "flat"


def _legacy_features(
    phase: int,
    env_cfg: dict[str, Any],
    ckpt_env_cfg: dict[str, Any],
    shapes: ShapeSpec,
) -> dict[str, Any]:
    return {
        "fog": str(env_cfg.get("fog_mode", ckpt_env_cfg.get("fog_mode", "none"))),
        "map_randomization": bool(
            env_cfg.get("map_randomization", ckpt_env_cfg.get("map_randomization", {}))
        ),
        "snapshot": bool(env_cfg.get("snapshot_paths", ckpt_env_cfg.get("snapshot_paths", ()))),
        "current_selfplay": bool(dict(env_cfg.get("self_play", {})).get("enabled", phase == 11)),
        "target_slot": shapes.target_action_dim > 0,
    }
