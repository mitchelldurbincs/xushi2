"""Phase registry shared across training entrypoints."""

from __future__ import annotations

from functools import partial
from typing import Callable

import gymnasium as gym

from xushi2.entity_obs import ENTITY_OBS_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.grid_obs import ENTITY_GRID_OBS_DIM, MULTI_ENEMY_ENTITY_GRID_OBS_DIM

PHASE10_TARGET_OBS_DIM = MULTI_ENEMY_ENTITY_GRID_OBS_DIM + MULTI_ENEMY_TOKEN_COUNT


def _make_phase2_env(episode_length: int, cue_visible_ticks: int):
    from envs.memory_toy import MemoryToyEnv

    return MemoryToyEnv(
        episode_length=episode_length,
        cue_visible_ticks=cue_visible_ticks,
    )


def _make_phase3_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase3_ranger import Phase3RangerEnv

    return Phase3RangerEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _make_phase4_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase4_mappo import Phase4MappoEnv

    return Phase4MappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _make_phase5_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase5_entity_mappo import Phase5EntityMappoEnv

    return Phase5EntityMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _make_phase6_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase6_grid_mappo import Phase6GridMappoEnv

    return Phase6GridMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _make_phase7_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
    fog_mode: str,
    visible_radius: float,
):
    from envs.phase7_fog_mappo import Phase7FogMappoEnv

    return Phase7FogMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
        fog_mode=fog_mode,
        visible_radius=visible_radius,
    )


def _make_phase8_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
    fog_mode: str,
    visible_radius: float,
    map_randomization: dict,
):
    from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv

    return Phase8RandomMapMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
        fog_mode=fog_mode,
        visible_radius=visible_radius,
        map_randomization=map_randomization,
    )


def _make_phase9_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
    fog_mode: str,
    visible_radius: float,
    map_randomization: dict,
    snapshot_paths: tuple[str, ...],
    snapshot_league: dict,
):
    from envs.phase9_snapshot_mappo import Phase9SnapshotMappoEnv

    return Phase9SnapshotMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
        fog_mode=fog_mode,
        visible_radius=visible_radius,
        map_randomization=map_randomization,
        snapshot_paths=list(snapshot_paths),
        snapshot_league=snapshot_league,
    )


def _make_phase10_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
    fog_mode: str,
    visible_radius: float,
    map_randomization: dict,
):
    from envs.phase10_target_slot_mappo import Phase10TargetSlotMappoEnv

    return Phase10TargetSlotMappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
        fog_mode=fog_mode,
        visible_radius=visible_radius,
        map_randomization=map_randomization,
    )


def _make_phase11_env(
    sim_cfg: dict,
    reward_cfg: dict,
    fog_mode: str,
    visible_radius: float,
    map_randomization: dict,
    self_play_schedule: dict | None,
    snapshot_league: dict,
):
    from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv

    return Phase11CurrentSelfplayMappoEnv(
        sim_cfg,
        reward_cfg=reward_cfg,
        fog_mode=fog_mode,
        visible_radius=visible_radius,
        map_randomization=map_randomization,
        self_play_schedule=self_play_schedule,
        snapshot_league=snapshot_league,
    )




def _extract_base_env_cfg(env_cfg: dict, *, opponent_default: str = "basic") -> dict:
    sim_cfg = dict(env_cfg.get("sim", {}))
    return {
        "sim": sim_cfg,
        "opponent_bot": str(env_cfg.get("opponent_bot", opponent_default)),
        "learner_team": str(env_cfg.get("learner_team", "A")),
        "reward": dict(env_cfg.get("reward", {})),
    }


def _extract_fog_env_cfg(env_cfg: dict, *, visible_radius_default: float = 0.65) -> dict:
    return {
        "fog_mode": str(env_cfg.get("fog_mode", "team_shared")),
        "visible_radius": float(env_cfg.get("visible_radius", visible_radius_default)),
    }


def _extract_map_randomization(env_cfg: dict) -> dict:
    return {"map_randomization": dict(env_cfg.get("map_randomization", {}))}


def _resolve_seed_base(env_cfg: dict, sim_cfg: dict) -> int:
    return int(env_cfg.get("seed_base", sim_cfg.get("seed", 0)))


def _phase2_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    ep_len = int(env_cfg.get("episode_length", 64))
    cue_ticks = int(env_cfg.get("cue_visible_ticks", 4))
    return (
        partial(_make_phase2_env, ep_len, cue_ticks),
        {"episode_length": ep_len, "cue_visible_ticks": cue_ticks},
        int(env_cfg.get("seed_base", 0)),
    )


def _phase3_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    return (
        partial(
            _make_phase3_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
        ),
        base_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase4_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    return (
        partial(
            _make_phase4_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
        ),
        base_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase5_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    return (
        partial(
            _make_phase5_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
        ),
        base_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase6_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    return (
        partial(
            _make_phase6_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
        ),
        base_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase7_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    fog_cfg = _extract_fog_env_cfg(env_cfg, visible_radius_default=0.6)
    ckpt_env_cfg = {**base_cfg, **fog_cfg}
    return (
        partial(
            _make_phase7_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
            fog_cfg["fog_mode"],
            fog_cfg["visible_radius"],
        ),
        ckpt_env_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase8_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    fog_cfg = _extract_fog_env_cfg(env_cfg)
    map_cfg = _extract_map_randomization(env_cfg)
    ckpt_env_cfg = {**base_cfg, **fog_cfg, **map_cfg}
    return (
        partial(
            _make_phase8_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
            fog_cfg["fog_mode"],
            fog_cfg["visible_radius"],
            map_cfg["map_randomization"],
        ),
        ckpt_env_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase9_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg, opponent_default="snapshot")
    fog_cfg = _extract_fog_env_cfg(env_cfg)
    map_cfg = _extract_map_randomization(env_cfg)
    snapshot_paths = tuple(str(p) for p in env_cfg.get("snapshot_paths", ()))
    snapshot_league = dict(env_cfg.get("snapshot_league", {}))
    self_play_schedule = dict(env_cfg.get("self_play_schedule", {}))
    ckpt_env_cfg = {
        **base_cfg,
        **fog_cfg,
        **map_cfg,
        "snapshot_paths": snapshot_paths,
        "snapshot_league": snapshot_league,
        "self_play_schedule": self_play_schedule,
    }
    return (
        partial(
            _make_phase9_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
            fog_cfg["fog_mode"],
            fog_cfg["visible_radius"],
            map_cfg["map_randomization"],
            snapshot_paths,
            snapshot_league,
        ),
        ckpt_env_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase10_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg, opponent_default="noop")
    fog_cfg = _extract_fog_env_cfg(env_cfg)
    map_cfg = _extract_map_randomization(env_cfg)
    ckpt_env_cfg = {**base_cfg, **fog_cfg, **map_cfg}
    return (
        partial(
            _make_phase10_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
            fog_cfg["fog_mode"],
            fog_cfg["visible_radius"],
            map_cfg["map_randomization"],
        ),
        ckpt_env_cfg,
        _resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase11_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    sim_cfg = dict(env_cfg.get("sim", {}))
    reward_cfg = dict(env_cfg.get("reward", {}))
    fog_cfg = _extract_fog_env_cfg(env_cfg)
    map_cfg = _extract_map_randomization(env_cfg)
    schedule_present = "self_play_schedule" in env_cfg
    self_play_schedule = dict(env_cfg.get("self_play_schedule", {}))
    snapshot_league = dict(env_cfg.get("snapshot_league", {}))
    ckpt_env_cfg = {
        "sim": sim_cfg,
        "reward": reward_cfg,
        **fog_cfg,
        **map_cfg,
        "match_type": "current",
    }
    if schedule_present:
        ckpt_env_cfg["self_play_schedule"] = self_play_schedule
    if snapshot_league:
        ckpt_env_cfg["snapshot_league"] = snapshot_league
    return (
        partial(
            _make_phase11_env,
            sim_cfg,
            reward_cfg,
            fog_cfg["fog_mode"],
            fog_cfg["visible_radius"],
            map_cfg["map_randomization"],
            self_play_schedule if schedule_present else None,
            snapshot_league,
        ),
        ckpt_env_cfg,
        _resolve_seed_base(env_cfg, sim_cfg),
    )


def _phase0_seed(config: dict) -> int:
    env_cfg = config.get("env", {})
    sim_cfg = config.get("sim", {})
    return int(env_cfg.get("seed_base", sim_cfg.get("seed", 0)))


PHASE_REGISTRY: dict[int, dict] = {
    0: {
        "label": "phase0",
        "training_variants": (),
        "seed_deriver": _phase0_seed,
    },
    2: {
        "label": "phase2",
        "obs_dim": 3,
        "action_dim": 2,
        "continuous_action_dim": 2,
        "binary_action_dim": 0,
        "training_variants": ("recurrent", "feedforward"),
        "env_bundle": _phase2_env_bundle,
    },
    3: {
        "label": "phase3",
        "obs_dim": 31,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("recurrent",),
        "env_bundle": _phase3_env_bundle,
    },
    4: {
        "label": "phase4",
        "obs_dim": 31,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase4_env_bundle,
    },
    5: {
        "label": "phase5",
        "obs_dim": ENTITY_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase5_env_bundle,
    },
    6: {
        "label": "phase6",
        "obs_dim": ENTITY_GRID_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase6_env_bundle,
    },
    7: {
        "label": "phase7",
        "obs_dim": MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase7_env_bundle,
    },
    8: {
        "label": "phase8",
        "obs_dim": MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase8_env_bundle,
    },
    9: {
        "label": "phase9",
        "obs_dim": MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase9_env_bundle,
    },
    10: {
        "label": "phase10",
        "obs_dim": PHASE10_TARGET_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 7,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "target_action_dim": MULTI_ENEMY_TOKEN_COUNT,
        "training_variants": ("mappo",),
        "env_bundle": _phase10_env_bundle,
    },
    11: {
        "label": "phase11",
        "obs_dim": MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        "critic_obs_dim": 135,
        "n_agents": 6,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase11_env_bundle,
    },
}


def resolve_phase(config: dict) -> tuple[int, dict]:
    raw_phase = config.get("phase", 2)
    try:
        phase = int(raw_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported phase/config shape: phase={raw_phase!r}") from exc

    spec = PHASE_REGISTRY.get(phase)
    if spec is None:
        raise ValueError(f"unsupported phase/config shape: phase={raw_phase!r}")
    return phase, spec
