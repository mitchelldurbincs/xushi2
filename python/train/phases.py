"""Phase registry shared across training entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import gymnasium as gym

from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM


def _make_phase4_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
    mini_game: str | None = None,
    mini_game_cfg: dict | None = None,
    self_play: bool = False,
    self_play_schedule: dict | None = None,
    snapshot_league: dict | None = None,
    actor_obs: str = "flat",
):
    from envs.runtime_factory import make_mappo_match_env

    return make_mappo_match_env(
        sim_cfg=sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
        actor_obs=actor_obs,
        mini_game=mini_game,
        mini_game_cfg=mini_game_cfg,
        self_play=self_play,
        self_play_schedule=self_play_schedule,
        snapshot_league=snapshot_league,
        n_agents=6 if self_play else 3,
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
    from envs.runtime_factory import make_mappo_match_env

    return make_mappo_match_env(
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        actor_obs="multi_enemy_entity_grid",
        fog_mode=fog_mode,
        visible_radius=visible_radius,
        map_randomization=map_randomization,
        self_play_schedule=self_play_schedule,
        snapshot_league=snapshot_league,
        n_agents=6,
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


def _phase4_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = _extract_base_env_cfg(env_cfg)
    mini_game = env_cfg.get("mini_game")
    mini_game_cfg = dict(env_cfg.get("mini_game_config", {}))
    self_play = bool(dict(env_cfg.get("self_play", {})).get("enabled", False))
    self_play_schedule = dict(env_cfg.get("self_play_schedule", {}))
    snapshot_league = dict(env_cfg.get("snapshot_league", {}))
    actor_obs = str(env_cfg.get("actor_obs", "flat"))
    ckpt_env_cfg = dict(base_cfg)
    if actor_obs != "flat":
        ckpt_env_cfg["actor_obs"] = actor_obs
    if self_play:
        ckpt_env_cfg["self_play"] = {"enabled": True}
        ckpt_env_cfg["match_type"] = "current"
        if "self_play_schedule" in env_cfg:
            ckpt_env_cfg["self_play_schedule"] = self_play_schedule
        if "snapshot_league" in env_cfg:
            ckpt_env_cfg["snapshot_league"] = snapshot_league
    if mini_game is not None:
        ckpt_env_cfg["mini_game"] = str(mini_game)
        ckpt_env_cfg["mini_game_config"] = mini_game_cfg
    return (
        partial(
            _make_phase4_env,
            base_cfg["sim"],
            base_cfg["opponent_bot"],
            base_cfg["learner_team"],
            base_cfg["reward"],
            None if mini_game is None else str(mini_game),
            mini_game_cfg,
            self_play,
            self_play_schedule if "self_play_schedule" in env_cfg else None,
            snapshot_league if "snapshot_league" in env_cfg else None,
            actor_obs,
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
    env_cfg = dict(config.get("env", {}))
    if (
        phase == 4
        and bool(dict(env_cfg.get("self_play", {})).get("enabled", False))
        and env_cfg.get("mini_game") != "cap_duel"
    ):
        spec = dict(spec)
        spec["label"] = "phase4_selfplay"
        spec["n_agents"] = 6
    elif phase == 4 and bool(dict(env_cfg.get("self_play", {})).get("enabled", False)):
        spec = dict(spec)
        spec["label"] = "phase4_selfplay"
    elif phase == 4 and str(env_cfg.get("actor_obs", "flat")) == "multi_enemy_entity_grid":
        spec = dict(spec)
        spec["label"] = "phase4_multi_enemy_actor_obs"
        spec["obs_dim"] = MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    return phase, spec
