"""Phase registry shared across training entrypoints.

Phase numbers are progress metadata; env construction is NOT phase-specific.
Each phase's env bundle translates the phase-shaped YAML into the canonical
env-config dict and hands it to ``envs.runtime_factory.mappo_env_fn_from_config``
— the same single factory the explicit ``env.kind`` path uses — so every
backend (including ``sim_pool``, which introspects the factory's partials)
works identically for both config styles.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from envs.runtime_factory import mappo_env_fn_from_config
from train.runtime_specs import base_env_cfg, resolve_seed_base
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM


def _phase4_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    base_cfg = base_env_cfg(env_cfg)
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
    # The canonical factory config carries explicitly what the phase number
    # used to imply (actor_obs, n_agents). self_play_schedule/snapshot_league
    # are keyed on presence in the config — independent of self_play, because
    # snapshot_league also seeds the snapshot-opponent pool.
    factory_env_cfg: dict = {
        **base_cfg,
        "actor_obs": actor_obs,
        "n_agents": 6 if self_play else 3,
        "self_play": {"enabled": self_play},
    }
    if mini_game is not None:
        factory_env_cfg["mini_game"] = str(mini_game)
        factory_env_cfg["mini_game_config"] = mini_game_cfg
    if "self_play_schedule" in env_cfg:
        factory_env_cfg["self_play_schedule"] = self_play_schedule
    if "snapshot_league" in env_cfg:
        factory_env_cfg["snapshot_league"] = snapshot_league
    return (
        mappo_env_fn_from_config(factory_env_cfg),
        ckpt_env_cfg,
        resolve_seed_base(env_cfg, base_cfg["sim"]),
    )


def _phase11_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    sim_cfg = dict(env_cfg.get("sim", {}))
    reward_cfg = dict(env_cfg.get("reward", {}))
    fog_cfg = {
        "fog_mode": str(env_cfg.get("fog_mode", "team_shared")),
        "visible_radius": float(env_cfg.get("visible_radius", 0.65)),
    }
    schedule_present = "self_play_schedule" in env_cfg
    self_play_schedule = dict(env_cfg.get("self_play_schedule", {}))
    snapshot_league = dict(env_cfg.get("snapshot_league", {}))
    ckpt_env_cfg = {
        "sim": sim_cfg,
        "reward": reward_cfg,
        **fog_cfg,
        "map_randomization": dict(env_cfg.get("map_randomization", {})),
        "match_type": "current",
    }
    if schedule_present:
        ckpt_env_cfg["self_play_schedule"] = self_play_schedule
    if snapshot_league:
        ckpt_env_cfg["snapshot_league"] = snapshot_league
    factory_env_cfg = {
        **ckpt_env_cfg,
        "actor_obs": "multi_enemy_entity_grid",
        "n_agents": 6,
    }
    return (
        mappo_env_fn_from_config(factory_env_cfg),
        ckpt_env_cfg,
        resolve_seed_base(env_cfg, sim_cfg),
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
