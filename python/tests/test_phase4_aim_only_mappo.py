from __future__ import annotations

import numpy as np
import pytest

from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv
from train.phases import resolve_phase
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice


def test_aim_only_env_shapes_and_visible_target_obs() -> None:
    env = Phase4AimOnlyMappoEnv(episode_decisions=4)
    obs, info = env.reset(seed=123)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert info["aim_hit_rate"] == 0.0
    assert np.all(obs[:, actor_field_slice("enemy_alive")] == 1.0)
    rel = obs[:, actor_field_slice("enemy_relative_position")]
    assert np.all(np.linalg.norm(rel, axis=1) > 0.0)


def test_exact_aim_and_fire_gets_hits() -> None:
    env = Phase4AimOnlyMappoEnv(
        episode_decisions=2,
        target_angle_limit=0.75,
        hit_tolerance=0.02,
        resample_target_each_step=False,
    )
    env.reset(seed=7)
    action = np.zeros((3, 6), dtype=np.float32)
    action[:, 2] = env._target_norm
    action[:, 3] = 1.0
    _obs, reward, term, trunc, info = env.step(action)
    assert not term
    assert not trunc
    assert np.all(reward > 0.0)
    assert info["aim_hits"] == 3
    assert info["aim_fires"] == 3
    assert info["aim_hit_rate"] == pytest.approx(1.0)


def test_bad_aim_and_fire_gets_misses() -> None:
    env = Phase4AimOnlyMappoEnv(
        episode_decisions=2,
        target_angle_limit=0.25,
        hit_tolerance=0.02,
        resample_target_each_step=False,
    )
    env.reset(seed=7)
    action = np.zeros((3, 6), dtype=np.float32)
    action[:, 2] = np.clip(env._target_norm + 0.5, -1.0, 1.0)
    action[:, 3] = 1.0
    _obs, reward, _term, _trunc, info = env.step(action)
    assert np.all(reward < 0.0)
    assert info["aim_hits"] == 0
    assert info["aim_fires"] == 3
    assert info["aim_misses"] == 3


def test_build_critic_obs_shape_and_phase4_registry_routing() -> None:
    env = Phase4AimOnlyMappoEnv(episode_decisions=4)
    obs, _info = env.reset(seed=0)
    out = np.empty(CRITIC_DIM, dtype=np.float32)
    env.build_critic_obs(out)
    assert out.shape == (CRITIC_DIM,)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:ACTOR_PHASE1_DIM], obs[0])

    cfg = {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "mini_game": "aim_only",
            "mini_game_config": {"episode_decisions": 4},
            "sim": {
                "seed": 0,
                "round_length_seconds": 3,
                "fog_of_war_enabled": False,
                "randomize_map": False,
                "action_repeat": 3,
                "mechanics": {
                    "revolver_damage_centi_hp": 1000,
                    "revolver_fire_cooldown_ticks": 15,
                    "revolver_hitbox_radius": 0.75,
                    "respawn_ticks": 240,
                },
            },
        },
    }
    _phase, spec = resolve_phase(cfg)
    env_fn, env_meta, seed = spec["env_bundle"](cfg)
    routed = env_fn()
    assert isinstance(routed, Phase4AimOnlyMappoEnv)
    assert env_meta["mini_game"] == "aim_only"
    assert seed == 0
