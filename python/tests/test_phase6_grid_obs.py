from __future__ import annotations

import numpy as np

from envs.phase6_grid_mappo import Phase6GridMappoEnv
from xushi2.grid_obs import (
    ENTITY_GRID_OBS_DIM,
    GRID_CHANNELS,
    GRID_FLAT_DIM,
    GRID_SIZE,
    actor_obs_to_entity_grid_obs,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice


def _make_sim_cfg(round_length: int = 5) -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": round_length,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def test_actor_obs_to_entity_grid_obs_shapes_and_marks_grid() -> None:
    obs = np.zeros((1, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[0, actor_field_slice("own_position")] = np.array([0.5, 0.0], dtype=np.float32)
    obs[0, actor_field_slice("enemy_relative_position")] = np.array(
        [-0.5, 0.0], dtype=np.float32
    )
    obs[0, actor_field_slice("enemy_alive")] = 1.0

    out = actor_obs_to_entity_grid_obs(obs)
    assert out.shape == (1, ENTITY_GRID_OBS_DIM)
    grid = out[0, -GRID_FLAT_DIM:].reshape(GRID_CHANNELS, GRID_SIZE, GRID_SIZE)
    assert grid[0].max() == 1.0
    assert grid[1].max() == 1.0
    assert grid[2].max() == 1.0


def test_phase6_env_returns_entity_grid_obs_and_phase4_critic_obs() -> None:
    env = Phase6GridMappoEnv(_make_sim_cfg(), opponent_bot="noop")
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (3, ENTITY_GRID_OBS_DIM)
        assert obs.dtype == np.float32
        assert info["learner_team"] == "A"

        critic_obs = np.zeros(CRITIC_DIM, dtype=np.float32)
        env.build_critic_obs(critic_obs)
        assert np.all(np.isfinite(critic_obs))

        next_obs, reward, term, trunc, _ = env.step(
            np.zeros((3, 6), dtype=np.float32)
        )
        assert next_obs.shape == (3, ENTITY_GRID_OBS_DIM)
        assert reward.shape == (3,)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
    finally:
        env.close()
