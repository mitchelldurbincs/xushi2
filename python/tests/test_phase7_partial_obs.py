from __future__ import annotations

import numpy as np

from xushi2.multi_enemy_obs import ENTITY_TOKEN_DIM, GRID_FLAT_DIM
from xushi2.partial_obs import ENTITY_TOKEN_COUNT
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice
from xushi2.partial_obs import actor_obs_to_partial_entity_grid_obs


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


def test_partial_obs_masks_enemy_token_and_grid_when_hidden() -> None:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("enemy_alive")] = 1.0
    obs[:, actor_field_slice("enemy_relative_position")] = np.array([0.9, 0.0], dtype=np.float32)

    out = actor_obs_to_partial_entity_grid_obs(obs, visible_radius=0.5, team_shared=False)
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = out[:, :token_width].reshape(3, ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    enemy_grid = out[:, -GRID_FLAT_DIM:].reshape(3, 3, 32, 32)[:, 2]
    assert np.all(tokens[:, 1, :] == 0.0)
    assert np.all(mask[:, 1] == 0.0)
    assert np.all(enemy_grid == 0.0)


def test_partial_obs_team_shared_unions_enemy_visibility() -> None:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("enemy_alive")] = 1.0
    obs[:, actor_field_slice("enemy_relative_position")] = np.array([0.9, 0.0], dtype=np.float32)
    obs[1, actor_field_slice("enemy_relative_position")] = np.array([0.1, 0.0], dtype=np.float32)

    out = actor_obs_to_partial_entity_grid_obs(obs, visible_radius=0.5, team_shared=True)
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    mask = out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    assert np.all(mask[:, 1] == 1.0)


def test_partial_obs_per_agent_keeps_visibility_local() -> None:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("enemy_alive")] = 1.0
    obs[:, actor_field_slice("enemy_relative_position")] = np.array([0.9, 0.0], dtype=np.float32)
    obs[1, actor_field_slice("enemy_relative_position")] = np.array([0.1, 0.0], dtype=np.float32)

    out = actor_obs_to_partial_entity_grid_obs(obs, visible_radius=0.5, team_shared=False)
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    mask = out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    assert mask[:, 1].tolist() == [0.0, 1.0, 0.0]


def test_partial_obs_line_of_sight_override_can_hide_near_enemy() -> None:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("enemy_alive")] = 1.0
    obs[:, actor_field_slice("enemy_relative_position")] = np.array([0.1, 0.0], dtype=np.float32)

    out = actor_obs_to_partial_entity_grid_obs(
        obs,
        visible_radius=0.5,
        team_shared=False,
        visible_override=np.array([True, False, True]),
    )
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    mask = out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    assert mask[:, 1].tolist() == [1.0, 0.0, 1.0]


def test_partial_obs_uses_last_seen_marker_when_enemy_hidden() -> None:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("enemy_alive")] = 1.0
    obs[:, actor_field_slice("own_position")] = np.array([0.1, 0.0], dtype=np.float32)
    obs[:, actor_field_slice("enemy_relative_position")] = np.array([0.9, 0.0], dtype=np.float32)
    last_seen = np.tile(np.array([0.3, 0.0], dtype=np.float32), (3, 1))

    out = actor_obs_to_partial_entity_grid_obs(
        obs,
        visible_radius=0.5,
        team_shared=False,
        last_seen_enemy_position=last_seen,
        last_seen_valid=np.array([True, False, True]),
    )
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = out[:, :token_width].reshape(3, ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    enemy_grid = out[:, -GRID_FLAT_DIM:].reshape(3, 3, 32, 32)[:, 2]

    assert mask[:, 1].tolist() == [1.0, 0.0, 1.0]
    assert tokens[0, 1, 7] == 0.0
    assert np.allclose(tokens[0, 1, 8:10], np.array([0.2, 0.0], dtype=np.float32))
    assert tokens[0, 1, 17] == 0.5
    assert np.max(enemy_grid[0]) == 0.5
    assert np.max(enemy_grid[1]) == 0.0


def test_partial_obs_hidden_enemy_live_fields_do_not_leak() -> None:
    obs_a = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs_a[:, actor_field_slice("own_position")] = np.array([0.1, -0.1], dtype=np.float32)
    obs_a[:, actor_field_slice("enemy_alive")] = 1.0
    obs_a[:, actor_field_slice("enemy_hp")] = 0.25
    obs_a[:, actor_field_slice("enemy_relative_position")] = np.array([0.2, 0.3], dtype=np.float32)
    obs_a[:, actor_field_slice("enemy_velocity")] = np.array([0.4, -0.5], dtype=np.float32)
    obs_a[:, actor_field_slice("enemy_on_point")] = 1.0
    obs_b = obs_a.copy()
    obs_b[:, actor_field_slice("enemy_hp")] = 1.0
    obs_b[:, actor_field_slice("enemy_relative_position")] = np.array([-0.7, 0.6], dtype=np.float32)
    obs_b[:, actor_field_slice("enemy_velocity")] = np.array([-0.3, 0.8], dtype=np.float32)
    obs_b[:, actor_field_slice("enemy_on_point")] = 0.0
    last_seen = np.tile(np.array([0.4, 0.2], dtype=np.float32), (3, 1))
    kwargs = {
        "visible_radius": 0.9,
        "team_shared": False,
        "visible_override": np.zeros(3, dtype=bool),
        "last_seen_enemy_position": last_seen,
        "last_seen_valid": np.ones(3, dtype=bool),
    }

    out_a = actor_obs_to_partial_entity_grid_obs(obs_a, **kwargs)
    out_b = actor_obs_to_partial_entity_grid_obs(obs_b, **kwargs)
    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = out_a[:, :token_width].reshape(3, ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    enemy_grid = out_a[:, -GRID_FLAT_DIM:].reshape(3, 3, 32, 32)[:, 2]

    assert np.allclose(out_a, out_b)
    assert np.all(tokens[:, 1, 6] == 0.0)
    assert np.all(tokens[:, 1, 7] == 0.0)
    assert np.allclose(tokens[:, 1, 8:10], np.array([0.3, 0.3], dtype=np.float32))
    assert np.all(tokens[:, 1, 10:17] == 0.0)
    assert np.all(tokens[:, 1, 17] == 0.5)
    assert np.max(enemy_grid) == 0.5
