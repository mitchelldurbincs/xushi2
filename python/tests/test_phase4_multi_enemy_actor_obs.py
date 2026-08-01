from __future__ import annotations

import numpy as np
import yaml

from envs.phase4_mappo import Phase4MappoEnv
from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv
from envs.runtime_factory import make_mappo_match_env
from tests._paths import config_path
from train.runtime_specs import resolve_runtime_spec
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import ENTITY_TOKEN_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    zero_masked_enemy_tokens,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice, critic_field_slice

_SELF_TOKEN = 0
_FIRST_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 4
_POSITION = slice(8, 10)
_AUX = 17


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


def _split(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = obs.reshape(-1, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
    token_width = MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = flat[:, :token_width].reshape(-1, MULTI_ENEMY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = flat[:, token_width : token_width + MULTI_ENEMY_TOKEN_COUNT]
    return tokens, mask


def _actor_obs() -> np.ndarray:
    obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
    obs[:, actor_field_slice("own_hp")] = 1.0
    obs[:, actor_field_slice("own_position")] = np.array([0.1, -0.2], dtype=np.float32)
    obs[:, actor_field_slice("own_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
    obs[:, actor_field_slice("objective_owner_onehot")] = np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    return obs


def _critic_obs() -> np.ndarray:
    critic = np.zeros((3, CRITIC_DIM), dtype=np.float32)
    for enemy_idx in range(3):
        critic[:, critic_field_slice(f"enemy{enemy_idx}/alive_flag")] = 1.0
        critic[:, critic_field_slice(f"enemy{enemy_idx}/hp_normalized")] = 1.0
        critic[:, critic_field_slice(f"enemy{enemy_idx}/world_position")] = np.array(
            [20.0 + 4.0 * enemy_idx, 25.0],
            dtype=np.float32,
        )
        critic[:, critic_field_slice(f"enemy{enemy_idx}/world_aim_unit")] = np.array(
            [0.0, -1.0],
            dtype=np.float32,
        )
    return critic


def test_zero_masked_multi_enemy_tokens_prevents_hidden_enemy_payload() -> None:
    visible = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    out = zero_masked_enemy_tokens(
        actor_obs_to_multi_enemy_entity_grid_obs(
            _actor_obs(),
            critic_obs=_critic_obs(),
            map_bounds={"min_x": 0.0, "min_y": 0.0, "max_x": 50.0, "max_y": 50.0},
            visible_radius=1.0,
            visible_override=visible,
        )
    )
    tokens, mask = _split(out)

    assert np.all(mask[:, _SELF_TOKEN] == 1.0)
    assert np.all(mask[:, _OBJECTIVE_TOKEN] == 1.0)
    np.testing.assert_array_equal(
        mask[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN], visible.astype(np.float32)
    )
    for row in range(3):
        for enemy_idx in range(3):
            token_idx = _FIRST_ENEMY_TOKEN + enemy_idx
            if not visible[row, enemy_idx]:
                np.testing.assert_array_equal(
                    tokens[row, token_idx], np.zeros(ENTITY_TOKEN_DIM, dtype=np.float32)
                )


def test_hidden_enemy_state_does_not_change_visible_multi_enemy_actor_obs() -> None:
    visible = np.array([[True, False, False]] * 3, dtype=bool)
    critic_a = _critic_obs()
    critic_b = critic_a.copy()
    critic_b[:, critic_field_slice("enemy1/hp_normalized")] = 0.25
    critic_b[:, critic_field_slice("enemy1/world_position")] = np.array(
        [5.0, 45.0], dtype=np.float32
    )
    critic_b[:, critic_field_slice("enemy1/world_velocity")] = np.array(
        [4.0, -3.0], dtype=np.float32
    )
    kwargs = {
        "map_bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 50.0, "max_y": 50.0},
        "visible_radius": 1.0,
        "visible_override": visible,
    }

    out_a = zero_masked_enemy_tokens(
        actor_obs_to_multi_enemy_entity_grid_obs(_actor_obs(), critic_obs=critic_a, **kwargs)
    )
    out_b = zero_masked_enemy_tokens(
        actor_obs_to_multi_enemy_entity_grid_obs(_actor_obs(), critic_obs=critic_b, **kwargs)
    )

    np.testing.assert_array_equal(out_a, out_b)


def test_visible_enemy_state_changes_visible_multi_enemy_actor_obs() -> None:
    visible = np.array([[True, False, False]] * 3, dtype=bool)
    critic_a = _critic_obs()
    critic_b = critic_a.copy()
    critic_b[:, critic_field_slice("enemy0/world_position")] = np.array(
        [40.0, 25.0], dtype=np.float32
    )
    kwargs = {
        "map_bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 50.0, "max_y": 50.0},
        "visible_radius": 1.0,
        "visible_override": visible,
    }

    out_a = zero_masked_enemy_tokens(
        actor_obs_to_multi_enemy_entity_grid_obs(_actor_obs(), critic_obs=critic_a, **kwargs)
    )
    out_b = zero_masked_enemy_tokens(
        actor_obs_to_multi_enemy_entity_grid_obs(_actor_obs(), critic_obs=critic_b, **kwargs)
    )

    assert not np.array_equal(out_a, out_b)


def test_phase4_multi_enemy_env_masks_match_native_visibility() -> None:
    env = Phase4MultiEnemyMappoEnv(_make_sim_cfg(), opponent_bot="noop")
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert info["learner_team"] == "A"
        _tokens, mask = _split(obs)
        expected = np.zeros((3, 3), dtype=np.float32)
        for row, own_slot in enumerate(env._own_slots):
            native = list(_cpp.observable_enemy_slots(env._sim, own_slot))
            for enemy_idx, enemy_slot in enumerate(env._enemy_slots):
                expected[row, enemy_idx] = 1.0 if native[enemy_slot] else 0.0
        np.testing.assert_array_equal(mask[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN], expected)
    finally:
        env.close()


def test_phase4_multi_enemy_team_b_frame_matches_flat_actor_obs() -> None:
    base = Phase4MappoEnv(_make_sim_cfg(), opponent_bot="noop", learner_team="B")
    env = Phase4MultiEnemyMappoEnv(_make_sim_cfg(), opponent_bot="noop", learner_team="B")
    try:
        flat_obs, _ = base.reset(seed=0)
        obs, _ = env.reset(seed=0)
        tokens, _mask = _split(obs)
        np.testing.assert_allclose(
            tokens[:, _SELF_TOKEN, _POSITION],
            flat_obs[:, actor_field_slice("own_position")],
        )
        np.testing.assert_allclose(
            tokens[:, _OBJECTIVE_TOKEN, _POSITION],
            -flat_obs[:, actor_field_slice("own_position")],
        )
        np.testing.assert_allclose(
            tokens[:, _SELF_TOKEN, _AUX],
            flat_obs[:, actor_field_slice("self_on_point")].reshape(3),
        )
    finally:
        base.close()
        env.close()


def test_phase4_multi_enemy_actor_obs_is_opt_in_only() -> None:
    flat = make_mappo_match_env(sim_cfg=_make_sim_cfg(), opponent_bot="noop")
    widened = make_mappo_match_env(
        sim_cfg=_make_sim_cfg(),
        opponent_bot="noop",
        actor_obs="multi_enemy_entity_grid",
    )
    try:
        assert flat.observation_space.shape == (3, ACTOR_PHASE1_DIM)
        assert widened.observation_space.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
    finally:
        flat.close()
        widened.close()


def test_phase4_multi_enemy_probe_config_resolves_runtime_shapes() -> None:
    path = config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    runtime = resolve_runtime_spec(config)

    assert runtime.env.actor_obs == "multi_enemy_entity_grid"
    assert runtime.shapes.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert runtime.shapes.action_dim == 6
    assert runtime.shapes.target_action_dim == 0
    assert config["ppo"]["target_selection_dim"] == 0
    assert config["run"]["warm_start_migration"] == "compatible_exact"
    assert runtime.env_fn is not None


def test_phase4_multi_enemy_supervised_bridge_config_is_opt_in() -> None:
    path = config_path("phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    runtime = resolve_runtime_spec(config)
    bridge_cfg = config["run"]["multi_enemy_supervised_bridge"]

    assert runtime.env.actor_obs == "multi_enemy_entity_grid"
    assert runtime.shapes.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert runtime.shapes.action_dim == 6
    assert runtime.shapes.target_action_dim == 0
    assert bridge_cfg["enabled"] is True
    assert bridge_cfg["teacher"] == "multi_enemy_visible"
    assert bridge_cfg["gate"]["min_team_a_visible_fire_rate"] > 0.0
    assert bridge_cfg["gate"]["min_team_a_hit_fire"] == 0.04
    assert bridge_cfg["gate"]["min_objective_on_point"] == 0.25
    assert bridge_cfg["gate"]["min_mean_score_a"] > 0.0


def test_phase4_multi_enemy_closed_loop_bridge_config_is_bounded_opt_in() -> None:
    path = config_path(
        "phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    runtime = resolve_runtime_spec(config)
    bridge_cfg = config["run"]["multi_enemy_supervised_bridge"]
    closed_loop = bridge_cfg["closed_loop"]

    assert config["wandb"]["enabled"] is False
    assert runtime.env.actor_obs == "multi_enemy_entity_grid"
    assert runtime.shapes.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert runtime.shapes.action_dim == 6
    assert runtime.shapes.target_action_dim == 0
    assert bridge_cfg["enabled"] is True
    assert bridge_cfg["teacher"] == "multi_enemy_visible"
    assert closed_loop["enabled"] is True
    assert 0 < closed_loop["rounds"] <= 20
    assert 0 < closed_loop["updates_per_round"] <= 50
    assert closed_loop["batch_size"] <= 384
    assert bridge_cfg["gate"]["min_team_a_hit_fire"] == 0.04
    assert bridge_cfg["gate"]["min_objective_on_point"] == 0.25


def test_phase4_objective_conversion_bridge_config_is_bounded_opt_in() -> None:
    path = config_path("phase4/probe/phase4_mappo_objective_conversion_bridge_v1.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    runtime = resolve_runtime_spec(config)
    bridge_cfg = config["run"]["multi_enemy_supervised_bridge"]
    closed_loop = bridge_cfg["closed_loop"]
    gate = bridge_cfg["gate"]

    assert config["wandb"]["enabled"] is False
    assert runtime.env.actor_obs == "multi_enemy_entity_grid"
    assert runtime.shapes.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert runtime.shapes.action_dim == 6
    assert runtime.shapes.target_action_dim == 0
    assert bridge_cfg["enabled"] is True
    assert bridge_cfg["teacher"] == "multi_enemy_conversion_hold"
    assert closed_loop["enabled"] is True
    assert 0 < closed_loop["rounds"] <= 20
    assert 0 < closed_loop["updates_per_round"] <= 50
    assert closed_loop["batch_size"] <= 384
    assert closed_loop["conversion_sample_weight"] > 0.0
    assert gate["min_team_a_hit_fire"] == 0.04
    assert gate["min_mean_score_a"] >= 1.0
    assert gate["min_uncontested_on_point_seconds_a"] >= 8.0
    assert gate["min_cap_progress_gain_ticks"] >= 240.0
