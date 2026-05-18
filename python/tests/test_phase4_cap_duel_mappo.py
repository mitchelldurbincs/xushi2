from __future__ import annotations

import numpy as np

from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv
from train.phases import resolve_phase
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice


def test_cap_duel_env_shapes_and_active_slot() -> None:
    env = Phase4CapDuelMappoEnv(episode_decisions=4)
    obs, info = env.reset(seed=123)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert info["cap_duel_hit_rate"] == 0.0
    assert obs[0].sum() != 0.0
    assert obs[1].sum() != obs[0].sum()
    assert obs[0, actor_field_slice("enemy_alive")][0] == 1.0


def test_score_requires_learner_on_point_and_enemy_displaced() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=8,
        enemy_hp=3,
        point_radius=0.2,
        score_ticks_to_clear=2,
        enemy_recontest_delay=2,
    )
    env.reset(seed=7)
    env._learner_pos = np.array([0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([0.0, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = np.zeros((3, 6), dtype=np.float32)
    _obs, reward, term, trunc, info = env.step(action)
    assert not term
    assert not trunc
    assert reward[0] < 1.0
    assert info["team_a_score"] == 0.0

    env._enemy_pos = np.array([0.5, 0.0], dtype=np.float32)
    env._build_actor_obs_all()
    _obs, reward, _term, _trunc, info = env.step(action)
    assert reward[0] > 0.0
    assert info["team_a_score"] == 1.0


def test_exact_aim_kills_enemy_and_opens_score_window() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=8,
        enemy_hp=1,
        point_radius=0.2,
        score_ticks_to_clear=2,
        enemy_recontest_delay=3,
        hit_tolerance=0.02,
    )
    env.reset(seed=7)
    env._learner_pos = np.array([0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([0.1, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = np.zeros((3, 6), dtype=np.float32)
    action[0, 2] = env._target_norm()
    action[0, 3] = 1.0
    _obs, reward, term, trunc, info = env.step(action)
    assert not term
    assert not trunc
    assert reward[0] > 2.0
    assert info["team_a_kills"] == 1
    assert info["team_a_score"] == 1.0
    assert info["cap_duel_enemy_alive"] is False


def test_bad_aim_does_not_kill_or_score_when_contested() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=4,
        enemy_hp=1,
        point_radius=0.2,
        hit_tolerance=0.02,
    )
    env.reset(seed=7)
    env._learner_pos = np.array([0.0, 0.0], dtype=np.float32)
    env._enemy_pos = np.array([0.1, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = np.zeros((3, 6), dtype=np.float32)
    action[0, 2] = np.clip(env._target_norm() + 0.5, -1.0, 1.0)
    action[0, 3] = 1.0
    _obs, reward, _term, _trunc, info = env.step(action)
    assert reward[0] < 0.0
    assert info["cap_duel_hits"] == 0
    assert info["cap_duel_misses"] == 1
    assert info["team_a_score"] == 0.0


def test_build_critic_obs_and_registry_routing() -> None:
    env = Phase4CapDuelMappoEnv(episode_decisions=4)
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
            "mini_game": "cap_duel",
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
    assert isinstance(routed, Phase4CapDuelMappoEnv)
    assert env_meta["mini_game"] == "cap_duel"
    assert seed == 0
