from __future__ import annotations

import numpy as np

from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv
from envs.phase4_combat_1v1_mappo import Phase4Combat1v1MappoEnv
from train.mappo_rollout_trainer import make_mappo_config
from train.phases import resolve_phase
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice


def _zero_action() -> np.ndarray:
    return np.zeros((3, 6), dtype=np.float32)


def test_cap_duel_tensor_shape_parity_with_combat_1v1() -> None:
    cap = Phase4CapDuelMappoEnv(episode_decisions=4)
    combat = Phase4Combat1v1MappoEnv(episode_decisions=4)

    assert cap.n_agents == combat.n_agents == 3
    assert cap.actor_obs_dim == combat.actor_obs_dim == ACTOR_PHASE1_DIM
    assert cap.critic_obs_dim == combat.critic_obs_dim == CRITIC_DIM
    assert cap.action_dim == combat.action_dim == 6
    assert cap.observation_space.shape == combat.observation_space.shape
    assert cap.action_space.shape == combat.action_space.shape


def test_cap_duel_env_shapes_spawn_radius_and_active_slot_mask() -> None:
    env = Phase4CapDuelMappoEnv(episode_decisions=4, point_radius=0.2)
    obs, info = env.reset(seed=123)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert info["cap_duel_hit_rate"] == 0.0
    assert info["loss_mask"].tolist() == [1.0, 0.0, 0.0]
    assert obs[0].sum() != 0.0
    assert np.allclose(obs[1], 0.0)
    assert np.allclose(obs[2], 0.0)
    assert obs[0, actor_field_slice("enemy_alive")][0] == 1.0
    assert np.linalg.norm(env._pos[0]) <= env.point_radius
    assert np.linalg.norm(env._pos[1]) <= env.point_radius


def test_anchor_match_ignores_inactive_slot_actions() -> None:
    env = Phase4CapDuelMappoEnv(episode_decisions=4, point_radius=0.2)
    env.reset(seed=7)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.0, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = _zero_action()
    action[1, :2] = np.array([1.0, 0.0], dtype=np.float32)
    action[2, :2] = np.array([-1.0, 0.0], dtype=np.float32)
    env.step(action)
    assert np.allclose(env._pos[0], [0.0, 0.0])
    assert np.allclose(env._pos[1], [0.0, 0.0])


def test_score_does_not_advance_while_enemy_alive_and_on_point() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=8,
        enemy_hp=3,
        point_radius=0.2,
        score_ticks_to_clear=2,
        enemy_recontest_delay=2,
    )
    env.reset(seed=7)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.0, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    _obs, reward, term, trunc, info = env.step(_zero_action())
    assert not term
    assert not trunc
    assert reward[0] == 0.0
    assert info["team_a_score"] == 0.0


def test_score_advances_after_configured_displacement_delay() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=8,
        enemy_hp=3,
        point_radius=0.2,
        score_ticks_to_clear=2,
        enemy_recontest_delay=2,
        score_per_tick=0.25,
    )
    env.reset(seed=7)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.5, 0.0], dtype=np.float32)
    env._off_point_decisions[1] = 1
    env._build_actor_obs_all()

    _obs, reward, _term, _trunc, info = env.step(_zero_action())
    assert info["team_a_score"] == 0.0
    assert reward[0] == 0.0

    _obs, reward, _term, _trunc, info = env.step(_zero_action())
    assert info["team_a_score"] == 1.0
    assert reward[0] == 0.25


def test_exact_aim_kill_opens_next_decision_score_window() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=8,
        enemy_hp=1,
        point_radius=0.2,
        score_ticks_to_clear=2,
        enemy_recontest_delay=3,
        hit_tolerance=0.02,
        hit_reward=1.0,
        kill_bonus=4.0,
        score_per_tick=0.1,
    )
    env.reset(seed=7)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.1, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = _zero_action()
    action[0, 2] = env._target_norm(0)
    action[0, 3] = 1.0
    _obs, reward, term, trunc, info = env.step(action)
    assert not term
    assert not trunc
    assert reward[0] == 5.0
    assert info["team_a_kills"] == 1
    assert info["team_a_score"] == 0.0
    assert info["cap_duel_enemy_alive"] is False

    _obs, reward, _term, _trunc, info = env.step(_zero_action())
    assert reward[0] == 0.1
    assert info["team_a_score"] == 1.0


def test_bad_aim_does_not_kill_or_score_when_contested() -> None:
    env = Phase4CapDuelMappoEnv(
        episode_decisions=4,
        enemy_hp=1,
        point_radius=0.2,
        hit_tolerance=0.02,
    )
    env.reset(seed=7)
    env._pos[0] = np.array([0.0, 0.0], dtype=np.float32)
    env._pos[1] = np.array([0.1, 0.0], dtype=np.float32)
    env._build_actor_obs_all()

    action = _zero_action()
    action[0, 2] = np.clip(env._target_norm(0) + 0.5, -1.0, 1.0)
    action[0, 3] = 1.0
    _obs, reward, _term, _trunc, info = env.step(action)
    assert reward[0] == 0.0
    assert info["cap_duel_hits"] == 0
    assert info["cap_duel_misses"] == 1
    assert info["team_a_score"] == 0.0


def test_deterministic_reset_under_fixed_seed() -> None:
    env_a = Phase4CapDuelMappoEnv(episode_decisions=4)
    env_b = Phase4CapDuelMappoEnv(episode_decisions=4)
    obs_a, info_a = env_a.reset(seed=101)
    obs_b, info_b = env_b.reset(seed=101)

    assert np.allclose(obs_a, obs_b)
    assert info_a["state_hash"] == info_b["state_hash"]


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


def test_cap_duel_selfplay_keeps_three_agent_shape_and_uses_loss_masks() -> None:
    cfg = {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "mini_game": "cap_duel",
            "mini_game_config": {"episode_decisions": 4},
            "self_play": {"enabled": True},
            "self_play_schedule": {
                "weights": {"current": 1.0, "anchor": 0.0, "snapshot": 0.0},
                "anchor_bot": "noop",
            },
            "sim": {"seed": 0},
        },
        "model": {
            "embed_dim": 32,
            "gru_hidden": 32,
            "head_hidden": 32,
            "action_log_std_init": -1.0,
        },
        "ppo": {
            "num_envs": 1,
            "rollout_len": 2,
            "num_epochs": 1,
            "minibatch_size": 1,
            "learning_rate": 1.0e-5,
            "value_per_agent": True,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "value_clip_ratio": 0.2,
            "entropy_coef": 0.001,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    }
    _phase, spec = resolve_phase(cfg)
    env_fn, env_meta, _seed = spec["env_bundle"](cfg)
    mappo_cfg = make_mappo_config(cfg)
    assert spec["label"] == "phase4_selfplay"
    assert mappo_cfg.n_agents == 3
    assert mappo_cfg.value_per_agent is True
    assert env_meta["self_play"]["enabled"] is True

    routed = env_fn()
    obs, info = routed.reset(seed=0)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert info["match_type"] == "current"
    assert info["loss_mask"].tolist() == [1.0, 1.0, 0.0]
    assert obs[1].sum() != 0.0
    out = np.empty(3 * CRITIC_DIM, dtype=np.float32)
    routed.build_critic_obs(out)
    assert np.all(np.isfinite(out))
