"""Tests for envs.phase4_mappo.Phase4MappoEnv.

3v3 MAPPO-shaped Gymnasium env. Per-agent (3, 31) actor obs and (3, 6)
actions, with a separate caller-buffered build_critic_obs(out) hook
returning 135 floats. See
docs/plans/2026-05-07-phase4-mappo-env-design.md for layout rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
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


def _make_env(opponent_bot: str = "noop", **kwargs) -> Phase4MappoEnv:
    return Phase4MappoEnv(_make_sim_cfg(), opponent_bot=opponent_bot, **kwargs)


# --- Task 1: construction / spaces / validation ---


def test_construct_observation_space_shape_is_3_by_actor_dim():
    env = _make_env()
    assert env.observation_space.shape == (3, ACTOR_PHASE1_DIM)
    assert env.observation_space.dtype == np.float32


def test_construct_action_space_shape_is_3_by_6():
    env = _make_env()
    assert env.action_space.shape == (3, 6)
    assert env.action_space.dtype == np.float32


def test_step_before_reset_raises():
    env = _make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((3, 6), dtype=np.float32))


def test_invalid_opponent_bot_raises():
    with pytest.raises(ValueError, match="opponent_bot"):
        Phase4MappoEnv(_make_sim_cfg(), opponent_bot="not_a_real_bot")


def test_invalid_learner_team_raises():
    with pytest.raises(ValueError, match="learner_team"):
        Phase4MappoEnv(_make_sim_cfg(), opponent_bot="noop", learner_team="C")


# --- Task 2: reset ---


def test_reset_returns_correct_shapes():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert info["tick"] == 0
    assert info["winner"] == "Neutral"
    assert info["learner_team"] == "A"
    assert info["objective_unlock_seconds"] == pytest.approx(15.0)
    assert info["objective_capture_seconds"] == pytest.approx(8.0)


def test_objective_timing_seconds_config_reaches_info():
    cfg = _make_sim_cfg()
    cfg["objective_unlock_seconds"] = 5.0
    cfg["objective_capture_seconds"] = 2.0
    env = Phase4MappoEnv(cfg, opponent_bot="noop")
    _obs, info = env.reset(seed=0)

    assert info["objective_unlock_ticks"] == 5 * _cpp.TICK_HZ
    assert info["objective_capture_ticks"] == 2 * _cpp.TICK_HZ
    assert info["objective_unlock_seconds"] == pytest.approx(5.0)
    assert info["objective_capture_seconds"] == pytest.approx(2.0)


def test_objective_timing_setter_overrides_seconds_config_on_future_reset():
    cfg = _make_sim_cfg()
    cfg["objective_unlock_seconds"] = 5.0
    cfg["objective_capture_seconds"] = 2.0
    env = Phase4MappoEnv(cfg, opponent_bot="noop")
    env.set_objective_timing_seconds(6.0, 3.0)

    _obs, info = env.reset(seed=0)

    assert info["objective_unlock_ticks"] == 6 * _cpp.TICK_HZ
    assert info["objective_capture_ticks"] == 3 * _cpp.TICK_HZ
    assert info["objective_unlock_seconds"] == pytest.approx(6.0)
    assert info["objective_capture_seconds"] == pytest.approx(3.0)


def test_objective_timing_setter_updates_current_and_future_episodes():
    env = _make_env()
    env.set_objective_timing_seconds(5.0, 2.0)
    _obs, info = env.reset(seed=0)
    assert info["objective_unlock_seconds"] == pytest.approx(5.0)
    assert info["objective_capture_seconds"] == pytest.approx(2.0)

    env.set_objective_timing_seconds(6.0, 3.0)
    _obs, _reward, _term, _trunc, info = env.step(np.zeros((3, 6), dtype=np.float32))
    assert info["objective_unlock_seconds"] == pytest.approx(6.0)
    assert info["objective_capture_seconds"] == pytest.approx(3.0)

    _obs, info = env.reset(seed=1)
    assert info["objective_unlock_seconds"] == pytest.approx(6.0)
    assert info["objective_capture_seconds"] == pytest.approx(3.0)


def test_team_b_learner_resets_with_correct_own_slots():
    env = _make_env(learner_team="B")
    assert env._own_slots == (3, 4, 5)
    assert env._enemy_slots == (0, 1, 2)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert info["learner_team"] == "B"


def test_reset_rejects_team_size_in_sim_cfg():
    cfg = _make_sim_cfg()
    cfg["team_size"] = 1
    env = Phase4MappoEnv(cfg, opponent_bot="noop")
    with pytest.raises(ValueError, match="team_size"):
        env.reset(seed=0)


# --- Task 3: step ---


def test_step_returns_correct_shapes_and_finite_values():
    env = _make_env(opponent_bot="noop")
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(np.zeros((3, 6), dtype=np.float32))
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert reward.shape == (3,)
    assert reward.dtype == np.float32
    assert np.all(np.isfinite(reward))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    # Per-team reward summary remains scalar in info for downstream loggers.
    assert isinstance(info["reward_team_a"], float)
    assert isinstance(info["reward_team_b"], float)


def test_majority_on_point_alpha_can_be_disabled_for_eval_info():
    env = _make_env(
        opponent_bot="noop",
        reward_cfg={"majority_on_point_coef": 0.2},
    )
    env.reset(seed=0)
    env.set_majority_on_point_alpha(0.0)
    _obs, _reward, _term, _trunc, info = env.step(np.zeros((3, 6), dtype=np.float32))

    assert info["majority_on_point_alpha"] == pytest.approx(0.0)
    assert info["majority_on_point_reward_a"] == pytest.approx(0.0)
    assert info["majority_on_point_reward_b"] == pytest.approx(0.0)
    assert isinstance(info["objective_metrics"], dict)


@pytest.mark.parametrize("bad_shape", [(6,), (3,), (3, 5), (4, 6)])
def test_action_shape_validation_raises(bad_shape):
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(bad_shape, dtype=np.float32))




def test_step_does_not_mutate_caller_action_array():
    env = _make_env(opponent_bot="noop")
    env.reset(seed=0)
    action = np.array(
        [
            [2.0, -2.0, 1.5, -1.0, 2.0, 0.75],
            [-1.5, 1.25, -2.0, 1.5, -0.5, 0.49],
            [0.25, -0.25, 3.0, 0.25, 1.25, -3.0],
        ],
        dtype=np.float32,
    )
    original = action.copy()

    env.step(action)

    np.testing.assert_array_equal(action, original)

def test_per_agent_reward_sum_stays_finite_across_full_episode():
    env = _make_env(opponent_bot="noop")
    env.reset(seed=42)
    cumulative_per_agent = np.zeros(3, dtype=np.float32)
    for _ in range(2000):
        action = np.zeros((3, 6), dtype=np.float32)
        _, reward, term, trunc, _info = env.step(action)
        assert reward.shape == (3,)
        assert reward.dtype == np.float32
        assert np.all(np.isfinite(reward))
        cumulative_per_agent += reward
        if term or trunc:
            break
    assert np.all(np.isfinite(cumulative_per_agent))


def test_hardcoded_walk_to_objective_scores_against_noop():
    env = Phase4MappoEnv(_make_sim_cfg(round_length=30), opponent_bot="noop")
    obs, _ = env.reset(seed=0)
    pos_slice = actor_field_slice("own_position")
    final_info = {}
    for _ in range(400):
        own_pos = obs[:, pos_slice]
        move = -own_pos.copy()
        norm = np.linalg.norm(move, axis=1, keepdims=True)
        move = np.where(norm > 0.02, move / np.maximum(norm, 1e-6), 0.0)
        action = np.zeros((3, 6), dtype=np.float32)
        action[:, :2] = move.astype(np.float32)
        obs, _reward, term, trunc, final_info = env.step(action)
        if term or trunc:
            break

    assert float(final_info["team_a_score"]) > 0.0
    assert float(final_info["team_b_score"]) == 0.0


# --- Task 4: build_critic_obs ---


def test_build_critic_obs_writes_135_finite_floats_with_actor_prefix():
    env = _make_env()
    env.reset(seed=0)
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    env.build_critic_obs(out)
    assert np.all(np.isfinite(out))
    actor = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(env._sim, env._own_slots[0], actor)
    np.testing.assert_array_equal(out[:ACTOR_PHASE1_DIM], actor)


def test_build_critic_obs_before_reset_raises():
    env = _make_env()
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    with pytest.raises(RuntimeError, match="reset"):
        env.build_critic_obs(out)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(CRITIC_DIM - 1, dtype=np.float32),
        np.zeros((CRITIC_DIM, 1), dtype=np.float32),
        np.zeros(CRITIC_DIM, dtype=np.float64),
    ],
)
def test_build_critic_obs_buffer_validation(bad):
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.build_critic_obs(bad)


def test_critic_obs_team_b_uses_team_b_actor_prefix():
    env = _make_env(learner_team="B")
    env.reset(seed=0)
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    env.build_critic_obs(out)
    actor_b0 = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(env._sim, 3, actor_b0)
    np.testing.assert_array_equal(out[:ACTOR_PHASE1_DIM], actor_b0)


# --- Task 5: determinism + smoke ---


def test_determinism_two_envs_same_seed_same_state_hash():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(100, 3, 6)).astype(np.float32)
    actions[..., 3:] = (actions[..., 3:] > 0).astype(np.float32)

    env_a = _make_env()
    env_b = _make_env()
    _, info_a = env_a.reset(seed=42)
    _, info_b = env_b.reset(seed=42)
    assert info_a["state_hash"] == info_b["state_hash"]

    for action in actions:
        _, _, term_a, trunc_a, info_a = env_a.step(action)
        _, _, term_b, trunc_b, info_b = env_b.step(action)
        assert info_a["state_hash"] == info_b["state_hash"], (
            f"state_hash divergence at tick {info_a['tick']}"
        )
        if term_a or trunc_a:
            assert term_b == term_a and trunc_b == trunc_a
            break


def test_idle_1000_ticks_no_crash():
    env = _make_env(opponent_bot="basic")
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        action = rng.uniform(-1, 1, size=(3, 6)).astype(np.float32)
        action[:, 3:] = (action[:, 3:] > 0).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))


# --- objective conversion reward terms (real sim integration) ---


def test_conversion_reward_terms_run_against_real_sim():
    """New env.reward keys flow into RewardCalculator and read real obs.

    With a noop opponent and Team A driven onto the point, the capture-progress
    potential must eventually pay positive reward and a capture completion must
    be observed once the objective unlocks and 'capture_seconds' elapse.
    """
    sim_cfg = _make_sim_cfg(round_length=40)
    sim_cfg["objective_unlock_seconds"] = 2.0
    sim_cfg["objective_capture_seconds"] = 2.0
    env = Phase4MappoEnv(
        sim_cfg,
        opponent_bot="noop",
        reward_cfg={
            "cap_progress_potential_coef": 1.0,
            "capture_completed_bonus": 2.0,
            "score_per_second": 0.0,
            "kill_bonus": 0.0,
            "death_penalty": 0.0,
        },
    )
    obs, info = env.reset(seed=7)
    pos_slice = actor_field_slice("own_position")
    total_conversion = 0.0
    captures = 0.0
    for _ in range(300):
        action = np.zeros((3, 6), dtype=np.float32)
        for i in range(3):
            own = obs[i][pos_slice]
            to_point = -own
            norm = float(np.hypot(to_point[0], to_point[1]))
            if norm > 1.0e-6:
                action[i, 0] = to_point[0] / norm
                action[i, 1] = to_point[1] / norm
        obs, _reward, terminated, truncated, info = env.step(action)
        metrics = info["reward_metrics"]
        total_conversion += float(metrics["cap_progress_potential_reward_a"])
        total_conversion += float(metrics["capture_completed_reward_a"])
        captures = float(metrics["captures_a"])
        if terminated or truncated:
            break
    assert captures >= 1.0
    assert total_conversion > 0.5
