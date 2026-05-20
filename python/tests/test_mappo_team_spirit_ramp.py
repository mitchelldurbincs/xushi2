"""Tests for the OAI Five-style team_spirit ramp schedule and the
``set_team_spirit`` propagation through the vector env wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from train.mappo import (
    compute_majority_on_point_alpha,
    compute_objective_timing_seconds,
    compute_team_spirit,
)
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.vector_env import XushiAsyncVectorEnv, XushiVectorEnv

# --- compute_team_spirit ramp ------------------------------------------


def test_team_spirit_at_start_is_initial():
    assert compute_team_spirit(
        update=1, total=1000, initial=0.3, final=1.0, ramp_fraction=0.3
    ) == pytest.approx(0.3 + (1 / 300) * (1.0 - 0.3))


def test_team_spirit_at_ramp_end_is_final():
    assert compute_team_spirit(
        update=300, total=1000, initial=0.3, final=1.0, ramp_fraction=0.3
    ) == pytest.approx(1.0)


def test_team_spirit_after_ramp_end_holds_at_final():
    assert compute_team_spirit(
        update=999, total=1000, initial=0.3, final=1.0, ramp_fraction=0.3
    ) == pytest.approx(1.0)


def test_team_spirit_midpoint_linear():
    # update=150 of 1000 with ramp=30%: ramp progress = 150/300 = 0.5.
    # tau = 0.3 + 0.5 * (1.0 - 0.3) = 0.65.
    assert compute_team_spirit(
        update=150, total=1000, initial=0.3, final=1.0, ramp_fraction=0.3
    ) == pytest.approx(0.65)


def test_team_spirit_ramp_fraction_zero_jumps_to_final():
    assert compute_team_spirit(
        update=0, total=1000, initial=0.3, final=1.0, ramp_fraction=0.0
    ) == pytest.approx(1.0)


def test_team_spirit_default_off_returns_zero_throughout():
    """Defaults (initial=0, final=0) keep team_spirit OFF for back-compat."""
    for u in (1, 50, 500, 999):
        assert compute_team_spirit(
            update=u, total=1000, initial=0.0, final=0.0, ramp_fraction=0.3
        ) == pytest.approx(0.0)


# --- majority-on-point alpha anneal ------------------------------------


def test_majority_on_point_alpha_reaches_zero_at_anneal_end():
    assert compute_majority_on_point_alpha(
        update=50, initial=0.2, anneal_updates=50
    ) == pytest.approx(0.0)


def test_majority_on_point_alpha_midpoint_linear():
    assert compute_majority_on_point_alpha(
        update=25, initial=0.2, anneal_updates=50
    ) == pytest.approx(0.1)


def test_majority_on_point_alpha_noanneal_holds_constant():
    assert compute_majority_on_point_alpha(
        update=999, initial=0.2, anneal_updates=0
    ) == pytest.approx(0.2)


# --- objective timing curriculum ---------------------------------------


def test_objective_timing_curriculum_midpoint_linear():
    unlock, capture = compute_objective_timing_seconds(
        update=50,
        initial_unlock_seconds=5.0,
        final_unlock_seconds=15.0,
        initial_capture_seconds=2.0,
        final_capture_seconds=8.0,
        anneal_updates=100,
    )
    assert unlock == pytest.approx(10.0)
    assert capture == pytest.approx(5.0)


def test_objective_timing_curriculum_reaches_final():
    unlock, capture = compute_objective_timing_seconds(
        update=100,
        initial_unlock_seconds=5.0,
        final_unlock_seconds=15.0,
        initial_capture_seconds=2.0,
        final_capture_seconds=8.0,
        anneal_updates=100,
    )
    assert unlock == pytest.approx(15.0)
    assert capture == pytest.approx(8.0)


def test_objective_timing_noanneal_holds_initial():
    unlock, capture = compute_objective_timing_seconds(
        update=999,
        initial_unlock_seconds=5.0,
        final_unlock_seconds=15.0,
        initial_capture_seconds=2.0,
        final_capture_seconds=8.0,
        anneal_updates=0,
    )
    assert unlock == pytest.approx(5.0)
    assert capture == pytest.approx(2.0)


# --- vector wrapper propagation ----------------------------------------


def _phase4_sim_cfg() -> dict:
    return {
        "seed": 0,
        "round_length_seconds": 5,
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


def _phase4_env_fn():
    return Phase4MappoEnv(
        sim_cfg=_phase4_sim_cfg(),
        opponent_bot="noop",
    )


def _phase4_majority_env_fn():
    return Phase4MappoEnv(
        sim_cfg=_phase4_sim_cfg(),
        opponent_bot="noop",
        reward_cfg={"majority_on_point_coef": 0.2},
    )


def test_xushi_vector_env_set_team_spirit_propagates_sync():
    env = XushiVectorEnv([_phase4_env_fn, _phase4_env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=0)
        env.set_team_spirit(0.7)
        for sub in env.envs:
            assert sub._reward_calc._team_spirit == pytest.approx(0.7)
    finally:
        env.close()


def test_xushi_vector_env_set_majority_on_point_alpha_propagates_sync():
    env = XushiVectorEnv(
        [_phase4_majority_env_fn, _phase4_majority_env_fn],
        critic_obs_dim=CRITIC_DIM,
    )
    try:
        env.reset(seed=0)
        env.set_majority_on_point_alpha(0.0)
        for sub in env.envs:
            assert sub._reward_calc.majority_on_point_alpha == pytest.approx(0.0)
    finally:
        env.close()


def test_xushi_vector_env_set_objective_timing_propagates_sync():
    env = XushiVectorEnv([_phase4_env_fn, _phase4_env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.set_objective_timing_seconds(5.0, 2.0)
        _obs, _critic, infos = env.reset(seed=0)
        assert [info["objective_unlock_seconds"] for info in infos] == pytest.approx(
            [5.0, 5.0]
        )
        assert [info["objective_capture_seconds"] for info in infos] == pytest.approx(
            [2.0, 2.0]
        )
    finally:
        env.close()


def test_xushi_async_vector_env_set_team_spirit_round_trips():
    env = XushiAsyncVectorEnv([_phase4_env_fn, _phase4_env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=0)
        # Async setter must complete (acks from each worker) without raising.
        env.set_team_spirit(0.55)
        # No way to inspect worker state directly without round-tripping a
        # follow-up action; the ack itself proves the worker dispatched the
        # command. A subsequent step must still succeed.
        actions = np.zeros((2, 3, 6), dtype=env.single_action_space.dtype)
        _obs, reward, _term, _trunc, _critic, _infos = env.step(actions)
        assert reward.shape == (2, 3)
    finally:
        env.close()


def test_xushi_async_vector_env_set_objective_timing_round_trips():
    env = XushiAsyncVectorEnv([_phase4_env_fn, _phase4_env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.set_objective_timing_seconds(5.0, 2.0)
        _obs, _critic, infos = env.reset(seed=0)
        assert [info["objective_unlock_seconds"] for info in infos] == pytest.approx(
            [5.0, 5.0]
        )
        assert [info["objective_capture_seconds"] for info in infos] == pytest.approx(
            [2.0, 2.0]
        )
    finally:
        env.close()
