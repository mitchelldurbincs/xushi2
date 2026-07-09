"""Tests for the respawn-time curriculum: schedule math, the env setter, the
multi-enemy wrapper delegation (regression for the silently-dropped runtime
setters), and vector-env propagation."""

from __future__ import annotations

import pytest

from envs.phase4_mappo import Phase4MappoEnv
from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv
from train.mappo import compute_respawn_ticks
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.vector_env import XushiAsyncVectorEnv, XushiVectorEnv


def _sim_cfg(respawn_ticks: int = 240) -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 5,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": respawn_ticks,
        },
    }


def _env_fn():
    return Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="noop")


# --- schedule math ------------------------------------------------------


def test_respawn_curriculum_midpoint_linear():
    assert compute_respawn_ticks(
        update=100, initial_ticks=2400, final_ticks=240, anneal_updates=200
    ) == 1320


def test_respawn_curriculum_reaches_final():
    assert compute_respawn_ticks(
        update=200, initial_ticks=2400, final_ticks=240, anneal_updates=200
    ) == 240


def test_respawn_curriculum_holds_final_past_anneal():
    assert compute_respawn_ticks(
        update=999, initial_ticks=2400, final_ticks=240, anneal_updates=200
    ) == 240


def test_respawn_curriculum_noanneal_holds_initial():
    assert compute_respawn_ticks(
        update=999, initial_ticks=2400, final_ticks=240, anneal_updates=0
    ) == 2400


def test_respawn_curriculum_rejects_nonpositive_ticks():
    with pytest.raises(ValueError):
        compute_respawn_ticks(update=1, initial_ticks=0, final_ticks=240, anneal_updates=10)


# --- env setter ---------------------------------------------------------


def test_env_reset_reports_configured_respawn_ticks():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(respawn_ticks=480), opponent_bot="noop")
    try:
        _obs, info = env.reset(seed=0)
        assert info["respawn_ticks"] == 480
    finally:
        env.close()


def test_set_respawn_ticks_applies_on_next_reset_only():
    env = _env_fn()
    try:
        _obs, info = env.reset(seed=0)
        assert info["respawn_ticks"] == 240
        env.set_respawn_ticks(2400)
        # Reset-time-only: the in-flight episode keeps the old value.
        assert env._make_info()["respawn_ticks"] == 240
        _obs, info = env.reset(seed=1)
        assert info["respawn_ticks"] == 2400
    finally:
        env.close()


def test_set_respawn_ticks_rejects_nonpositive():
    env = _env_fn()
    try:
        with pytest.raises(ValueError):
            env.set_respawn_ticks(0)
    finally:
        env.close()


def test_set_respawn_ticks_does_not_mutate_caller_sim_cfg():
    sim_cfg = _sim_cfg()
    env = Phase4MappoEnv(sim_cfg=sim_cfg, opponent_bot="noop")
    try:
        env.set_respawn_ticks(2400)
        assert sim_cfg["mechanics"]["respawn_ticks"] == 240
    finally:
        env.close()


# --- multi-enemy wrapper delegation (setter-drop regression) ------------


def test_multi_enemy_wrapper_delegates_respawn_and_timing_setters():
    env = Phase4MultiEnemyMappoEnv(_sim_cfg(), opponent_bot="noop")
    try:
        env.set_respawn_ticks(2400)
        env.set_objective_timing_seconds(5.0, 2.0)
        env.set_team_spirit(0.7)
        env.set_majority_on_point_alpha(0.0)
        env.set_uncontested_on_point_alpha(0.0)
        _obs, info = env.reset(seed=0)
        assert _obs.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert info["respawn_ticks"] == 2400
        assert info["objective_unlock_seconds"] == pytest.approx(5.0)
        assert info["objective_capture_seconds"] == pytest.approx(2.0)
        assert env._base._reward_calc._team_spirit == pytest.approx(0.7)
    finally:
        env.close()


def test_multi_enemy_wrapper_exposes_every_runtime_setter_the_base_env_has():
    """The vector env discovers setters with getattr() and silently skips
    envs that lack them. Any Phase4MappoEnv runtime setter that the wrapper
    forgets to delegate is dropped without an error — this is exactly the bug
    that disabled the timing anneal in the 2026-06-10 conversion runs."""
    setters = [
        name
        for name in dir(Phase4MappoEnv)
        if name.startswith("set_") and callable(getattr(Phase4MappoEnv, name))
    ]
    assert setters, "expected Phase4MappoEnv to expose runtime setters"
    for name in setters:
        assert callable(getattr(Phase4MultiEnemyMappoEnv, name, None)), (
            f"Phase4MultiEnemyMappoEnv is missing runtime setter {name!r}; "
            "the vector env will silently drop it for multi-enemy runs"
        )


# --- vector env propagation ---------------------------------------------


def test_xushi_vector_env_set_respawn_ticks_propagates_sync():
    env = XushiVectorEnv([_env_fn, _env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.set_respawn_ticks(2400)
        _obs, _critic, infos = env.reset(seed=0)
        assert [info["respawn_ticks"] for info in infos] == [2400, 2400]
    finally:
        env.close()


def test_xushi_async_vector_env_set_respawn_ticks_round_trips():
    env = XushiAsyncVectorEnv([_env_fn, _env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        env.set_respawn_ticks(2400)
        _obs, _critic, infos = env.reset(seed=0)
        assert [info["respawn_ticks"] for info in infos] == [2400, 2400]
    finally:
        env.close()
