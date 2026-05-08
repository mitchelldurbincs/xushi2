from __future__ import annotations

import numpy as np

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.vector_env import XushiAsyncVectorEnv, XushiVectorEnv


def _make_sim_cfg(round_length: int = 1) -> dict:
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


def _make_env() -> Phase4MappoEnv:
    return Phase4MappoEnv(_make_sim_cfg(), opponent_bot="noop")


def test_vector_env_reset_returns_actor_and_critic_batches() -> None:
    env = XushiVectorEnv([_make_env, _make_env], critic_obs_dim=CRITIC_DIM)
    try:
        obs, critic_obs, infos = env.reset(seed=123)
    finally:
        env.close()

    assert obs.shape == (2, 3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert critic_obs.shape == (2, CRITIC_DIM)
    assert critic_obs.dtype == np.float32
    assert len(infos) == 2
    assert infos[0]["tick"] == 0


def test_vector_env_step_validates_action_batch_shape() -> None:
    env = XushiVectorEnv([_make_env, _make_env], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=123)
        try:
            env.step(np.zeros((3, 6), dtype=np.float32))
        except ValueError as exc:
            assert "actions shape" in str(exc)
        else:
            raise AssertionError("expected action shape validation")
    finally:
        env.close()


def test_vector_env_auto_resets_done_envs_and_preserves_final_info() -> None:
    env = XushiVectorEnv([_make_env, _make_env], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=123)
        action = np.zeros((2, 3, 6), dtype=np.float32)
        saw_done = False
        final_infos = []
        for _ in range(20):
            obs, reward, terminated, truncated, critic_obs, infos = env.step(action)
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                saw_done = True
                final_infos.extend(info for info in infos if "final_info" in info)
                assert obs.shape == (2, 3, ACTOR_PHASE1_DIM)
                assert reward.shape == (2, 3)
                assert critic_obs.shape == (2, CRITIC_DIM)
                break

        assert saw_done
        assert final_infos
        assert all(info["final_info"]["tick"] > 0 for info in final_infos)
    finally:
        env.close()


def test_async_vector_env_matches_sync_shapes_and_auto_reset() -> None:
    env = XushiAsyncVectorEnv([_make_env, _make_env], critic_obs_dim=CRITIC_DIM)
    try:
        obs, critic_obs, infos = env.reset(seed=123)
        assert obs.shape == (2, 3, ACTOR_PHASE1_DIM)
        assert obs.dtype == np.float32
        assert critic_obs.shape == (2, CRITIC_DIM)
        assert critic_obs.dtype == np.float32
        assert len(infos) == 2
        assert infos[0]["tick"] == 0

        action = np.zeros((2, 3, 6), dtype=np.float32)
        saw_done = False
        for _ in range(20):
            obs, reward, terminated, truncated, critic_obs, infos = env.step(action)
            done = np.logical_or(terminated, truncated)
            if np.any(done):
                saw_done = True
                assert any("final_info" in info for info in infos)
                assert obs.shape == (2, 3, ACTOR_PHASE1_DIM)
                assert reward.shape == (2, 3)
                assert critic_obs.shape == (2, CRITIC_DIM)
                break
        assert saw_done
    finally:
        env.close()


def test_async_vector_env_step_validates_action_batch_shape() -> None:
    env = XushiAsyncVectorEnv([_make_env, _make_env], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=123)
        try:
            env.step(np.zeros((3, 6), dtype=np.float32))
        except ValueError as exc:
            assert "actions shape" in str(exc)
        else:
            raise AssertionError("expected action shape validation")
    finally:
        env.close()
