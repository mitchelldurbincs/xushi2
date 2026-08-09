from __future__ import annotations

import numpy as np
import gymnasium as gym
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2 import vector_env
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


class _DictActionEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self.action_space = gym.spaces.Dict(
            {"move": gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 2), dtype=np.float32)}
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        return np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            False,
            False,
            {},
        )

    def build_critic_obs(self, out: np.ndarray) -> None:
        out.fill(0.0)


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


def test_vector_env_accepts_box_action_space() -> None:
    env = XushiVectorEnv([_make_env], critic_obs_dim=CRITIC_DIM)
    try:
        assert isinstance(env.single_action_space, gym.spaces.Box)
    finally:
        env.close()


def test_vector_env_rejects_dict_action_space_with_clear_error() -> None:
    try:
        XushiVectorEnv([_DictActionEnv], critic_obs_dim=CRITIC_DIM)
    except TypeError as exc:
        message = str(exc)
        assert "gym.spaces.Box" in message
        assert "Dict" in message
        assert "serialization/stacking" in message
    else:
        raise AssertionError("expected Dict action space rejection")


def test_async_vector_env_rejects_dict_action_space_with_clear_error() -> None:
    try:
        XushiAsyncVectorEnv([_DictActionEnv], critic_obs_dim=CRITIC_DIM)
    except TypeError as exc:
        message = str(exc)
        assert "gym.spaces.Box" in message
        assert "Dict" in message
        assert "serialization/stacking" in message
    else:
        raise AssertionError("expected Dict action space rejection")


class _WorkerFailureEnv(gym.Env):
    """Env whose async worker fails without replying on the first step.

    ``PROBE`` selects the failure mode, set on the class before the worker is
    spawned so the value travels with the pickled factory:

    - ``"kill"``  -- the worker dies by signal, standing in for an abort
      inside the C++ sim. The parent's pipe copy is already closed, so this
      shows up as EOF rather than a hang.
    - ``"wedge"`` -- the worker stays alive but never replies. Nothing closes
      the pipe, so an unbounded recv() would block forever.
    """

    PROBE = "kill"

    # This fake exists to die or wedge; no curriculum knob applies to it.
    UNSUPPORTED_CURRICULUM_SETTERS = {
        "set_team_spirit": "failure-injection fake; no reward path",
        "set_majority_on_point_alpha": "failure-injection fake; no reward path",
        "set_uncontested_on_point_alpha": "failure-injection fake; no reward path",
        "set_objective_timing_seconds": "failure-injection fake; no objective",
        "set_respawn_ticks": "failure-injection fake; no respawn",
    }

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, ACTOR_PHASE1_DIM), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3, 6), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        return np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32), {}

    def step(self, action):
        import os
        import signal
        import time

        if _WorkerFailureEnv.PROBE == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(3600)

    def build_critic_obs(self, out: np.ndarray) -> None:
        out[:] = 0.0

    def close(self) -> None:
        return None


def _make_killing_env() -> _WorkerFailureEnv:
    _WorkerFailureEnv.PROBE = "kill"
    return _WorkerFailureEnv()


def _make_wedging_env() -> _WorkerFailureEnv:
    _WorkerFailureEnv.PROBE = "wedge"
    return _WorkerFailureEnv()


@pytest.mark.timeout(120)
def test_async_worker_signal_death_reports_worker_and_exitcode() -> None:
    """A dead worker must name itself and its exit code, not raise a bare EOFError."""
    env = XushiAsyncVectorEnv([_make_killing_env], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=0)
        with pytest.raises(RuntimeError) as excinfo:
            env.step(np.zeros((1, 3, 6), dtype=np.float32))
        message = str(excinfo.value)
        assert "worker 0" in message
        # -9 is SIGKILL; a negative exitcode is the signal-death signature.
        assert "exitcode=-9" in message
    finally:
        env.close()


@pytest.mark.timeout(120)
def test_async_worker_wedge_times_out_instead_of_hanging(monkeypatch) -> None:
    """An alive-but-wedged worker must time out rather than block forever."""
    monkeypatch.setattr(vector_env, "_WORKER_RECV_TIMEOUT_SECONDS", 2.0)
    env = XushiAsyncVectorEnv([_make_wedging_env], critic_obs_dim=CRITIC_DIM)
    try:
        env.reset(seed=0)
        with pytest.raises(TimeoutError) as excinfo:
            env.step(np.zeros((1, 3, 6), dtype=np.float32))
        message = str(excinfo.value)
        assert "worker 0" in message
        assert "still alive" in message
    finally:
        env.close()
