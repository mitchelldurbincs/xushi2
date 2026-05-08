"""Vector wrappers for Xushi multi-agent envs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import multiprocessing as mp
from typing import Any

import gymnasium as gym
import numpy as np


def _worker_critic_obs(env: gym.Env, critic_obs_dim: int) -> np.ndarray:
    out = np.zeros(critic_obs_dim, dtype=np.float32)
    env.build_critic_obs(out)
    return out


def _async_worker(
    conn,
    env_fn: Callable[[], gym.Env],
    critic_obs_dim: int,
    seed_base: int,
    env_idx: int,
    auto_reset: bool,
) -> None:
    env = env_fn()
    episode_count = 0
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                seed = seed_base + env_idx if payload is None else int(payload) + env_idx
                episode_count = 0
                obs, info = env.reset(seed=seed)
                conn.send((obs, _worker_critic_obs(env, critic_obs_dim), dict(info)))
            elif cmd == "step":
                obs, reward, term, trunc, info = env.step(payload)
                term = bool(term)
                trunc = bool(trunc)
                info = dict(info)
                if (term or trunc) and auto_reset:
                    info["final_info"] = dict(info)
                    info["final_observation"] = np.asarray(
                        obs, dtype=env.observation_space.dtype
                    )
                    episode_count += 1
                    reset_seed = seed_base + 10_000 * episode_count + env_idx
                    obs, reset_info = env.reset(seed=reset_seed)
                    info["reset_info"] = dict(reset_info)
                conn.send(
                    (
                        obs,
                        np.asarray(reward, dtype=np.float32),
                        term,
                        trunc,
                        _worker_critic_obs(env, critic_obs_dim),
                        info,
                    )
                )
            elif cmd == "critic_obs":
                conn.send(_worker_critic_obs(env, critic_obs_dim))
            elif cmd == "close":
                conn.send(None)
                break
            else:
                raise RuntimeError(f"unknown async vector env command {cmd!r}")
    except BaseException as exc:  # pragma: no cover - exercised through parent errors.
        try:
            conn.send(exc)
        finally:
            raise
    finally:
        env.close()
        conn.close()


class XushiVectorEnv:
    """Batched wrapper over independent env instances.

    The wrapped envs must expose per-agent actor observations and a
    caller-buffered ``build_critic_obs(out)`` method, as ``Phase4MappoEnv`` does.
    Terminal envs are reset immediately so rollouts can continue without a
    second trainer-side reset path; final observations and infos are preserved
    under Gymnasium-style ``final_*`` keys.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        critic_obs_dim: int,
        seed_base: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if not env_fns:
            raise ValueError("env_fns must contain at least one env factory")
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.critic_obs_dim = int(critic_obs_dim)
        self.seed_base = int(seed_base)
        self.auto_reset = bool(auto_reset)
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        first = self.envs[0]
        self.single_observation_space = first.observation_space
        self.single_action_space = first.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, *first.observation_space.shape),
            dtype=first.observation_space.dtype,
        )
        self.action_space = gym.spaces.Box(
            low=np.broadcast_to(
                first.action_space.low,
                self.observation_space.shape[:1] + first.action_space.shape,
            ),
            high=np.broadcast_to(
                first.action_space.high,
                self.observation_space.shape[:1] + first.action_space.shape,
            ),
            dtype=first.action_space.dtype,
        )
        self._last_obs: np.ndarray | None = None

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        obs_parts: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        self._episode_counts[:] = 0
        base_seed = self.seed_base if seed is None else int(seed)
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=base_seed + i)
            obs_parts.append(np.asarray(obs, dtype=self.single_observation_space.dtype))
            infos.append(dict(info))
        obs_batch = np.stack(obs_parts, axis=0)
        self._last_obs = obs_batch
        return obs_batch, self.critic_obs(), infos

    def critic_obs(self) -> np.ndarray:
        out = np.zeros((self.num_envs, self.critic_obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            env.build_critic_obs(out[i])
        return out

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        actions = np.asarray(actions, dtype=self.single_action_space.dtype)
        expected_shape = (self.num_envs, *self.single_action_space.shape)
        if actions.shape != expected_shape:
            raise ValueError(
                f"actions shape must be {expected_shape}, got {actions.shape}"
            )

        obs_parts: list[np.ndarray] = []
        reward_parts: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(actions[i])
            term = bool(term)
            trunc = bool(trunc)
            info = dict(info)
            terminated[i] = term
            truncated[i] = trunc
            reward_parts.append(np.asarray(reward, dtype=np.float32))
            if (term or trunc) and self.auto_reset:
                info["final_info"] = dict(info)
                info["final_observation"] = np.asarray(
                    obs, dtype=self.single_observation_space.dtype
                )
                self._episode_counts[i] += 1
                reset_seed = self.seed_base + 10_000 * int(self._episode_counts[i]) + i
                obs, reset_info = env.reset(seed=reset_seed)
                info["reset_info"] = dict(reset_info)
            obs_parts.append(np.asarray(obs, dtype=self.single_observation_space.dtype))
            infos.append(info)

        obs_batch = np.stack(obs_parts, axis=0)
        reward_batch = np.stack(reward_parts, axis=0)
        self._last_obs = obs_batch
        return obs_batch, reward_batch, terminated, truncated, self.critic_obs(), infos

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._last_obs = None


class XushiAsyncVectorEnv:
    """Multiprocessing vector wrapper with the same API as ``XushiVectorEnv``."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        *,
        critic_obs_dim: int,
        seed_base: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if not env_fns:
            raise ValueError("env_fns must contain at least one env factory")
        self.num_envs = len(env_fns)
        self.critic_obs_dim = int(critic_obs_dim)
        self.seed_base = int(seed_base)
        self.auto_reset = bool(auto_reset)
        probe = env_fns[0]()
        try:
            self.single_observation_space = probe.observation_space
            self.single_action_space = probe.action_space
        finally:
            probe.close()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, *self.single_observation_space.shape),
            dtype=self.single_observation_space.dtype,
        )
        self.action_space = gym.spaces.Box(
            low=np.broadcast_to(
                self.single_action_space.low,
                self.observation_space.shape[:1] + self.single_action_space.shape,
            ),
            high=np.broadcast_to(
                self.single_action_space.high,
                self.observation_space.shape[:1] + self.single_action_space.shape,
            ),
            dtype=self.single_action_space.dtype,
        )
        self._ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        self._closed = False
        for i, env_fn in enumerate(env_fns):
            parent_conn, child_conn = self._ctx.Pipe()
            proc = self._ctx.Process(
                target=_async_worker,
                args=(
                    child_conn,
                    env_fn,
                    self.critic_obs_dim,
                    self.seed_base,
                    i,
                    self.auto_reset,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)
        self._last_critic_obs = np.zeros((self.num_envs, self.critic_obs_dim), dtype=np.float32)

    def _recv(self, idx: int):
        msg = self._conns[idx].recv()
        if isinstance(msg, BaseException):
            raise RuntimeError(f"async env worker {idx} failed") from msg
        return msg

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("vector env is closed")
        for conn in self._conns:
            conn.send(("reset", seed))
        obs_parts: list[np.ndarray] = []
        critic_parts: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            obs, critic_obs, info = self._recv(i)
            obs_parts.append(np.asarray(obs, dtype=self.single_observation_space.dtype))
            critic_parts.append(np.asarray(critic_obs, dtype=np.float32))
            infos.append(dict(info))
        obs_batch = np.stack(obs_parts, axis=0)
        self._last_critic_obs = np.stack(critic_parts, axis=0)
        return obs_batch, self._last_critic_obs.copy(), infos

    def critic_obs(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("vector env is closed")
        for conn in self._conns:
            conn.send(("critic_obs", None))
        self._last_critic_obs = np.stack(
            [np.asarray(self._recv(i), dtype=np.float32) for i in range(self.num_envs)],
            axis=0,
        )
        return self._last_critic_obs.copy()

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        if self._closed:
            raise RuntimeError("vector env is closed")
        actions = np.asarray(actions, dtype=self.single_action_space.dtype)
        expected_shape = (self.num_envs, *self.single_action_space.shape)
        if actions.shape != expected_shape:
            raise ValueError(
                f"actions shape must be {expected_shape}, got {actions.shape}"
            )
        for i, conn in enumerate(self._conns):
            conn.send(("step", actions[i]))

        obs_parts: list[np.ndarray] = []
        reward_parts: list[np.ndarray] = []
        critic_parts: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        for i in range(self.num_envs):
            obs, reward, term, trunc, critic_obs, info = self._recv(i)
            obs_parts.append(np.asarray(obs, dtype=self.single_observation_space.dtype))
            reward_parts.append(np.asarray(reward, dtype=np.float32))
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            critic_parts.append(np.asarray(critic_obs, dtype=np.float32))
            infos.append(dict(info))
        obs_batch = np.stack(obs_parts, axis=0)
        reward_batch = np.stack(reward_parts, axis=0)
        self._last_critic_obs = np.stack(critic_parts, axis=0)
        return (
            obs_batch,
            reward_batch,
            terminated,
            truncated,
            self._last_critic_obs.copy(),
            infos,
        )

    def close(self) -> None:
        if self._closed:
            return
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for i, conn in enumerate(self._conns):
            try:
                if self._procs[i].is_alive():
                    self._recv(i)
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            conn.close()
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._closed = True


def make_xushi_vector_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    *,
    critic_obs_dim: int,
    seed_base: int = 0,
    auto_reset: bool = True,
    backend: str = "sync",
) -> XushiVectorEnv | XushiAsyncVectorEnv:
    if backend == "sync":
        return XushiVectorEnv(
            env_fns,
            critic_obs_dim=critic_obs_dim,
            seed_base=seed_base,
            auto_reset=auto_reset,
        )
    if backend == "async":
        return XushiAsyncVectorEnv(
            env_fns,
            critic_obs_dim=critic_obs_dim,
            seed_base=seed_base,
            auto_reset=auto_reset,
        )
    raise ValueError(f"vector backend must be 'sync' or 'async', got {backend!r}")
