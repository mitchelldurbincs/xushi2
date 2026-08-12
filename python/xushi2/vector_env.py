"""Vector wrappers for Xushi multi-agent envs."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from .env_capabilities import (
    CURRICULUM_SETTERS,
    resolve_curriculum_setter,
    supported_curriculum_setters,
)

# Upper bound on how long the parent waits for a single worker reply.
#
# A worker that *dies* does not hang the parent: __init__ closes the parent's
# copy of the child pipe end, so a dead worker surfaces as EOF right away
# (measured: exitcode -9 reported in ~0.0s). What does hang the parent is a
# worker that is alive but wedged -- deadlocked, or stuck in native code --
# because nothing ever closes the pipe. An unbounded recv() then blocks
# forever and the run holds its allocation with no output and no error.
#
# Generous enough that a slow first step or a large model never trips it; the
# point is to bound the hang, not to police latency.
_WORKER_RECV_TIMEOUT_SECONDS: float = 300.0

# Shorter bound during shutdown, where a worker is expected to be exiting
# and a slow reply is not worth waiting five minutes for.
_WORKER_CLOSE_TIMEOUT_SECONDS: float = 10.0

# A dead worker's pipe closes before the OS reaps it, so Process.exitcode can
# still be None at the moment we notice. Join briefly before reporting, or the
# error says "exitcode=None" and tells the operator nothing.
_WORKER_REAP_TIMEOUT_SECONDS: float = 5.0


def _worker_critic_obs(env: gym.Env, critic_obs_dim: int) -> np.ndarray:
    out = np.zeros(critic_obs_dim, dtype=np.float32)
    env.build_critic_obs(out)
    return out


def _auto_reset_transition_metadata(
    *,
    obs: Any,
    info: dict[str, Any],
    term: bool,
    trunc: bool,
    auto_reset: bool,
    episode_count: int,
    seed_base: int,
    env_idx: int,
    obs_dtype: np.dtype[Any],
    final_critic_obs: np.ndarray | None = None,
) -> tuple[dict[str, Any], int, int | None, np.ndarray]:
    info_out = dict(info)
    obs_out = np.asarray(obs, dtype=obs_dtype)
    reset_seed: int | None = None
    next_episode_count = int(episode_count)
    if (term or trunc) and auto_reset:
        info_out["final_info"] = dict(info_out)
        info_out["final_observation"] = obs_out
        # The critic observation of the pre-reset state. Without it the value
        # of a truncated state is unrecoverable -- critic_obs() below is called
        # after the auto-reset and therefore describes the NEW episode -- and
        # MAPPO cannot bootstrap a time-limit cutoff correctly.
        if final_critic_obs is not None:
            info_out["final_critic_observation"] = np.asarray(
                final_critic_obs, dtype=np.float32
            )
        next_episode_count += 1
        reset_seed = seed_base + 10_000 * next_episode_count + env_idx
    return info_out, next_episode_count, reset_seed, obs_out


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
                # Read the terminal state's critic obs before any auto-reset
                # replaces it with the next episode's.
                final_critic = (
                    _worker_critic_obs(env, critic_obs_dim) if (term or trunc) else None
                )
                info, episode_count, reset_seed, obs_out = _auto_reset_transition_metadata(
                    obs=obs,
                    info=dict(info),
                    term=term,
                    trunc=trunc,
                    auto_reset=auto_reset,
                    episode_count=episode_count,
                    seed_base=seed_base,
                    env_idx=env_idx,
                    obs_dtype=env.observation_space.dtype,
                    final_critic_obs=final_critic,
                )
                if reset_seed is not None:
                    obs, reset_info = env.reset(seed=reset_seed)
                    info["reset_info"] = dict(reset_info)
                    obs_out = np.asarray(obs, dtype=env.observation_space.dtype)
                conn.send(
                    (
                        obs_out,
                        np.asarray(reward, dtype=np.float32),
                        term,
                        trunc,
                        _worker_critic_obs(env, critic_obs_dim),
                        info,
                    )
                )
            elif cmd == "critic_obs":
                conn.send(_worker_critic_obs(env, critic_obs_dim))
            elif cmd == "supported_curriculum_setters":
                conn.send(supported_curriculum_setters(env))
            elif cmd in CURRICULUM_SETTERS:
                # Generic dispatch: payload is always the argument tuple, so
                # adding a curriculum knob does not mean adding a branch here.
                setter = resolve_curriculum_setter(env, cmd)
                if setter is not None:
                    setter(*payload)
                conn.send(None)
            elif cmd == "set_opponent_bot":
                setter = getattr(env, "set_opponent_bot", None)
                if setter is not None:
                    setter(str(payload))
                conn.send(None)
            elif cmd == "set_opponent_handicap":
                bot, aim_noise, cadence = payload
                setter = getattr(env, "set_opponent_handicap", None)
                if setter is not None:
                    setter(str(bot), float(aim_noise), int(cadence))
                conn.send(None)
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

    Action spaces are currently constrained to ``gym.spaces.Box``. The
    wrappers assume a dense ndarray contract for action batching with stable
    ``shape``/``dtype`` and ``low``/``high`` bounds used for vectorized
    broadcast construction.
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
        if not isinstance(self.single_action_space, gym.spaces.Box):
            raise TypeError(
                "XushiVectorEnv currently supports only gym.spaces.Box action "
                "spaces. Expected a dense ndarray contract with shape/dtype and "
                "low/high bounds for vector batching; got "
                f"{type(self.single_action_space).__name__}. Add explicit "
                "serialization/stacking for non-Box spaces (for example Dict) "
                "before enabling them."
            )
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
        # Validate capability declarations up front: an env that neither
        # implements nor declares a curriculum setter fails here, at
        # construction, instead of silently dropping the curriculum later.
        # Runs after the space checks so their more specific errors win.
        self._supported_setters = frozenset.intersection(
            *(supported_curriculum_setters(env) for env in self.envs)
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
            raise ValueError(f"actions shape must be {expected_shape}, got {actions.shape}")

        obs_parts: list[np.ndarray] = []
        reward_parts: list[np.ndarray] = []
        infos: list[dict[str, Any]] = []
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(actions[i])
            term = bool(term)
            trunc = bool(trunc)
            final_critic = None
            if term or trunc:
                # Terminal-state critic obs, read before the auto-reset below
                # swaps in the next episode's.
                final_critic = np.zeros(self.critic_obs_dim, dtype=np.float32)
                env.build_critic_obs(final_critic)
            info, next_episode_count, reset_seed, obs_out = _auto_reset_transition_metadata(
                obs=obs,
                info=dict(info),
                term=term,
                trunc=trunc,
                auto_reset=self.auto_reset,
                episode_count=int(self._episode_counts[i]),
                seed_base=self.seed_base,
                env_idx=i,
                obs_dtype=self.single_observation_space.dtype,
                final_critic_obs=final_critic,
            )
            terminated[i] = term
            truncated[i] = trunc
            reward_parts.append(np.asarray(reward, dtype=np.float32))
            self._episode_counts[i] = next_episode_count
            if reset_seed is not None:
                obs, reset_info = env.reset(seed=reset_seed)
                info["reset_info"] = dict(reset_info)
                obs_out = np.asarray(obs, dtype=self.single_observation_space.dtype)
            obs_parts.append(obs_out)
            infos.append(info)

        obs_batch = np.stack(obs_parts, axis=0)
        reward_batch = np.stack(reward_parts, axis=0)
        self._last_obs = obs_batch
        return obs_batch, reward_batch, terminated, truncated, self.critic_obs(), infos

    def supported_curriculum_setters(self) -> frozenset[str]:
        """Setter names every wrapped env can apply.

        Intersected across envs so a caller that checks this cannot be told a
        knob is available when only some envs would honor it.
        """
        return self._supported_setters

    def _apply_setter(self, name: str, *args: Any) -> None:
        for env in self.envs:
            setter = resolve_curriculum_setter(env, name)
            if setter is not None:
                setter(*args)

    def set_team_spirit(self, value: float) -> None:
        self._apply_setter("set_team_spirit", float(value))

    def set_majority_on_point_alpha(self, value: float) -> None:
        self._apply_setter("set_majority_on_point_alpha", float(value))

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self._apply_setter("set_uncontested_on_point_alpha", float(value))

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        self._apply_setter(
            "set_objective_timing_seconds", float(unlock_seconds), float(capture_seconds)
        )

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        self._apply_setter("set_respawn_ticks", int(respawn_ticks))

    def set_opponent_bots(self, opponent_bots: Sequence[str]) -> None:
        """Push a per-env opponent assignment (opponent-mix curriculum)."""
        bots = [str(bot) for bot in opponent_bots]
        if len(bots) != len(self.envs):
            raise ValueError(
                f"opponent_bots must have one entry per env; got {len(bots)} "
                f"for {len(self.envs)} envs"
            )
        for env, bot in zip(self.envs, bots):
            setter = getattr(env, "set_opponent_bot", None)
            if setter is not None:
                setter(bot)

    def set_opponent_handicap(
        self, bot: str, aim_noise_radians: float, fire_cadence_ticks: int
    ) -> None:
        """Push the opponent-handicap curriculum to envs that support it."""
        for env in self.envs:
            setter = getattr(env, "set_opponent_handicap", None)
            if setter is not None:
                setter(str(bot), float(aim_noise_radians), int(fire_cadence_ticks))

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._last_obs = None


class XushiAsyncVectorEnv:
    """Multiprocessing vector wrapper with the same API as ``XushiVectorEnv``.

    Action spaces are currently constrained to ``gym.spaces.Box`` only.
    Non-Box spaces require explicit serialization/stacking support before they
    can be safely batched across workers.
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
        self.num_envs = len(env_fns)
        self.critic_obs_dim = int(critic_obs_dim)
        self.seed_base = int(seed_base)
        self.auto_reset = bool(auto_reset)
        probe = env_fns[0]()
        try:
            self.single_observation_space = probe.observation_space
            self.single_action_space = probe.action_space
            if not isinstance(self.single_action_space, gym.spaces.Box):
                raise TypeError(
                    "XushiAsyncVectorEnv currently supports only gym.spaces.Box "
                    "action spaces. Expected a dense ndarray contract with "
                    "shape/dtype and low/high bounds for worker batching; got "
                    f"{type(self.single_action_space).__name__}. Add explicit "
                    "serialization/stacking for non-Box spaces (for example "
                    "Dict) before enabling them."
                )
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
        # Ask each worker what its env can apply. This also runs the capability
        # validation inside the worker, so an env that neither implements nor
        # declares a curriculum setter surfaces here rather than dropping the
        # curriculum silently for the whole run.
        for conn in self._conns:
            conn.send(("supported_curriculum_setters", ()))
        self._supported_setters = frozenset.intersection(
            *(frozenset(self._recv(i)) for i in range(self.num_envs))
        )

    def _recv(self, idx: int, timeout: float | None = None):
        """Receive one worker reply, bounding the wait.

        Two distinct failure modes, neither previously diagnosable:

        - The worker died (signal, OOM kill). The parent closes its copy of
          the child pipe end at construction, so this surfaces as EOF rather
          than a hang -- but as a bare ``EOFError`` with an empty message and
          no indication of which worker or why.
        - The worker is alive but wedged (deadlock, stuck in native code).
          Nothing closes the pipe, so an unbounded ``recv()`` blocks forever
          and the training run hangs silently, holding its allocation.

        The timeout is resolved here rather than as a default argument so
        the module constant stays late-bound and overridable.
        """
        deadline = _WORKER_RECV_TIMEOUT_SECONDS if timeout is None else timeout
        conn = self._conns[idx]
        if not conn.poll(deadline):
            proc = self._procs[idx]
            if not proc.is_alive():
                raise RuntimeError(
                    f"async env worker {idx} died without reporting an error "
                    f"(exitcode={proc.exitcode}). A negative exitcode is a signal "
                    f"death; the usual cause is an abort inside the C++ sim from "
                    f"an invalid MatchConfig."
                )
            raise TimeoutError(
                f"async env worker {idx} did not respond within {deadline:.0f}s "
                f"(pid={proc.pid}, still alive). The worker is wedged rather than "
                f"dead -- inspect that pid before retrying."
            )
        try:
            msg = conn.recv()
        except EOFError as exc:
            proc = self._procs[idx]
            proc.join(timeout=_WORKER_REAP_TIMEOUT_SECONDS)
            raise RuntimeError(
                f"async env worker {idx} closed its pipe without replying "
                f"(exitcode={proc.exitcode}). A negative exitcode is a signal "
                f"death; the usual cause is an abort inside the C++ sim from "
                f"an invalid MatchConfig."
            ) from exc
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
            raise ValueError(f"actions shape must be {expected_shape}, got {actions.shape}")
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

    def supported_curriculum_setters(self) -> frozenset[str]:
        """Setter names every worker's env can apply.

        Queried once at construction; the workers' envs are built there and do
        not change identity afterwards.
        """
        return self._supported_setters

    def _broadcast_setter(self, name: str, *args: Any) -> None:
        """Send a curriculum setter to every worker and await each ack."""
        if self._closed:
            raise RuntimeError("vector env is closed")
        for conn in self._conns:
            conn.send((name, args))
        for i in range(self.num_envs):
            self._recv(i)

    def set_team_spirit(self, value: float) -> None:
        self._broadcast_setter("set_team_spirit", float(value))

    def set_majority_on_point_alpha(self, value: float) -> None:
        self._broadcast_setter("set_majority_on_point_alpha", float(value))

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self._broadcast_setter("set_uncontested_on_point_alpha", float(value))

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        self._broadcast_setter(
            "set_objective_timing_seconds", float(unlock_seconds), float(capture_seconds)
        )

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        self._broadcast_setter("set_respawn_ticks", int(respawn_ticks))

    def set_opponent_bots(self, opponent_bots: Sequence[str]) -> None:
        """Push a per-env opponent assignment (opponent-mix curriculum)."""
        if self._closed:
            raise RuntimeError("vector env is closed")
        bots = [str(bot) for bot in opponent_bots]
        if len(bots) != self.num_envs:
            raise ValueError(
                f"opponent_bots must have one entry per env; got {len(bots)} "
                f"for {self.num_envs} envs"
            )
        for conn, bot in zip(self._conns, bots):
            conn.send(("set_opponent_bot", bot))
        for i in range(self.num_envs):
            self._recv(i)

    def set_opponent_handicap(
        self, bot: str, aim_noise_radians: float, fire_cadence_ticks: int
    ) -> None:
        """Push the opponent-handicap curriculum to every worker."""
        if self._closed:
            raise RuntimeError("vector env is closed")
        payload = (str(bot), float(aim_noise_radians), int(fire_cadence_ticks))
        for conn in self._conns:
            conn.send(("set_opponent_handicap", payload))
        for i in range(self.num_envs):
            self._recv(i)

    def close(self) -> None:
        if self._closed:
            return
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, EOFError):
                conn.send(("close", None))
        for i, conn in enumerate(self._conns):
            try:
                if self._procs[i].is_alive():
                    self._recv(i, timeout=_WORKER_CLOSE_TIMEOUT_SECONDS)
            except (BrokenPipeError, EOFError, RuntimeError, TimeoutError):
                # Shutdown is best-effort; the join/terminate below is what
                # actually guarantees the workers go away.
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
    if backend == "sim_pool":
        return _make_sim_pool_vector_env(
            env_fns,
            critic_obs_dim=critic_obs_dim,
            seed_base=seed_base,
            auto_reset=auto_reset,
        )
    raise ValueError(
        f"vector backend must be 'sync', 'async', or 'sim_pool', got {backend!r}"
    )


def _make_sim_pool_vector_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    *,
    critic_obs_dim: int,
    seed_base: int,
    auto_reset: bool,
):
    """Build a SimPoolVectorEnv from the trainer's env factories.

    The trainer only hands us env factories, but the batched pool needs the
    underlying env parameters, so this reads them off the
    ``make_mappo_match_env`` partial that ``mappo_env_fn_from_config``
    produces. Any configuration the pool backend cannot reproduce exactly
    (self-play, snapshot opponents, mini-games, flat obs, legacy Python obs
    assembly) is rejected loudly rather than approximated.
    """
    import functools

    from envs.runtime_factory import make_mappo_match_env
    from xushi2.sim_pool_env import SimPoolVectorEnv

    fn = env_fns[0]
    if not (isinstance(fn, functools.partial) and fn.func is make_mappo_match_env):
        raise ValueError(
            "vector_env 'sim_pool' requires envs built by "
            "envs.runtime_factory.mappo_env_fn_from_config"
        )
    kw = dict(fn.keywords)

    def _reject(condition: bool, what: str) -> None:
        if condition:
            raise ValueError(
                f"vector_env 'sim_pool' does not support {what}; use "
                "vector_env 'sync' or 'async' for this config"
            )

    _reject(bool(kw.get("self_play")), "current self-play")
    _reject(kw.get("mini_game") not in (None, ""), "mini-games")
    _reject(int(kw.get("n_agents", 3)) != 3, "six-agent (phase11) envs")
    _reject(
        str(kw.get("actor_obs")) != "multi_enemy_entity_grid",
        f"actor_obs {kw.get('actor_obs')!r}",
    )
    _reject(
        not bool(kw.get("native_entity_obs", False)),
        "the legacy Python obs path (set env.native_entity_obs: true)",
    )
    opponent_bot = str(kw.get("opponent_bot", "basic"))
    _reject(opponent_bot == "snapshot", "snapshot opponents")
    fog_mode = kw.get("fog_mode")
    _reject(
        fog_mode not in (None, "", "none"),
        f"fog_mode {fog_mode!r} on the 3-agent wrapper",
    )
    _reject(bool(kw.get("map_randomization")), "map randomization")

    return SimPoolVectorEnv(
        num_envs=len(env_fns),
        sim_cfg=dict(kw.get("sim_cfg", {})),
        opponent_bot=opponent_bot,
        learner_team=str(kw.get("learner_team", "A")),
        reward_cfg=dict(kw.get("reward_cfg", {}) or {}),
        critic_obs_dim=critic_obs_dim,
        seed_base=seed_base,
        auto_reset=auto_reset,
    )
