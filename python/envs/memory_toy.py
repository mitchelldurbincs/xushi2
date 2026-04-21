from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MemoryToyEnv(gym.Env):
    """Phase-2 memory sanity toy. See docs/memory_toy.md.

    Timing semantics:
        - ``reset()`` initializes ``_t = 0`` and returns the observation for
          tick 0.
        - Each ``step(action)`` applies the action *for the current tick*
          (i.e. the tick whose observation was last returned), then advances
          ``_t`` by one and returns the observation for the new tick.
        - The terminal tick is ``T - 1``. After the step whose action was for
          tick ``T - 1``, ``_t == T`` and ``terminated`` is ``True``.
        - Reward is zero on all non-terminal steps and
          ``max(-2.0, -||action - target||_2)`` on the terminal step.
    """

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 64, cue_visible_ticks: int = 4):
        super().__init__()
        if cue_visible_ticks >= episode_length:
            raise ValueError("cue_visible_ticks must be < episode_length")
        self.T = int(episode_length)
        self.k = int(cue_visible_ticks)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self._rng: np.random.Generator | None = None
        self._target: np.ndarray | None = None
        self._t: int = 0

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        # When seed is not None, refresh our local rng from it; otherwise
        # reuse the gym-managed ``self.np_random`` (which persists across
        # resets once seeded). This is required for VectorEnv auto-reset
        # determinism: auto-resets pass ``seed=None`` and must advance
        # the per-env RNG rather than spin up a fresh non-deterministic
        # one each time.
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._target = np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32,
        )
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        if self._target is None:
            raise RuntimeError("must call reset() before step()")
        # The action is taken at the current tick (_t). Check whether this is
        # the terminal tick *before* advancing.
        terminated = self._t >= self.T - 1
        reward = 0.0
        if terminated:
            a = np.asarray(action, dtype=np.float32).reshape(2)
            dist = float(np.linalg.norm(a - self._target))
            reward = max(-2.0, -dist)
        self._t += 1
        obs = self._obs()
        return obs, reward, bool(terminated), False, {}

    def _obs(self) -> np.ndarray:
        assert self._target is not None
        if self._t < self.k:
            return np.array(
                [self._target[0], self._target[1], 1.0], dtype=np.float32,
            )
        return np.zeros(3, dtype=np.float32)
