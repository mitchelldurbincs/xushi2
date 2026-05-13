from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
    sim_cfg_with_map_bounds,
)


class BaseMappoPhaseEnv(gym.Env):
    """Reusable wrapper for MAPPO phases that only transform actor observations."""

    def __init__(self, *, base_env: gym.Env, actor_obs_dim: int) -> None:
        super().__init__()
        self._base = base_env
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, actor_obs_dim), dtype=np.float32
        )
        self.action_space = self._base.action_space

    def convert_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def update_info(self, info: dict[str, Any]) -> dict[str, Any]:
        return info

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        return self.convert_obs(obs), self.update_info(dict(info))

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._base.step(action)
        return self.convert_obs(obs), reward, terminated, truncated, self.update_info(dict(info))

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._base.build_critic_obs(out)

    def close(self) -> None:
        self._base.close()


class RandomizedMapMixin:
    def _init_map_randomization(self, sim_cfg: dict[str, Any], map_randomization: dict[str, Any] | None):
        self._base_sim_cfg = dict(sim_cfg)
        self._map_randomization = dict(map_randomization or {})
        self._last_map_bounds: dict[str, float] | None = None
        self._last_cover_markers: list[dict[str, float]] = []
        self._last_wall_segments: list[dict[str, float]] = []
        self._last_layout_hash: str | None = None

    def _sample_map(self, seed: int) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]], str]:
        bounds = randomized_map_bounds(seed, self._map_randomization)
        covers = randomized_cover_markers(seed, self._map_randomization)
        walls = randomized_wall_segments(seed, self._map_randomization)
        layout = map_layout_hash(bounds, covers, walls)
        self._last_map_bounds = bounds
        self._last_cover_markers = covers
        self._last_wall_segments = walls
        self._last_layout_hash = layout
        return bounds, covers, walls, layout

    def _map_info(self) -> dict[str, Any]:
        return {
            "map_bounds": dict(self._last_map_bounds or {}),
            "cover_markers": [dict(x) for x in self._last_cover_markers],
            "wall_segments": [dict(x) for x in self._last_wall_segments],
            "map_layout_hash": self._last_layout_hash,
        }

    def _randomized_sim_cfg(self, bounds, covers, walls) -> dict[str, Any]:
        sim_cfg = sim_cfg_with_map_bounds(self._base_sim_cfg, bounds)
        sim_cfg["randomize_map"] = True
        sim_cfg["cover_circles"] = [dict(marker) for marker in covers]
        sim_cfg["wall_segments"] = [dict(wall) for wall in walls]
        return sim_cfg
