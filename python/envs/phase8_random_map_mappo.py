"""Phase-8 MAPPO env wrapper with deterministic per-episode map randomization."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.mappo_phase_common import RandomizedMapMixin
from envs.phase7_fog_mappo import Phase7FogMappoEnv
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase8RandomMapMappoEnv"]


class Phase8RandomMapMappoEnv(RandomizedMapMixin, gym.Env):
    """Phase-7 observation stack with deterministic randomized map bounds."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
        fog_mode: str = "team_shared",
        visible_radius: float = 0.65,
        map_randomization: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._init_map_randomization(sim_cfg, map_randomization)
        self._opponent_bot = opponent_bot
        self._learner_team = learner_team
        self._reward_cfg = dict(reward_cfg or {})
        self._fog_mode = fog_mode
        self._visible_radius = float(visible_radius)
        self._env: Phase7FogMappoEnv | None = None
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        probe = Phase7FogMappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
            fog_mode=fog_mode,
            visible_radius=visible_radius,
        )
        try:
            self.action_space = probe.action_space
        finally:
            probe.close()

    @property
    def last_map_bounds(self) -> dict[str, float] | None:
        return None if self._last_map_bounds is None else dict(self._last_map_bounds)

    @property
    def last_cover_markers(self) -> list[dict[str, float]]:
        return [dict(marker) for marker in self._last_cover_markers]

    @property
    def last_wall_segments(self) -> list[dict[str, float]]:
        return [dict(wall) for wall in self._last_wall_segments]

    @property
    def last_layout_hash(self) -> str | None:
        return self._last_layout_hash

    def reset(self, *, seed=None, options=None):
        seed_int = 0 if seed is None else int(seed)
        bounds, covers, walls, layout_hash = self._sample_map(seed_int)
        if self._env is not None:
            self._env.close()
        sim_cfg = self._randomized_sim_cfg(bounds, covers, walls)
        self._env = Phase7FogMappoEnv(
            sim_cfg,
            opponent_bot=self._opponent_bot,
            learner_team=self._learner_team,
            reward_cfg=self._reward_cfg,
            fog_mode=self._fog_mode,
            visible_radius=self._visible_radius,
        )
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        info.update(self._map_info())
        return obs, info

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        if self._last_map_bounds is not None:
            info.update(self._map_info())
        return obs, reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._env is None:
            raise RuntimeError("reset() must be called before build_critic_obs()")
        self._env.build_critic_obs(out)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
