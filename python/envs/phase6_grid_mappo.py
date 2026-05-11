"""Phase-6 MAPPO env wrapper with entity tokens plus egocentric grid."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.grid_obs import ENTITY_GRID_OBS_DIM, actor_obs_to_entity_grid_obs
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase6GridMappoEnv"]


class Phase6GridMappoEnv(gym.Env):
    """3v3 MAPPO env that adds a compact grid branch to Phase-5 tokens."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = ENTITY_GRID_OBS_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._base = Phase4MappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = self._base.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        return actor_obs_to_entity_grid_obs(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._base.step(action)
        return actor_obs_to_entity_grid_obs(obs), reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._base.build_critic_obs(out)

    def close(self) -> None:
        self._base.close()
