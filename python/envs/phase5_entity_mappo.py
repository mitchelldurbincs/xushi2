"""Phase-5 MAPPO env wrapper with entity-token actor observations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv, VALID_OPPONENT_BOTS
from xushi2.entity_obs import ENTITY_OBS_DIM, actor_obs_to_entity_obs
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase5EntityMappoEnv"]


class Phase5EntityMappoEnv(gym.Env):
    """3v3 MAPPO env that exposes flattened entity tokens to the actor.

    The sim, action space, reward, and centralized critic stay identical to
    Phase 4. Only the actor observation changes, which keeps the first Phase 5
    diagnostic focused on the attention-observation delta.
    """

    metadata = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = ENTITY_OBS_DIM
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
        if opponent_bot not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {opponent_bot!r}; "
                f"valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        super().__init__()
        self._base = Phase4MappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3, ENTITY_OBS_DIM), dtype=np.float32
        )
        self.action_space = self._base.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        return actor_obs_to_entity_obs(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._base.step(action)
        return actor_obs_to_entity_obs(obs), reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._base.build_critic_obs(out)

    def close(self) -> None:
        self._base.close()
