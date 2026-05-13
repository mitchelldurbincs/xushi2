"""Phase-6 MAPPO env wrapper with entity tokens plus egocentric grid."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from envs.mappo_phase_common import BaseMappoPhaseEnv
from envs.phase4_mappo import Phase4MappoEnv
from xushi2.grid_obs import ENTITY_GRID_OBS_DIM, actor_obs_to_entity_grid_obs
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase6GridMappoEnv"]


class Phase6GridMappoEnv(BaseMappoPhaseEnv):
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
        base = Phase4MappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
        )
        super().__init__(base_env=base, actor_obs_dim=ENTITY_GRID_OBS_DIM)

    def convert_obs(self, obs: np.ndarray) -> np.ndarray:
        return actor_obs_to_entity_grid_obs(obs)
