"""Phase-5 MAPPO env wrapper with entity-token actor observations."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from envs.mappo_phase_common import BaseMappoPhaseEnv
from envs.phase4_mappo import VALID_OPPONENT_BOTS, Phase4MappoEnv
from xushi2.entity_obs import ENTITY_OBS_DIM, actor_obs_to_entity_obs
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase5EntityMappoEnv"]


class Phase5EntityMappoEnv(BaseMappoPhaseEnv):
    """3v3 MAPPO env that exposes flattened entity tokens to the actor.

    The sim, action space, reward, and centralized critic stay identical to
    Phase 4. Only the actor observation changes, which keeps the first Phase 5
    diagnostic focused on the attention-observation delta.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

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
                f"unknown opponent_bot {opponent_bot!r}; valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        base = Phase4MappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
        )
        super().__init__(base_env=base, actor_obs_dim=ENTITY_OBS_DIM)

    def convert_obs(self, obs: np.ndarray) -> np.ndarray:
        return actor_obs_to_entity_obs(obs)
