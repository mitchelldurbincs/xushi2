"""Opt-in Phase-4 MAPPO wrapper with current visible multi-enemy actor tokens."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.entity_obs_native import phase4_multi_enemy_obs_config
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase4MultiEnemyMappoEnv"]


class Phase4MultiEnemyMappoEnv(gym.Env):
    """Phase 4 rules/actions with widened actor-side visible enemy tokens.

    The wrapper delegates simulation, reward, action semantics, and critic
    observation to ``Phase4MappoEnv``. Only actor observations are transformed,
    and enemy token fields are filled only when the corresponding enemy slot is
    currently visible to that actor through the native C++ line-of-sight mask.
    """

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
        opponent_policy: Any | None = None,
        opponent_snapshot_stochastic: bool = False,
    ) -> None:
        super().__init__()
        # The C++ ObservationEngine owns visibility (alive & native LoS, no
        # radius) and token zeroing (docs/observation_spec.md invariant 1).
        # Regression seal: python/tests/test_entity_obs_golden.py.
        self._obs_engine = _cpp.ObservationEngine(phase4_multi_enemy_obs_config())
        self._base = Phase4MappoEnv(
            dict(sim_cfg),
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
            opponent_snapshot_stochastic=opponent_snapshot_stochastic,
            opponent_policy=opponent_policy,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = self._base.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        self._obs_engine.reset()
        return self._convert(obs), dict(info)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._base.step(action)
        return self._convert(obs), reward, terminated, truncated, dict(info)

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._base.build_critic_obs(out)

    # Runtime curriculum / reward setters must be forwarded explicitly: the
    # vector env discovers them with getattr() and silently skips envs that
    # lack them. Before these delegations existed, every multi-enemy run
    # (including the 2026-06-10 conversion_v1 runs) trained with the
    # objective-timing anneal, team_spirit ramp, and eval alpha/timing
    # overrides silently dropped.
    def set_team_spirit(self, value: float) -> None:
        self._base.set_team_spirit(value)

    def set_majority_on_point_alpha(self, value: float) -> None:
        self._base.set_majority_on_point_alpha(value)

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self._base.set_uncontested_on_point_alpha(value)

    def set_objective_timing_ticks(self, unlock_ticks: int, capture_ticks: int) -> None:
        self._base.set_objective_timing_ticks(unlock_ticks, capture_ticks)

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None:
        self._base.set_objective_timing_seconds(unlock_seconds, capture_seconds)

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        self._base.set_respawn_ticks(respawn_ticks)

    def set_opponent_bot(self, opponent_bot: str) -> None:
        self._base.set_opponent_bot(opponent_bot)

    def set_opponent_handicap(
        self, bot: str, aim_noise_radians: float, fire_cadence_ticks: int
    ) -> None:
        self._base.set_opponent_handicap(bot, aim_noise_radians, fire_cadence_ticks)

    def close(self) -> None:
        self._base.close()

    @property
    def _sim(self):
        return self._base._sim

    @property
    def _own_slots(self) -> tuple[int, int, int]:
        return self._base._own_slots

    @property
    def _enemy_slots(self) -> tuple[int, int, int]:
        return self._base._enemy_slots

    @property
    def _learner_team_str(self) -> str:
        return self._base._learner_team_str

    def _convert(self, obs: np.ndarray) -> np.ndarray:
        if self._base._sim is None:
            raise RuntimeError("reset() must be called before converting obs")
        out = np.zeros((3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM), dtype=np.float32)
        for row, own_slot in enumerate(self._own_slots):
            self._obs_engine.build_entity_obs(self._base._sim, int(own_slot), out[row])
        return out
