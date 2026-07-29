"""Opt-in Phase-4 MAPPO wrapper with current visible multi-enemy actor tokens."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    map_bounds_from_sim_cfg,
    zero_masked_enemy_tokens,
)
from xushi2.obs_manifest import CRITIC_DIM, critic_field_slice

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
    ) -> None:
        super().__init__()
        self._base = Phase4MappoEnv(
            dict(sim_cfg),
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
            opponent_policy=opponent_policy,
        )
        self._map_bounds = map_bounds_from_sim_cfg(sim_cfg)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = self._base.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
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
        critic = np.zeros((3, CRITIC_DIM), dtype=np.float32)
        self._base.build_critic_obs(critic[0])
        critic[1:] = critic[0]
        team_b = np.full(3, self._learner_team_str == "B", dtype=bool)
        visible = self._enemy_visibility_matrix(critic)
        converted = actor_obs_to_multi_enemy_entity_grid_obs(
            obs,
            critic_obs=critic,
            map_bounds=self._map_bounds,
            visible_radius=1.0,
            visible_override=visible,
            team_b_view=team_b,
        )
        return zero_masked_enemy_tokens(converted)

    def _enemy_visibility_matrix(self, critic: np.ndarray) -> np.ndarray:
        visible = np.zeros((3, 3), dtype=bool)
        for row, own_slot in enumerate(self._own_slots):
            native = _cpp.observable_enemy_slots(self._base._sim, own_slot)
            for enemy_idx, enemy_slot in enumerate(self._enemy_slots):
                alive = (
                    float(critic[row, critic_field_slice(f"enemy{enemy_idx}/alive_flag")][0]) > 0.5
                )
                visible[row, enemy_idx] = alive and bool(native[enemy_slot])
        return visible
