"""Phase-7 MAPPO env wrapper with diagnostic partial observation."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    map_bounds_from_sim_cfg,
    normalize_world_for_team,
)
from xushi2.obs_manifest import CRITIC_DIM, actor_field_slice, critic_field_slice

__all__ = ["Phase7FogMappoEnv"]


class Phase7FogMappoEnv(gym.Env):
    """3v3 MAPPO env with team-shared or per-agent diagnostic fog masking."""

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
        visible_radius: float = 0.6,
    ) -> None:
        super().__init__()
        if fog_mode not in ("team_shared", "per_agent"):
            raise ValueError("fog_mode must be 'team_shared' or 'per_agent'")
        self._team_shared = fog_mode == "team_shared"
        self._visible_radius = float(visible_radius)
        self._base = Phase4MappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = self._base.action_space
        self._map_bounds = map_bounds_from_sim_cfg(sim_cfg)
        self._last_seen_enemy_position = np.zeros((3, 3, 2), dtype=np.float32)
        self._last_seen_valid = np.zeros((3, 3), dtype=bool)

    def reset(self, *, seed=None, options=None):
        obs, info = self._base.reset(seed=seed, options=options)
        self._last_seen_enemy_position[:] = 0.0
        self._last_seen_valid[:] = False
        out = self._convert(obs)
        info = dict(info)
        self._add_last_seen_info(info)
        return out, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self._base.step(action)
        out = self._convert(obs)
        info = dict(info)
        self._add_last_seen_info(info)
        return out, reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._base.build_critic_obs(out)

    def close(self) -> None:
        self._base.close()

    def _add_last_seen_info(self, info: dict[str, Any]) -> None:
        info["last_seen_enemy_position"] = self._last_seen_enemy_position.copy()
        info["last_seen_valid"] = self._last_seen_valid.copy()

    def _convert(self, obs: np.ndarray) -> np.ndarray:
        critic = np.zeros((3, CRITIC_DIM), dtype=np.float32)
        self._base.build_critic_obs(critic[0])
        critic[1:] = critic[0]
        flat = np.asarray(obs, dtype=np.float32).reshape(3, -1)
        visible = self._enemy_visibility_matrix(flat, critic)
        for row in range(3):
            for enemy_idx in range(3):
                if visible[row, enemy_idx]:
                    pos = self._enemy_norm_position(critic[row], enemy_idx)
                    self._last_seen_enemy_position[row, enemy_idx] = pos
                    self._last_seen_valid[row, enemy_idx] = True
        return actor_obs_to_multi_enemy_entity_grid_obs(
            obs,
            critic_obs=critic,
            map_bounds=self._map_bounds,
            visible_radius=self._visible_radius,
            visible_override=visible,
            last_seen_enemy_position=self._last_seen_enemy_position,
            last_seen_valid=self._last_seen_valid,
        )

    def _enemy_visibility_matrix(self, flat_obs: np.ndarray, critic: np.ndarray) -> np.ndarray:
        if self._base._sim is None:
            raise RuntimeError("reset() must be called before converting obs")
        own_slots = self._base._own_slots
        enemy_slots = self._base._enemy_slots
        own_pos = flat_obs[:, actor_field_slice("own_position")]
        enemy_pos = np.zeros((3, 3, 2), dtype=np.float32)
        alive = np.zeros((3, 3), dtype=bool)
        for row in range(3):
            for enemy_idx in range(3):
                enemy_pos[row, enemy_idx] = self._enemy_norm_position(critic[row], enemy_idx)
                alive[row, enemy_idx] = (
                    critic[row, critic_field_slice(f"enemy{enemy_idx}/alive_flag")][0] > 0.5
                )
        radius = np.linalg.norm(enemy_pos - own_pos[:, None, :], axis=2) <= float(
            self._visible_radius
        )
        los = np.zeros((3, 3), dtype=bool)
        if self._team_shared:
            for enemy_idx, enemy_slot in enumerate(enemy_slots):
                for row in range(3):
                    los[row, enemy_idx] = any(
                        bool(_cpp.observable_enemy_slots(self._base._sim, ally)[enemy_slot])
                        for ally in own_slots
                    )
                    radius[row, enemy_idx] = bool(radius[:, enemy_idx].any())
        else:
            for row, own_slot in enumerate(own_slots):
                native = _cpp.observable_enemy_slots(self._base._sim, own_slot)
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[row, enemy_idx] = bool(native[enemy_slot])
        return alive & radius & los

    def _enemy_norm_position(self, critic: np.ndarray, enemy_idx: int) -> np.ndarray:
        return normalize_world_for_team(
            critic[critic_field_slice(f"enemy{enemy_idx}/world_position")],
            self._map_bounds,
            team_b_view=False,
        )
