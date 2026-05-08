"""Phase-10 MAPPO env wrapper that enables the target-slot action factor.

The current simulator still consumes the Phase-1 six-control action surface.
This wrapper accepts a seventh categorical target-token field, records it in
``info``, and forwards all controls into the Phase-8 observation stack. The
five categories match the appended token mask: self, three enemies, objective.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv
from xushi2.entity_obs import ENTITY_TOKEN_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import CRITIC_DIM

__all__ = ["Phase10TargetSlotMappoEnv"]

TARGET_SLOT_MASK_DIM: int = MULTI_ENEMY_TOKEN_COUNT
PHASE10_TARGET_OBS_DIM: int = MULTI_ENEMY_ENTITY_GRID_OBS_DIM + TARGET_SLOT_MASK_DIM


class Phase10TargetSlotMappoEnv(gym.Env):
    """Phase-8 env stack plus a categorical target-token action column."""

    metadata = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = PHASE10_TARGET_OBS_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 7
    target_action_dim: int = TARGET_SLOT_MASK_DIM

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
        self._env = Phase8RandomMapMappoEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
            reward_cfg=reward_cfg,
            fog_mode=fog_mode,
            visible_radius=visible_radius,
            map_randomization=map_randomization,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, PHASE10_TARGET_OBS_DIM),
            dtype=np.float32,
        )
        low = np.concatenate(
            (
                self._env.action_space.low,
                np.zeros((3, 1), dtype=np.float32),
            ),
            axis=1,
        )
        high = np.concatenate(
            (
                self._env.action_space.high,
                np.full((3, 1), self.target_action_dim - 1, dtype=np.float32),
            ),
            axis=1,
        )
        self.action_space = spaces.Box(low=low, high=high, shape=(3, 7), dtype=np.float32)
        self._last_target_slots = np.zeros(3, dtype=np.int64)

    @property
    def last_map_bounds(self) -> dict[str, float] | None:
        return self._env.last_map_bounds

    @property
    def last_target_slots(self) -> np.ndarray:
        return self._last_target_slots.copy()

    def reset(self, *, seed=None, options=None):
        self._last_target_slots[:] = 0
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        info["target_slot_mask"] = self._target_slot_mask(obs)
        info["target_slots"] = self.last_target_slots
        return self._append_target_slot_mask(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 7):
            raise ValueError(f"action shape must be (3, 7), got {action.shape}")
        self._last_target_slots = (
            np.rint(action[:, 6])
            .clip(0, self.target_action_dim - 1)
            .astype(np.int64, copy=False)
        )
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        info["target_slot_mask"] = self._target_slot_mask(obs)
        info["target_slots"] = self.last_target_slots
        return self._append_target_slot_mask(obs), reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        self._env.build_critic_obs(out)

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _target_slot_mask(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != MULTI_ENEMY_ENTITY_GRID_OBS_DIM:
            raise ValueError(
                "base obs last dim must be "
                f"{MULTI_ENEMY_ENTITY_GRID_OBS_DIM}, got {obs.shape}"
            )
        mask_offset = MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM
        mask = obs[..., mask_offset : mask_offset + TARGET_SLOT_MASK_DIM]
        return mask.astype(np.float32, copy=True)

    @classmethod
    def _append_target_slot_mask(cls, obs: np.ndarray) -> np.ndarray:
        mask = cls._target_slot_mask(obs)
        return np.concatenate((obs, mask), axis=-1).astype(np.float32, copy=False)
