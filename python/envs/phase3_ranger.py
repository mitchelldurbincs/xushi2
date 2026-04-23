from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from xushi2.env import XushiEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM


_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase3RangerEnv(gym.Env):
    """Flat-action wrapper around ``XushiEnv`` for recurrent PPO.

    Internal action layout:
        [move_x, move_y, aim_delta_unit, primary_fire, ability_1, ability_2]

    The first three coordinates are continuous in ``[-1, 1]``. The third is
    scaled to ``[-pi/4, pi/4]`` before calling the underlying env. The final
    three coordinates are interpreted as binary actions via ``>= 0.5``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
    ) -> None:
        super().__init__()
        self._env = XushiEnv(
            sim_cfg,
            opponent_bot=opponent_bot,
            learner_team=learner_team,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ACTOR_PHASE1_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, self._sanitize_info(info)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(
            self._action_to_dict(action)
        )
        return obs, reward, terminated, truncated, self._sanitize_info(info)

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _action_to_dict(action: np.ndarray | list[float]) -> dict[str, Any]:
        arr = np.asarray(action, dtype=np.float32).reshape(6)
        arr[:3] = np.clip(arr[:3], -1.0, 1.0)
        arr[3:] = np.clip(arr[3:], 0.0, 1.0)
        return {
            "move_x": float(arr[0]),
            "move_y": float(arr[1]),
            "aim_delta": float(arr[2] * _AIM_DELTA_LIMIT),
            "primary_fire": int(arr[3] >= 0.5),
            "ability_1": int(arr[4] >= 0.5),
            "ability_2": int(arr[5] >= 0.5),
        }

    @staticmethod
    def _sanitize_info(info: dict[str, Any]) -> dict[str, Any]:
        out = dict(info)
        if "state_hash" in out:
            out["state_hash"] = np.uint64(out["state_hash"])
        return out
