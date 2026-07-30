"""Phase 4-compatible 1v1 combat mini-game.

This diagnostic keeps Phase 4 MAPPO tensor shapes while activating only one
learner slot against one visible duel target. It strips away 3v3 coordination
and objective timing so we can test whether the recurrent policy can learn a
basic shoot/aim/kill loop before returning to the full simulator.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    CRITIC_DIM,
    actor_field_slice,
    critic_field_slice,
)

__all__ = ["Phase4Combat1v1MappoEnv"]

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4Combat1v1MappoEnv(gym.Env):
    """Synthetic one-active-agent duel using Phase 4 MAPPO shapes."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    # See xushi2.env_capabilities. Synthetic duel with no C++ Sim, no
    # RewardCalculator, and no objective at all, so these knobs have no target.
    UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {
        "set_majority_on_point_alpha": "no RewardCalculator; reward is computed in-env",
        "set_uncontested_on_point_alpha": "no RewardCalculator; reward is computed in-env",
        "set_objective_timing_seconds": "no objective in this mini-game",
        "set_respawn_ticks": "no C++ Sim; target respawn comes from mini_game_config",
    }

    n_agents: int = 3
    actor_obs_dim: int = ACTOR_PHASE1_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        *,
        episode_decisions: int = 64,
        target_distance: float = 0.45,
        target_angle_limit: float = 0.8,
        target_drift: float = 0.08,
        hit_tolerance: float = 0.12,
        target_hp: int = 3,
        hit_reward: float = 1.0,
        kill_reward: float = 4.0,
        miss_penalty: float = 0.15,
        no_fire_penalty: float = 0.02,
        aim_error_coef: float = 0.05,
    ) -> None:
        super().__init__()
        if episode_decisions <= 0:
            raise ValueError("episode_decisions must be positive")
        if not (0.0 < target_distance <= 1.0):
            raise ValueError("target_distance must be in (0, 1]")
        if not (0.0 < target_angle_limit <= 1.0):
            raise ValueError("target_angle_limit must be in (0, 1]")
        if target_hp <= 0:
            raise ValueError("target_hp must be positive")

        self.episode_decisions = int(episode_decisions)
        self.target_distance = float(target_distance)
        self.target_angle_limit = float(target_angle_limit)
        self.target_drift = float(target_drift)
        self.hit_tolerance = float(hit_tolerance)
        self.target_hp = int(target_hp)
        self.hit_reward = float(hit_reward)
        self.kill_reward = float(kill_reward)
        self.miss_penalty = float(miss_penalty)
        self.no_fire_penalty = float(no_fire_penalty)
        self.aim_error_coef = float(aim_error_coef)

        self._rng = np.random.default_rng(0)
        self._tick = 0
        self._target_norm = 0.0
        self._target_vel = 0.0
        self._hp = self.target_hp
        self._hits = 0
        self._fires = 0
        self._misses = 0
        self._kills = 0
        self._actor_obs_buf = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, ACTOR_PHASE1_DIM),
            dtype=np.float32,
        )
        low = np.tile(
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (3, 1),
        )
        high = np.tile(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (3, 1),
        )
        self.action_space = spaces.Box(low=low, high=high, shape=(3, 6), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._tick = 0
        self._hits = 0
        self._fires = 0
        self._misses = 0
        self._kills = 0
        self._respawn_target()
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(f"action shape must be (3, 6), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        aim = float(action[0, 2])
        fire = bool(action[0, 3] >= 0.5)
        error = abs(aim - self._target_norm)
        reward = np.zeros(3, dtype=np.float32)
        if fire:
            self._fires += 1
            if error <= self.hit_tolerance:
                self._hits += 1
                self._hp -= 1
                reward[0] += self.hit_reward
                if self._hp <= 0:
                    self._kills += 1
                    reward[0] += self.kill_reward
                    self._respawn_target()
            else:
                self._misses += 1
                reward[0] -= self.miss_penalty
        else:
            reward[0] -= self.no_fire_penalty
        reward[0] -= np.float32(error * self.aim_error_coef)

        self._tick += 1
        truncated = self._tick >= self.episode_decisions
        terminated = False
        if not truncated:
            self._advance_target()
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), reward, terminated, truncated, self._make_info()

    def build_critic_obs(self, out: np.ndarray) -> None:
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        if out.shape != (CRITIC_DIM,) or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 ndarray of shape ({CRITIC_DIM},), got {out.shape} {out.dtype}"
            )
        out.fill(0.0)
        for slot in range(3):
            for name in (
                "own_hp",
                "own_velocity",
                "own_aim_unit",
                "own_position",
                "own_ammo",
                "own_reloading",
                "own_combat_roll_cd",
                "enemy_alive",
                "enemy_respawn_timer",
                "enemy_relative_position",
                "enemy_hp",
                "enemy_velocity",
                "objective_owner_onehot",
                "cap_team_onehot",
                "cap_progress",
                "contested",
                "objective_unlocked",
                "own_score",
                "enemy_score",
                "self_on_point",
                "enemy_on_point",
                "round_timer",
            ):
                src = actor_field_slice(name)
                dst = critic_field_slice(f"slot{slot}/{name}")
                out[dst] = self._actor_obs_buf[slot, src]
        out[critic_field_slice("tick_raw")] = float(self._tick)

    def set_team_spirit(self, value: float) -> None:
        return None

    def close(self) -> None:
        return None

    def _respawn_target(self) -> None:
        self._hp = self.target_hp
        self._target_norm = float(
            self._rng.uniform(-self.target_angle_limit, self.target_angle_limit)
        )
        self._target_vel = float(self._rng.uniform(-self.target_drift, self.target_drift))

    def _advance_target(self) -> None:
        self._target_norm += self._target_vel
        if abs(self._target_norm) > self.target_angle_limit:
            self._target_norm = float(
                np.clip(self._target_norm, -self.target_angle_limit, self.target_angle_limit)
            )
            self._target_vel = -self._target_vel

    def _build_actor_obs_all(self) -> None:
        obs = self._actor_obs_buf
        obs.fill(0.0)
        target_angle = self._target_norm * _AIM_DELTA_LIMIT
        rel = np.array(
            [
                np.cos(target_angle) * self.target_distance,
                np.sin(target_angle) * self.target_distance,
            ],
            dtype=np.float32,
        )
        obs[:, actor_field_slice("own_hp")] = np.array(
            [[1.0], [0.0], [0.0]], dtype=np.float32
        )
        obs[:, actor_field_slice("own_ammo")] = np.array(
            [[1.0], [0.0], [0.0]], dtype=np.float32
        )
        obs[:, actor_field_slice("own_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
        obs[0, actor_field_slice("enemy_alive")] = 1.0
        obs[0, actor_field_slice("enemy_relative_position")] = rel
        obs[0, actor_field_slice("enemy_hp")] = float(self._hp) / float(self.target_hp)
        obs[:, actor_field_slice("objective_owner_onehot")] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        obs[:, actor_field_slice("cap_team_onehot")] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        obs[:, actor_field_slice("objective_unlocked")] = 1.0
        obs[:, actor_field_slice("round_timer")] = float(self._tick) / float(
            self.episode_decisions
        )

    def _make_info(self) -> dict[str, Any]:
        return {
            "tick": int(self._tick),
            "state_hash": f"combat-1v1-{self._tick}-{self._kills}-{self._hits}-{self._fires}",
            "team_a_score": float(self._kills),
            "team_b_score": 0.0,
            "team_a_kills": int(self._kills),
            "team_b_kills": 0,
            "winner": "Neutral",
            "learner_team": "A",
            "combat_1v1_hits": int(self._hits),
            "combat_1v1_fires": int(self._fires),
            "combat_1v1_misses": int(self._misses),
            "combat_1v1_hit_rate": float(self._hits / self._fires) if self._fires else 0.0,
        }
