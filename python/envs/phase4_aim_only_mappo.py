"""Phase 4-compatible aim-only MAPPO mini-game.

This synthetic env preserves the Phase 4 flat actor observation, critic
observation, and action shapes while removing objective movement and opponent
policy complexity. It exists as an Escape Protocol 5.4 diagnostic: can the
current MAPPO actor learn to map visible enemy relative position to aim_delta
and primary_fire when the reward is direct hit feedback?
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

__all__ = ["Phase4AimOnlyMappoEnv"]

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4AimOnlyMappoEnv(gym.Env):
    """Synthetic 3-agent aim-and-fire mini-game with Phase 4 tensor shapes."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    # See xushi2.env_capabilities. Aim-only is a stationary shooting drill: no
    # C++ Sim, no RewardCalculator, no objective, and no deaths to respawn from.
    UNSUPPORTED_CURRICULUM_SETTERS: ClassVar[dict[str, str]] = {
        "set_majority_on_point_alpha": "no RewardCalculator; reward is computed in-env",
        "set_uncontested_on_point_alpha": "no RewardCalculator; reward is computed in-env",
        "set_objective_timing_seconds": "no objective in this mini-game",
        "set_respawn_ticks": "no respawn in this mini-game",
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
        target_angle_limit: float = 0.75,
        hit_tolerance: float = 0.12,
        hit_reward: float = 1.0,
        miss_penalty: float = 0.25,
        no_fire_penalty: float = 0.05,
        aim_error_coef: float = 0.1,
        resample_target_each_step: bool = True,
    ) -> None:
        super().__init__()
        if episode_decisions <= 0:
            raise ValueError("episode_decisions must be positive")
        if not (0.0 < target_distance <= 1.0):
            raise ValueError("target_distance must be in (0, 1]")
        if not (0.0 < target_angle_limit <= 1.0):
            raise ValueError("target_angle_limit must be in (0, 1]")
        if not (0.0 < hit_tolerance <= 2.0):
            raise ValueError("hit_tolerance must be in (0, 2]")

        self.episode_decisions = int(episode_decisions)
        self.target_distance = float(target_distance)
        self.target_angle_limit = float(target_angle_limit)
        self.hit_tolerance = float(hit_tolerance)
        self.hit_reward = float(hit_reward)
        self.miss_penalty = float(miss_penalty)
        self.no_fire_penalty = float(no_fire_penalty)
        self.aim_error_coef = float(aim_error_coef)
        self.resample_target_each_step = bool(resample_target_each_step)

        self._rng = np.random.default_rng(0)
        self._tick = 0
        self._target_norm = np.zeros(3, dtype=np.float32)
        self._hits = np.zeros(3, dtype=np.int32)
        self._fires = np.zeros(3, dtype=np.int32)
        self._misses = np.zeros(3, dtype=np.int32)
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
        self._hits.fill(0)
        self._fires.fill(0)
        self._misses.fill(0)
        self._sample_targets()
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(f"action shape must be (3, 6), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        aim = action[:, 2]
        fire = action[:, 3] >= 0.5
        error = np.abs(aim - self._target_norm)
        hit = fire & (error <= self.hit_tolerance)
        miss = fire & ~hit

        self._hits += hit.astype(np.int32)
        self._fires += fire.astype(np.int32)
        self._misses += miss.astype(np.int32)

        reward = (
            hit.astype(np.float32) * self.hit_reward
            - miss.astype(np.float32) * self.miss_penalty
            - (~fire).astype(np.float32) * self.no_fire_penalty
            - error.astype(np.float32) * self.aim_error_coef
        ).astype(np.float32)

        self._tick += 1
        truncated = self._tick >= self.episode_decisions
        terminated = False
        if self.resample_target_each_step and not truncated:
            self._sample_targets()
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
            for name, width in (
                ("own_hp", 1),
                ("own_velocity", 2),
                ("own_aim_unit", 2),
                ("own_position", 2),
                ("own_ammo", 1),
                ("own_reloading", 1),
                ("own_combat_roll_cd", 1),
                ("enemy_alive", 1),
                ("enemy_respawn_timer", 1),
                ("enemy_relative_position", 2),
                ("enemy_hp", 1),
                ("enemy_velocity", 2),
                ("objective_owner_onehot", 3),
                ("cap_team_onehot", 3),
                ("cap_progress", 1),
                ("contested", 1),
                ("objective_unlocked", 1),
                ("own_score", 1),
                ("enemy_score", 1),
                ("self_on_point", 1),
                ("enemy_on_point", 1),
                ("round_timer", 1),
            ):
                src = actor_field_slice(name)
                dst = critic_field_slice(f"slot{slot}/{name}")
                if src.stop - src.start != width or dst.stop - dst.start != width:
                    raise RuntimeError(f"manifest width mismatch for {name}")
                out[dst] = self._actor_obs_buf[slot, src]
        out[critic_field_slice("tick_raw")] = float(self._tick)

    def set_team_spirit(self, value: float) -> None:
        return None

    def close(self) -> None:
        return None

    def _sample_targets(self) -> None:
        self._target_norm = self._rng.uniform(
            -self.target_angle_limit,
            self.target_angle_limit,
            size=3,
        ).astype(np.float32)

    def _build_actor_obs_all(self) -> None:
        obs = self._actor_obs_buf
        obs.fill(0.0)
        target_angle = self._target_norm * _AIM_DELTA_LIMIT
        rel = np.stack(
            [
                np.cos(target_angle) * self.target_distance,
                np.sin(target_angle) * self.target_distance,
            ],
            axis=1,
        ).astype(np.float32)
        own_y = np.array([-0.25, 0.0, 0.25], dtype=np.float32)
        obs[:, actor_field_slice("own_hp")] = 1.0
        obs[:, actor_field_slice("own_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
        obs[:, actor_field_slice("own_position")] = np.stack(
            [np.zeros(3, dtype=np.float32), own_y], axis=1
        )
        obs[:, actor_field_slice("own_ammo")] = 1.0
        obs[:, actor_field_slice("enemy_alive")] = 1.0
        obs[:, actor_field_slice("enemy_relative_position")] = rel
        obs[:, actor_field_slice("enemy_hp")] = 1.0
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
        fires = int(self._fires.sum())
        hits = int(self._hits.sum())
        return {
            "tick": int(self._tick),
            "state_hash": f"aim-only-{self._tick}-{hits}-{fires}",
            "team_a_score": float(hits),
            "team_b_score": 0.0,
            "team_a_kills": int(hits),
            "team_b_kills": 0,
            "winner": "Neutral",
            "learner_team": "A",
            "aim_hits": hits,
            "aim_fires": fires,
            "aim_misses": int(self._misses.sum()),
            "aim_hit_rate": float(hits / fires) if fires else 0.0,
        }
