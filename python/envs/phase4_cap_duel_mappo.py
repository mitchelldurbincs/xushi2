"""Phase 4-compatible objective-coupled 1v1 cap duel mini-game.

This diagnostic keeps the Phase 4 MAPPO actor, critic, and action tensor
shapes while activating one learner slot against one scripted recontesting
enemy near the objective. It isolates the composition event Phase 4 currently
misses: create local advantage, step on point, and convert it into score.
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

__all__ = ["Phase4CapDuelMappoEnv"]

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4CapDuelMappoEnv(gym.Env):
    """Synthetic one-active-agent cap duel using Phase 4 MAPPO shapes."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = ACTOR_PHASE1_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        *,
        episode_decisions: int = 96,
        enemy_hp: int = 3,
        point_radius: float = 0.18,
        score_ticks_to_clear: int = 12,
        enemy_recontest_delay: int = 12,
        hit_tolerance: float = 0.12,
    ) -> None:
        super().__init__()
        if episode_decisions <= 0:
            raise ValueError("episode_decisions must be positive")
        if enemy_hp <= 0:
            raise ValueError("enemy_hp must be positive")
        if not (0.02 <= point_radius <= 0.75):
            raise ValueError("point_radius must be in [0.02, 0.75]")
        if score_ticks_to_clear <= 0:
            raise ValueError("score_ticks_to_clear must be positive")
        if enemy_recontest_delay < 0:
            raise ValueError("enemy_recontest_delay must be non-negative")
        if not (0.0 < hit_tolerance <= 1.0):
            raise ValueError("hit_tolerance must be in (0, 1]")

        self.episode_decisions = int(episode_decisions)
        self.enemy_hp = int(enemy_hp)
        self.point_radius = float(point_radius)
        self.score_ticks_to_clear = int(score_ticks_to_clear)
        self.enemy_recontest_delay = int(enemy_recontest_delay)
        self.hit_tolerance = float(hit_tolerance)

        self._move_speed = 0.08
        self._enemy_speed = 0.035
        self._hit_push = self.point_radius * 1.15
        self._rng = np.random.default_rng(0)
        self._tick = 0
        self._score_ticks = 0
        self._hp = self.enemy_hp
        self._enemy_alive = True
        self._enemy_respawn_timer = 0
        self._learner_pos = np.zeros(2, dtype=np.float32)
        self._enemy_pos = np.zeros(2, dtype=np.float32)
        self._last_move = np.zeros(2, dtype=np.float32)
        self._last_enemy_move = np.zeros(2, dtype=np.float32)
        self._hits = 0
        self._fires = 0
        self._misses = 0
        self._kills = 0
        self._score_events = 0
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
        self._score_ticks = 0
        self._hits = 0
        self._fires = 0
        self._misses = 0
        self._kills = 0
        self._score_events = 0
        self._enemy_alive = True
        self._enemy_respawn_timer = 0
        self._hp = self.enemy_hp
        self._learner_pos = self._sample_near_point(1.25)
        self._enemy_pos = self._sample_near_point(0.65)
        self._last_move.fill(0.0)
        self._last_enemy_move.fill(0.0)
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(f"action shape must be (3, 6), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        move = np.asarray(action[0, :2], dtype=np.float32)
        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0:
            move = move / move_norm
        prev_pos = self._learner_pos.copy()
        self._learner_pos = np.clip(
            self._learner_pos + move * self._move_speed,
            -1.0,
            1.0,
        ).astype(np.float32)
        self._last_move = self._learner_pos - prev_pos

        reward = np.zeros(3, dtype=np.float32)
        aim = float(action[0, 2])
        fire = bool(action[0, 3] >= 0.5)
        target_norm = self._target_norm()
        error = abs(aim - target_norm) if self._enemy_alive else 0.0
        if fire and self._enemy_alive:
            self._fires += 1
            if error <= self.hit_tolerance:
                self._hits += 1
                self._hp -= 1
                self._push_enemy()
                reward[0] += 0.6
                if self._hp <= 0:
                    self._kills += 1
                    self._enemy_alive = False
                    self._enemy_respawn_timer = self.enemy_recontest_delay
                    self._enemy_pos = self._enemy_pos.astype(np.float32)
                    reward[0] += 2.0
            else:
                self._misses += 1
                reward[0] -= 0.10
        elif self._enemy_alive:
            reward[0] -= 0.01

        if self._enemy_alive:
            reward[0] -= np.float32(0.03 * error)

        self_on_point = self._is_on_point(self._learner_pos)
        enemy_on_point = self._enemy_alive and self._is_on_point(self._enemy_pos)
        if self_on_point:
            reward[0] += 0.03
        if self_on_point and not enemy_on_point:
            self._score_ticks += 1
            self._score_events += 1
            reward[0] += 1.0

        self._tick += 1
        terminated = self._score_ticks >= self.score_ticks_to_clear
        truncated = self._tick >= self.episode_decisions and not terminated
        if not terminated and not truncated:
            self._advance_enemy()
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
        out[critic_field_slice("enemy0/world_position")] = self._enemy_pos
        out[critic_field_slice("enemy0/world_velocity")] = self._last_enemy_move
        out[critic_field_slice("enemy0/world_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
        out[critic_field_slice("enemy0/hp_normalized")] = float(self._hp) / float(self.enemy_hp)
        out[critic_field_slice("enemy0/alive_flag")] = 1.0 if self._enemy_alive else 0.0
        out[critic_field_slice("enemy0/respawn_timer")] = self._respawn_fraction()
        out[critic_field_slice("enemy0/ammo")] = 1.0
        out[critic_field_slice("cap_progress_ticks")] = float(self._score_ticks)
        out[critic_field_slice("team_a_score_ticks")] = float(self._score_ticks)
        out[critic_field_slice("team_b_score_ticks")] = 0.0
        out[critic_field_slice("tick_raw")] = float(self._tick)

    def set_team_spirit(self, value: float) -> None:
        return None

    def close(self) -> None:
        return None

    def _sample_near_point(self, radius_scale: float) -> np.ndarray:
        angle = float(self._rng.uniform(-np.pi, np.pi))
        radius = self.point_radius * float(radius_scale)
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)

    def _is_on_point(self, pos: np.ndarray) -> bool:
        return float(np.linalg.norm(pos)) <= self.point_radius

    def _target_norm(self) -> float:
        rel = self._enemy_pos - self._learner_pos
        angle_from_x = float(np.arctan2(float(rel[1]), float(rel[0])))
        return float(np.clip(angle_from_x / _AIM_DELTA_LIMIT, -1.0, 1.0))

    def _push_enemy(self) -> None:
        direction = self._enemy_pos - self._learner_pos
        norm = float(np.linalg.norm(direction))
        direction = (
            np.array([1.0, 0.0], dtype=np.float32)
            if norm < 1.0e-6
            else direction / norm
        )
        prev = self._enemy_pos.copy()
        self._enemy_pos = np.clip(self._enemy_pos + direction * self._hit_push, -1.0, 1.0).astype(
            np.float32
        )
        self._last_enemy_move = self._enemy_pos - prev

    def _advance_enemy(self) -> None:
        self._last_enemy_move.fill(0.0)
        if not self._enemy_alive:
            if self._enemy_respawn_timer > 0:
                self._enemy_respawn_timer -= 1
            if self._enemy_respawn_timer <= 0:
                self._enemy_alive = True
                self._hp = self.enemy_hp
                self._enemy_pos = self._sample_near_point(0.75)
            return
        direction = -self._enemy_pos
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-6:
            return
        prev = self._enemy_pos.copy()
        self._enemy_pos = (self._enemy_pos + direction / norm * self._enemy_speed).astype(
            np.float32
        )
        self._last_enemy_move = self._enemy_pos - prev

    def _respawn_fraction(self) -> float:
        if self.enemy_recontest_delay <= 0:
            return 0.0
        return float(self._enemy_respawn_timer) / float(self.enemy_recontest_delay)

    def _build_actor_obs_all(self) -> None:
        obs = self._actor_obs_buf
        obs.fill(0.0)
        self_on_point = self._is_on_point(self._learner_pos)
        enemy_on_point = self._enemy_alive and self._is_on_point(self._enemy_pos)
        contested = self_on_point and enemy_on_point
        score_frac = float(self._score_ticks) / float(self.score_ticks_to_clear)
        rel = self._enemy_pos - self._learner_pos
        target_norm = self._target_norm()
        target_angle = target_norm * _AIM_DELTA_LIMIT
        aim_unit = np.array([np.sin(target_angle), np.cos(target_angle)], dtype=np.float32)

        obs[:, actor_field_slice("own_hp")] = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
        obs[:, actor_field_slice("own_ammo")] = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
        obs[:, actor_field_slice("own_aim_unit")] = aim_unit
        obs[0, actor_field_slice("own_velocity")] = self._last_move
        obs[0, actor_field_slice("own_position")] = self._learner_pos
        obs[0, actor_field_slice("enemy_alive")] = 1.0 if self._enemy_alive else 0.0
        obs[0, actor_field_slice("enemy_respawn_timer")] = self._respawn_fraction()
        obs[0, actor_field_slice("enemy_relative_position")] = rel if self._enemy_alive else 0.0
        obs[0, actor_field_slice("enemy_hp")] = (
            float(self._hp) / float(self.enemy_hp) if self._enemy_alive else 0.0
        )
        obs[0, actor_field_slice("enemy_velocity")] = (
            self._last_enemy_move if self._enemy_alive else 0.0
        )
        obs[:, actor_field_slice("objective_owner_onehot")] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        obs[:, actor_field_slice("cap_team_onehot")] = np.array(
            [0.0, 1.0, 0.0] if self_on_point and not enemy_on_point else [1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        obs[:, actor_field_slice("cap_progress")] = score_frac
        obs[:, actor_field_slice("contested")] = 1.0 if contested else 0.0
        obs[:, actor_field_slice("objective_unlocked")] = 1.0
        obs[:, actor_field_slice("own_score")] = score_frac
        obs[:, actor_field_slice("enemy_score")] = 0.0
        obs[0, actor_field_slice("self_on_point")] = 1.0 if self_on_point else 0.0
        obs[0, actor_field_slice("enemy_on_point")] = 1.0 if enemy_on_point else 0.0
        obs[:, actor_field_slice("round_timer")] = float(self._tick) / float(
            self.episode_decisions
        )

    def _make_info(self) -> dict[str, Any]:
        winner = "A" if self._score_ticks >= self.score_ticks_to_clear else "Neutral"
        return {
            "tick": int(self._tick),
            "state_hash": (
                f"cap-duel-{self._tick}-{self._score_ticks}-{self._kills}-"
                f"{self._hits}-{self._fires}-{int(self._enemy_alive)}"
            ),
            "team_a_score": float(self._score_ticks),
            "team_b_score": 0.0,
            "team_a_kills": int(self._kills),
            "team_b_kills": 0,
            "winner": winner,
            "learner_team": "A",
            "cap_duel_score_ticks": int(self._score_ticks),
            "cap_duel_score_events": int(self._score_events),
            "cap_duel_hits": int(self._hits),
            "cap_duel_fires": int(self._fires),
            "cap_duel_misses": int(self._misses),
            "cap_duel_kills": int(self._kills),
            "cap_duel_enemy_alive": bool(self._enemy_alive),
            "cap_duel_enemy_on_point": bool(
                self._enemy_alive and self._is_on_point(self._enemy_pos)
            ),
            "cap_duel_self_on_point": bool(self._is_on_point(self._learner_pos)),
            "cap_duel_hit_rate": float(self._hits / self._fires) if self._fires else 0.0,
        }
