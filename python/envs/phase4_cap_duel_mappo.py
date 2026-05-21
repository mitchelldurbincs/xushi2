"""Phase 4-compatible objective-coupled 1v1 cap duel mini-game.

This diagnostic keeps Phase 4 MAPPO actor, critic, and action tensor shapes
while activating one learner slot against one enemy near the objective. It
isolates the composition event Phase 4 currently misses: create local
advantage, stay on point, and convert that advantage into score.
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
from xushi2.self_play_schedule import SelfPlayMatch, SelfPlaySchedule

__all__ = ["Phase4CapDuelMappoEnv"]

_AIM_DELTA_LIMIT = float(np.pi / 4.0)
_ACTIVE_A = 0
_ACTIVE_B = 1
_INACTIVE = 2


class Phase4CapDuelMappoEnv(gym.Env):
    """Synthetic cap duel using Phase 4 MAPPO shapes.

    Slot 0 is the learner/team-A duelist. Slot 1 is active only for current
    self-play matches; anchor matches use a deterministic scripted
    recontester while masking slot 1 out of the loss. Slot 2 is always
    inactive and remains in the tensor shape for checkpoint compatibility.
    """

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
        hit_reward: float = 1.0,
        kill_bonus: float = 4.0,
        score_per_tick: float = 0.1,
        off_point_penalty: float = 0.0,
        time_penalty_per_decision: float = 0.0,
        self_play_schedule: dict[str, Any] | None = None,
        snapshot_league: dict[str, Any] | None = None,
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
        self.hit_reward = float(hit_reward)
        self.kill_bonus = float(kill_bonus)
        self.score_per_tick = float(score_per_tick)
        self.off_point_penalty = float(off_point_penalty)
        self.time_penalty_per_decision = float(time_penalty_per_decision)

        self._schedule = (
            None
            if self_play_schedule is None
            else SelfPlaySchedule.from_config(self_play_schedule, snapshot_league)
        )
        if self._schedule is not None and "snapshot" in self._schedule.weights:
            raise ValueError("cap_duel mini_game does not support snapshot self-play")

        self._move_speed = max(0.025, self.point_radius * 0.45)
        self._enemy_speed = max(0.01, self.point_radius * 0.20)
        self._hit_push = (
            self.point_radius
            + self._enemy_speed * float(max(1, self.enemy_recontest_delay + 1))
            + self.point_radius * 0.25
        )

        self._rng = np.random.default_rng(0)
        self._tick = 0
        self._score_ticks = np.zeros(2, dtype=np.int32)
        self._hp = np.full(2, self.enemy_hp, dtype=np.int32)
        self._alive = np.ones(2, dtype=np.bool_)
        self._respawn_timer = np.zeros(2, dtype=np.int32)
        self._off_point_decisions = np.zeros(2, dtype=np.int32)
        self._pos = np.zeros((2, 2), dtype=np.float32)
        self._last_move = np.zeros((2, 2), dtype=np.float32)
        self._hits = np.zeros(2, dtype=np.int32)
        self._fires = np.zeros(2, dtype=np.int32)
        self._misses = np.zeros(2, dtype=np.int32)
        self._kills = np.zeros(2, dtype=np.int32)
        self._score_events = np.zeros(2, dtype=np.int32)
        self._actor_obs_buf = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
        self._loss_mask = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._last_match = SelfPlayMatch(
            match_type="anchor",
            group="scripted",
            anchor_bot="recontest",
        )

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
        seed_int = 0 if seed is None else int(seed)
        self._last_match = (
            self._schedule.sample(seed_int)
            if self._schedule is not None
            else SelfPlayMatch(match_type="anchor", group="scripted", anchor_bot="recontest")
        )
        self._loss_mask = (
            np.array([1.0, 1.0, 0.0], dtype=np.float32)
            if self._last_match.match_type == "current"
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        self._tick = 0
        self._score_ticks.fill(0)
        self._hp.fill(self.enemy_hp)
        self._alive.fill(True)
        self._respawn_timer.fill(0)
        self._off_point_decisions.fill(0)
        self._hits.fill(0)
        self._fires.fill(0)
        self._misses.fill(0)
        self._kills.fill(0)
        self._score_events.fill(0)
        self._pos[_ACTIVE_A] = self._sample_near_point(0.85)
        self._pos[_ACTIVE_B] = self._sample_near_point(0.85)
        self._last_move.fill(0.0)
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(f"action shape must be (3, 6), got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        score_ready_a = self._score_ready(_ACTIVE_A)
        score_ready_b = (
            self._last_match.match_type == "current" and self._score_ready(_ACTIVE_B)
        )

        reward = np.zeros(3, dtype=np.float32)
        self._apply_agent_action(_ACTIVE_A, action[0], reward)
        if self._last_match.match_type == "current":
            self._apply_agent_action(_ACTIVE_B, action[1], reward)
        else:
            scripted = self._scripted_enemy_action()
            self._apply_agent_action(_ACTIVE_B, scripted, reward, reward_slot=None)

        if score_ready_a:
            self._score_ticks[_ACTIVE_A] += 1
            self._score_events[_ACTIVE_A] += 1
            reward[_ACTIVE_A] += self.score_per_tick
        if score_ready_b:
            self._score_ticks[_ACTIVE_B] += 1
            self._score_events[_ACTIVE_B] += 1
            reward[_ACTIVE_B] += self.score_per_tick

        for slot in (_ACTIVE_A, _ACTIVE_B):
            if self._loss_mask[slot] <= 0.0:
                continue
            reward[slot] -= self.time_penalty_per_decision
            if self.off_point_penalty > 0.0 and not self._is_on_point(self._pos[slot]):
                reward[slot] -= self.off_point_penalty

        self._tick += 1
        terminated = bool(
            self._score_ticks[_ACTIVE_A] >= self.score_ticks_to_clear
            or self._score_ticks[_ACTIVE_B] >= self.score_ticks_to_clear
        )
        truncated = self._tick >= self.episode_decisions and not terminated
        if not terminated and not truncated:
            self._advance_respawns()
            self._update_off_point_counters()
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), reward, terminated, truncated, self._make_info()

    def build_critic_obs(self, out: np.ndarray) -> None:
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        if out.dtype != np.float32:
            raise ValueError(f"out must be float32 ndarray, got {out.dtype}")
        if out.shape == (CRITIC_DIM,):
            self._fill_critic_view(out, actor_slot=0, own_idx=_ACTIVE_A)
            return
        if out.shape == (3 * CRITIC_DIM,):
            views = out.reshape(3, CRITIC_DIM)
            self._fill_critic_view(views[0], actor_slot=0, own_idx=_ACTIVE_A)
            self._fill_critic_view(views[1], actor_slot=1, own_idx=_ACTIVE_B)
            views[_INACTIVE].fill(0.0)
            return
        raise ValueError(
            "out must be float32 ndarray of shape "
            f"({CRITIC_DIM},) or ({3 * CRITIC_DIM},), got {out.shape}"
        )

    def set_team_spirit(self, value: float) -> None:
        return None

    def close(self) -> None:
        return None

    def _sample_near_point(self, radius_scale: float) -> np.ndarray:
        angle = float(self._rng.uniform(-np.pi, np.pi))
        radius = self.point_radius * min(1.0, max(0.0, float(radius_scale)))
        radius *= float(np.sqrt(self._rng.uniform(0.0, 1.0)))
        return np.array([np.cos(angle) * radius, np.sin(angle) * radius], dtype=np.float32)

    def _is_on_point(self, pos: np.ndarray) -> bool:
        return float(np.linalg.norm(pos)) <= self.point_radius

    def _score_ready(self, own_idx: int) -> bool:
        other_idx = 1 - own_idx
        if not self._alive[own_idx] or not self._is_on_point(self._pos[own_idx]):
            return False
        if not self._alive[other_idx]:
            return True
        if self._is_on_point(self._pos[other_idx]):
            return False
        return int(self._off_point_decisions[other_idx]) >= self.enemy_recontest_delay

    def _target_norm(self, own_idx: int) -> float:
        rel = self._pos[1 - own_idx] - self._pos[own_idx]
        angle_from_x = float(np.arctan2(float(rel[1]), float(rel[0])))
        return float(np.clip(angle_from_x / _AIM_DELTA_LIMIT, -1.0, 1.0))

    def _apply_agent_action(
        self,
        actor_idx: int,
        action: np.ndarray,
        reward: np.ndarray,
        *,
        reward_slot: int | None = None,
    ) -> None:
        if reward_slot is None:
            reward_slot = actor_idx
        if not self._alive[actor_idx]:
            self._last_move[actor_idx].fill(0.0)
            return

        move = np.asarray(action[:2], dtype=np.float32)
        move_norm = float(np.linalg.norm(move))
        if move_norm > 1.0:
            move = move / move_norm
        prev_pos = self._pos[actor_idx].copy()
        self._pos[actor_idx] = np.clip(
            self._pos[actor_idx] + move * self._move_speed,
            -1.0,
            1.0,
        ).astype(np.float32)
        self._last_move[actor_idx] = self._pos[actor_idx] - prev_pos

        target_idx = 1 - actor_idx
        fire = bool(action[3] >= 0.5)
        if not fire or not self._alive[target_idx]:
            return

        self._fires[actor_idx] += 1
        error = abs(float(action[2]) - self._target_norm(actor_idx))
        if error > self.hit_tolerance:
            self._misses[actor_idx] += 1
            return

        self._hits[actor_idx] += 1
        self._hp[target_idx] -= 1
        self._push_agent(target_idx, away_from=actor_idx)
        if reward_slot is not None:
            reward[reward_slot] += self.hit_reward
        if self._hp[target_idx] > 0:
            return

        self._kills[actor_idx] += 1
        self._alive[target_idx] = False
        self._respawn_timer[target_idx] = self.enemy_recontest_delay + 1
        self._off_point_decisions[target_idx] = 0
        if reward_slot is not None:
            reward[reward_slot] += self.kill_bonus

    def _push_agent(self, target_idx: int, *, away_from: int) -> None:
        direction = self._pos[target_idx] - self._pos[away_from]
        norm = float(np.linalg.norm(direction))
        direction = (
            np.array([1.0, 0.0], dtype=np.float32)
            if norm < 1.0e-6
            else direction / norm
        )
        prev = self._pos[target_idx].copy()
        self._pos[target_idx] = np.clip(
            self._pos[target_idx] + direction * self._hit_push,
            -1.0,
            1.0,
        ).astype(np.float32)
        self._last_move[target_idx] = self._pos[target_idx] - prev

    def _scripted_enemy_action(self) -> np.ndarray:
        action = np.zeros(6, dtype=np.float32)
        if not self._alive[_ACTIVE_B]:
            return action
        to_point = -self._pos[_ACTIVE_B]
        norm = float(np.linalg.norm(to_point))
        if norm > 1.0e-6:
            action[:2] = to_point / norm
        bot = self._last_match.anchor_bot or "recontest"
        if bot not in ("noop", "recontest") and self._alive[_ACTIVE_A]:
            action[2] = self._target_norm(_ACTIVE_B)
            action[3] = 1.0
        return action

    def _advance_respawns(self) -> None:
        for idx in (_ACTIVE_A, _ACTIVE_B):
            if self._alive[idx]:
                continue
            if self._respawn_timer[idx] > 0:
                self._respawn_timer[idx] -= 1
            if self._respawn_timer[idx] <= 0:
                self._alive[idx] = True
                self._hp[idx] = self.enemy_hp
                self._pos[idx] = self._sample_near_point(0.85)
                self._last_move[idx].fill(0.0)

    def _update_off_point_counters(self) -> None:
        for idx in (_ACTIVE_A, _ACTIVE_B):
            if self._alive[idx] and not self._is_on_point(self._pos[idx]):
                self._off_point_decisions[idx] += 1
            else:
                self._off_point_decisions[idx] = 0

    def _respawn_fraction(self, idx: int) -> float:
        if self.enemy_recontest_delay <= 0:
            return 0.0
        return float(self._respawn_timer[idx]) / float(self.enemy_recontest_delay)

    def _build_actor_obs_all(self) -> None:
        obs = self._actor_obs_buf
        obs.fill(0.0)
        self._fill_actor_slot(slot=0, own_idx=_ACTIVE_A)
        if self._last_match.match_type == "current":
            self._fill_actor_slot(slot=1, own_idx=_ACTIVE_B)

    def _fill_actor_slot(self, *, slot: int, own_idx: int) -> None:
        other_idx = 1 - own_idx
        obs = self._actor_obs_buf[slot]
        self_on_point = self._is_on_point(self._pos[own_idx])
        enemy_on_point = self._alive[other_idx] and self._is_on_point(self._pos[other_idx])
        contested = self_on_point and enemy_on_point
        own_score_frac = float(self._score_ticks[own_idx]) / float(self.score_ticks_to_clear)
        enemy_score_frac = float(self._score_ticks[other_idx]) / float(
            self.score_ticks_to_clear
        )

        obs[actor_field_slice("own_hp")] = (
            float(self._hp[own_idx]) / float(self.enemy_hp) if self._alive[own_idx] else 0.0
        )
        obs[actor_field_slice("own_velocity")] = self._last_move[own_idx]
        obs[actor_field_slice("own_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
        obs[actor_field_slice("own_position")] = self._pos[own_idx]
        obs[actor_field_slice("own_ammo")] = 1.0 if self._alive[own_idx] else 0.0
        obs[actor_field_slice("enemy_alive")] = 1.0 if self._alive[other_idx] else 0.0
        obs[actor_field_slice("enemy_respawn_timer")] = self._respawn_fraction(other_idx)
        obs[actor_field_slice("enemy_relative_position")] = (
            self._pos[other_idx] - self._pos[own_idx] if self._alive[other_idx] else 0.0
        )
        obs[actor_field_slice("enemy_hp")] = (
            float(self._hp[other_idx]) / float(self.enemy_hp)
            if self._alive[other_idx]
            else 0.0
        )
        obs[actor_field_slice("enemy_velocity")] = (
            self._last_move[other_idx] if self._alive[other_idx] else 0.0
        )
        obs[actor_field_slice("objective_owner_onehot")] = np.array(
            [1.0, 0.0, 0.0], dtype=np.float32
        )
        if self._score_ready(own_idx):
            cap_team = [0.0, 1.0, 0.0] if own_idx == _ACTIVE_A else [0.0, 0.0, 1.0]
        else:
            cap_team = [1.0, 0.0, 0.0]
        obs[actor_field_slice("cap_team_onehot")] = np.array(cap_team, dtype=np.float32)
        obs[actor_field_slice("cap_progress")] = own_score_frac
        obs[actor_field_slice("contested")] = 1.0 if contested else 0.0
        obs[actor_field_slice("objective_unlocked")] = 1.0
        obs[actor_field_slice("own_score")] = own_score_frac
        obs[actor_field_slice("enemy_score")] = enemy_score_frac
        obs[actor_field_slice("self_on_point")] = 1.0 if self_on_point else 0.0
        obs[actor_field_slice("enemy_on_point")] = 1.0 if enemy_on_point else 0.0
        obs[actor_field_slice("round_timer")] = float(self._tick) / float(
            self.episode_decisions
        )

    def _fill_critic_view(self, out: np.ndarray, *, actor_slot: int, own_idx: int) -> None:
        out.fill(0.0)
        if actor_slot == _INACTIVE or self._actor_obs_buf[actor_slot].sum() == 0.0:
            return
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
            dst = critic_field_slice(f"slot{actor_slot}/{name}")
            out[dst] = self._actor_obs_buf[actor_slot, src]
        other_idx = 1 - own_idx
        out[critic_field_slice("enemy0/world_position")] = self._pos[other_idx]
        out[critic_field_slice("enemy0/world_velocity")] = self._last_move[other_idx]
        out[critic_field_slice("enemy0/world_aim_unit")] = np.array([0.0, 1.0], dtype=np.float32)
        out[critic_field_slice("enemy0/hp_normalized")] = float(self._hp[other_idx]) / float(
            self.enemy_hp
        )
        out[critic_field_slice("enemy0/alive_flag")] = 1.0 if self._alive[other_idx] else 0.0
        out[critic_field_slice("enemy0/respawn_timer")] = self._respawn_fraction(other_idx)
        out[critic_field_slice("enemy0/ammo")] = 1.0 if self._alive[other_idx] else 0.0
        out[critic_field_slice("cap_progress_ticks")] = float(self._score_ticks[own_idx])
        out[critic_field_slice("team_a_score_ticks")] = float(self._score_ticks[_ACTIVE_A])
        out[critic_field_slice("team_b_score_ticks")] = float(self._score_ticks[_ACTIVE_B])
        out[critic_field_slice("tick_raw")] = float(self._tick)

    def _make_info(self) -> dict[str, Any]:
        if self._score_ticks[_ACTIVE_A] >= self.score_ticks_to_clear:
            winner = "A"
        elif self._score_ticks[_ACTIVE_B] >= self.score_ticks_to_clear:
            winner = "B"
        else:
            winner = "Neutral"
        return {
            "tick": int(self._tick),
            "state_hash": (
                f"cap-duel-{self._tick}-{int(self._score_ticks[0])}-"
                f"{int(self._score_ticks[1])}-{int(self._kills[0])}-"
                f"{int(self._kills[1])}-{int(self._hits.sum())}-"
                f"{int(self._fires.sum())}-{self._last_match.match_type}"
            ),
            "team_a_score": float(self._score_ticks[_ACTIVE_A]),
            "team_b_score": float(self._score_ticks[_ACTIVE_B]),
            "team_a_kills": int(self._kills[_ACTIVE_A]),
            "team_b_kills": int(self._kills[_ACTIVE_B]),
            "winner": winner,
            "learner_team": "A",
            "match_type": self._last_match.match_type,
            "schedule": self._schedule.summary if self._schedule is not None else "anchor:1",
            "loss_mask": self._loss_mask.copy(),
            "snapshot_path": self._last_match.snapshot_path,
            "snapshot_group": self._last_match.group,
            "anchor_bot": self._last_match.anchor_bot,
            "cap_duel_score_ticks": int(self._score_ticks[_ACTIVE_A]),
            "cap_duel_enemy_score_ticks": int(self._score_ticks[_ACTIVE_B]),
            "cap_duel_score_events": int(self._score_events[_ACTIVE_A]),
            "cap_duel_hits": int(self._hits[_ACTIVE_A]),
            "cap_duel_fires": int(self._fires[_ACTIVE_A]),
            "cap_duel_misses": int(self._misses[_ACTIVE_A]),
            "cap_duel_kills": int(self._kills[_ACTIVE_A]),
            "cap_duel_enemy_alive": bool(self._alive[_ACTIVE_B]),
            "cap_duel_enemy_on_point": bool(
                self._alive[_ACTIVE_B] and self._is_on_point(self._pos[_ACTIVE_B])
            ),
            "cap_duel_self_on_point": bool(self._is_on_point(self._pos[_ACTIVE_A])),
            "cap_duel_hit_rate": (
                float(self._hits[_ACTIVE_A]) / float(self._fires[_ACTIVE_A])
                if self._fires[_ACTIVE_A]
                else 0.0
            ),
        }
