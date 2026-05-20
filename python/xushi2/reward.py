"""Phase-1 reward calculator."""

from __future__ import annotations

import numpy as np

__all__ = [
    "DAMAGE_DEALT_COEF_DEFAULT",
    "DEATH_PENALTY_DEFAULT",
    "DISTANCE_SHAPING_COEF_DEFAULT",
    "KILL_BONUS_DEFAULT",
    "ON_POINT_SHAPING_COEF_DEFAULT",
    "SCORE_PER_SECOND_DEFAULT",
    "SHAPING_CLIP_DEFAULT",
    "TERMINAL_LOSS_DEFAULT",
    "TERMINAL_WIN_DEFAULT",
    "TICK_HZ",
    "TIME_PENALTY_PER_SECOND_DEFAULT",
    "RewardCalculator",
]

from . import xushi2_cpp as _cpp
from .reward_components import CumulativeClipper, EventCounters, EventDeltaExtractor, ObsAccessor, ShapingTerms, TICK_HZ

SHAPING_CLIP_DEFAULT: float = 3.0
TERMINAL_WIN_DEFAULT: float = 10.0
TERMINAL_LOSS_DEFAULT: float = -10.0
KILL_BONUS_DEFAULT: float = 0.25
DEATH_PENALTY_DEFAULT: float = 0.25
SCORE_PER_SECOND_DEFAULT: float = 0.01
DISTANCE_SHAPING_COEF_DEFAULT: float = 0.0
ON_POINT_SHAPING_COEF_DEFAULT: float = 0.0
TIME_PENALTY_PER_SECOND_DEFAULT: float = 0.0
DAMAGE_DEALT_COEF_DEFAULT: float = 0.0


class RewardCalculator:
    def __init__(self, *, shaping_clip: float = SHAPING_CLIP_DEFAULT, terminal_win: float = TERMINAL_WIN_DEFAULT, terminal_loss: float = TERMINAL_LOSS_DEFAULT, kill_bonus: float = KILL_BONUS_DEFAULT, death_penalty: float = DEATH_PENALTY_DEFAULT, score_per_second: float = SCORE_PER_SECOND_DEFAULT, distance_shaping_coef: float = DISTANCE_SHAPING_COEF_DEFAULT, on_point_shaping_coef: float = ON_POINT_SHAPING_COEF_DEFAULT, time_penalty_per_second: float = TIME_PENALTY_PER_SECOND_DEFAULT, per_agent_rewards: bool = False, team_spirit: float = 0.0, damage_dealt_coef: float = DAMAGE_DEALT_COEF_DEFAULT) -> None:
        if shaping_clip <= 0.0:
            raise ValueError("shaping_clip must be > 0")
        if distance_shaping_coef < 0.0:
            raise ValueError("distance_shaping_coef must be >= 0")
        if on_point_shaping_coef < 0.0:
            raise ValueError("on_point_shaping_coef must be >= 0")
        if not 0.0 <= team_spirit <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {team_spirit}")
        if damage_dealt_coef < 0.0:
            raise ValueError("damage_dealt_coef must be >= 0")
        self._terminal_win = float(terminal_win)
        self._terminal_loss = float(terminal_loss)
        self._kill_bonus = float(kill_bonus)
        self._death_penalty = float(death_penalty)
        self._score_per_second = float(score_per_second)
        self._distance_shaping_coef = float(distance_shaping_coef)
        self._on_point_shaping_coef = float(on_point_shaping_coef)
        self._time_penalty_per_second = float(time_penalty_per_second)
        self._per_agent = bool(per_agent_rewards)
        self._team_spirit = float(team_spirit)
        self._damage_dealt_coef = float(damage_dealt_coef)
        self._extractor = EventDeltaExtractor(per_agent=self._per_agent)
        self._prev = EventCounters()
        self._clipper = CumulativeClipper(shaping_clip)
        self._obs = ObsAccessor(enabled=(self._distance_shaping_coef > 0.0 or self._on_point_shaping_coef > 0.0 or self._per_agent))
        self._obs_buf_a = self._obs.obs_buf_a
        self._obs_buf_b = self._obs.obs_buf_b
        self._pos_slice = self._obs.pos_slice

    def reset(self, sim) -> None:
        self._prev = self._extractor.read(sim)
        self._clipper.reset()

    def set_team_spirit(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {value}")
        self._team_spirit = float(value)

    def step(self, sim):
        return self._step_per_agent(sim) if self._per_agent else self._step_scalar(sim)

    def _step_scalar(self, sim) -> tuple[float, float]:
        now = self._extractor.read(sim)
        a_s, b_s, a_k, b_k = self._extractor.scalar_delta(now, self._prev)
        raw_a = ShapingTerms.score_kill_death_scalar(a_s, b_s, a_k, b_k, score_per_second=self._score_per_second, kill_bonus=self._kill_bonus, death_penalty=self._death_penalty)
        raw_a += self._obs.distance_term(sim, self._distance_shaping_coef)
        raw_a += self._obs.on_point_term(sim, self._on_point_shaping_coef)
        raw_b = -raw_a
        tp = ShapingTerms.time_penalty_per_tick(self._time_penalty_per_second)
        raw_a += tp
        raw_b += tp
        reward_a = self._clipper.apply_clip(raw_a, "a")
        reward_b = self._clipper.apply_clip(raw_b, "b")
        self._prev = now
        return reward_a, reward_b

    def _step_per_agent(self, sim) -> tuple[np.ndarray, np.ndarray]:
        now = self._extractor.read(sim)
        a_s = (now.a_score_ticks - self._prev.a_score_ticks) / float(TICK_HZ)
        b_s = (now.b_score_ticks - self._prev.b_score_ticks) / float(TICK_HZ)
        kills_delta_slot = (now.kills_by_slot - self._prev.kills_by_slot).astype(np.float32)
        deaths_delta_slot = (now.deaths_by_slot - self._prev.deaths_by_slot).astype(np.float32)
        raw_a = np.zeros(3, dtype=np.float32)
        raw_b = np.zeros(3, dtype=np.float32)
        raw_a += self._kill_bonus * kills_delta_slot[0:3]
        raw_b += self._kill_bonus * kills_delta_slot[3:6]
        raw_a -= self._death_penalty * deaths_delta_slot[0:3]
        raw_b -= self._death_penalty * deaths_delta_slot[3:6]
        dmg_a, dmg_b = ShapingTerms.damage_by_slot(now, self._prev, self._damage_dealt_coef)
        raw_a += dmg_a
        raw_b += dmg_b
        if a_s != 0.0:
            shares_a = self._obs.on_point_shares(sim, (0, 1, 2))
            raw_a += self._score_per_second * a_s * shares_a
            raw_b -= (self._score_per_second * a_s) / 3.0
        if b_s != 0.0:
            shares_b = self._obs.on_point_shares(sim, (3, 4, 5))
            raw_b += self._score_per_second * b_s * shares_b
            raw_a -= (self._score_per_second * b_s) / 3.0
        sym = self._obs.distance_term(sim, self._distance_shaping_coef) + self._obs.on_point_term(sim, self._on_point_shaping_coef)
        raw_a += sym
        raw_b -= sym
        tp = ShapingTerms.time_penalty_per_tick(self._time_penalty_per_second)
        raw_a += tp
        raw_b += tp
        if self._team_spirit > 0.0:
            tau = self._team_spirit
            raw_a = (1.0 - tau) * raw_a + tau * float(raw_a.mean())
            raw_b = (1.0 - tau) * raw_b + tau * float(raw_b.mean())
        self._clipper.scale_to_clipped_sum(raw_a, "a")
        self._clipper.scale_to_clipped_sum(raw_b, "b")
        self._prev = now
        return raw_a, raw_b

    def add_terminal(self, sim):
        if not sim.episode_over:
            raise RuntimeError("add_terminal called before episode_over; step until terminal before querying terminal rewards")
        winner = sim.winner
        if winner == _cpp.Team.A:
            ta, tb = self._terminal_win, self._terminal_loss
        elif winner == _cpp.Team.B:
            ta, tb = self._terminal_loss, self._terminal_win
        else:
            ta, tb = 0.0, 0.0
        if self._per_agent:
            return np.full(3, ta, dtype=np.float32), np.full(3, tb, dtype=np.float32)
        return ta, tb

    @property
    def cumulative_shaped_a(self) -> float:
        return self._clipper.cumulative_shaped_a

    @property
    def cumulative_shaped_b(self) -> float:
        return self._clipper.cumulative_shaped_b
