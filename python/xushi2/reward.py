"""Phase-1 reward calculator.

Implements the reward scheme from docs/rl_design.md §5:

- **Terminal** reward dominates: +10 win, -10 loss, 0 draw. Not clipped.
- **Shaped** reward per team per step is the symmetrized event delta:
    team_reward = own_events - enemy_events
  where events are objective score gain and kills/deaths. Cumulative per
  team is clipped to ``[-shaping_clip, +shaping_clip]`` per episode
  (default 3.0) so that shaping cannot outrun the terminal signal.

The calculator is stateful: ``reset(sim)`` captures the starting counters;
``step(sim)`` returns ``(team_a_reward, team_b_reward)`` for the tick (or
decision) that just happened. ``add_terminal(sim)`` emits the terminal
reward at episode end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DAMAGE_DEALT_COEF_DEFAULT",
    "DEATH_PENALTY_DEFAULT",
    "DISTANCE_SHAPING_COEF_DEFAULT",
    "KILL_BONUS_DEFAULT",
    "MAJORITY_ON_POINT_COEF_DEFAULT",
    "ON_POINT_SHAPING_COEF_DEFAULT",
    "SCORE_PER_SECOND_DEFAULT",
    "SHAPING_CLIP_DEFAULT",
    "TERMINAL_LOSS_DEFAULT",
    "TERMINAL_WIN_DEFAULT",
    "TICK_HZ",
    "TIME_PENALTY_PER_SECOND_DEFAULT",
    "UNCONTESTED_ON_POINT_COEF_DEFAULT",
    "RewardCalculator",
]

from . import xushi2_cpp as _cpp
from .obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

TICK_HZ: int = _cpp.TICK_HZ
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
MAJORITY_ON_POINT_COEF_DEFAULT: float = 0.0
UNCONTESTED_ON_POINT_COEF_DEFAULT: float = 0.0
_CENTI_HP_PER_HP: float = 100.0

_TEAM_A_RANGER_SLOT: int = 0
_TEAM_B_RANGER_SLOT: int = 3


@dataclass
class _EventCounters:
    tick: int = 0
    a_score_ticks: int = 0
    b_score_ticks: int = 0
    a_kills: int = 0
    b_kills: int = 0
    kills_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )
    deaths_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )
    damage_dealt_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )


class _CounterReader:
    """Reads sim counters and computes deltas versus previous snapshot."""

    def __init__(self, *, per_agent: bool) -> None:
        self._per_agent = bool(per_agent)

    def read(self, sim) -> _EventCounters:
        out = _EventCounters(
            tick=int(getattr(sim, "tick", 0)),
            a_score_ticks=int(sim.team_a_score_ticks),
            b_score_ticks=int(sim.team_b_score_ticks),
            a_kills=int(sim.team_a_kills),
            b_kills=int(sim.team_b_kills),
        )
        if self._per_agent:
            out.kills_by_slot = np.asarray(sim.kills_by_slot, dtype=np.int64)
            out.deaths_by_slot = np.asarray(sim.deaths_by_slot, dtype=np.int64)
            damage_attr = getattr(sim, "damage_dealt_by_slot", None)
            if damage_attr is not None:
                out.damage_dealt_by_slot = np.asarray(damage_attr, dtype=np.int64)
        return out

    @staticmethod
    def scalar_delta(now: _EventCounters, prev: _EventCounters) -> tuple[float, float, int, int]:
        a_score_seconds = (now.a_score_ticks - prev.a_score_ticks) / float(TICK_HZ)
        b_score_seconds = (now.b_score_ticks - prev.b_score_ticks) / float(TICK_HZ)
        a_kills_delta = now.a_kills - prev.a_kills
        b_kills_delta = now.b_kills - prev.b_kills
        return a_score_seconds, b_score_seconds, a_kills_delta, b_kills_delta


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class _CumulativeClipper:
    def __init__(self, shaping_clip: float) -> None:
        self._shaping_clip = float(shaping_clip)
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

    def reset(self) -> None:
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

    @property
    def cumulative_shaped_a(self) -> float:
        return self._cum_shaped_a

    @property
    def cumulative_shaped_b(self) -> float:
        return self._cum_shaped_b

    def apply_clip(self, raw_delta: float, team: str) -> float:
        if team == "a":
            old = self._cum_shaped_a
            new = _clip(old + raw_delta, -self._shaping_clip, self._shaping_clip)
            self._cum_shaped_a = new
        else:
            old = self._cum_shaped_b
            new = _clip(old + raw_delta, -self._shaping_clip, self._shaping_clip)
            self._cum_shaped_b = new
        return new - old

    def scale_to_clipped_sum(self, raw: np.ndarray, team: str) -> None:
        team_step = float(raw.sum())
        clipped_step = self.apply_clip(team_step, team)
        if abs(team_step) > 1e-12 and clipped_step != team_step:
            raw *= clipped_step / team_step


class _ScalarRewardStrategy:
    def __init__(self, owner: "RewardCalculator") -> None:
        self._owner = owner

    def step(self, sim) -> tuple[float, float]:
        return self._owner._step_scalar(sim)


class _PerAgentRewardStrategy:
    def __init__(self, owner: "RewardCalculator") -> None:
        self._owner = owner

    def step(self, sim) -> tuple[np.ndarray, np.ndarray]:
        return self._owner._step_per_agent(sim)


class RewardCalculator:
    """Per-episode tracker of shaped + terminal rewards for both teams."""

    def __init__(
        self,
        *,
        shaping_clip: float = SHAPING_CLIP_DEFAULT,
        terminal_win: float = TERMINAL_WIN_DEFAULT,
        terminal_loss: float = TERMINAL_LOSS_DEFAULT,
        kill_bonus: float = KILL_BONUS_DEFAULT,
        death_penalty: float = DEATH_PENALTY_DEFAULT,
        score_per_second: float = SCORE_PER_SECOND_DEFAULT,
        distance_shaping_coef: float = DISTANCE_SHAPING_COEF_DEFAULT,
        on_point_shaping_coef: float = ON_POINT_SHAPING_COEF_DEFAULT,
        time_penalty_per_second: float = TIME_PENALTY_PER_SECOND_DEFAULT,
        per_agent_rewards: bool = False,
        team_spirit: float = 0.0,
        damage_dealt_coef: float = DAMAGE_DEALT_COEF_DEFAULT,
        majority_on_point_coef: float = MAJORITY_ON_POINT_COEF_DEFAULT,
        majority_on_point_distribute: str = "on_point",
        uncontested_on_point_coef: float = UNCONTESTED_ON_POINT_COEF_DEFAULT,
        uncontested_on_point_distribute: str = "on_point",
    ) -> None:
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
        if majority_on_point_coef < 0.0:
            raise ValueError("majority_on_point_coef must be >= 0")
        if uncontested_on_point_coef < 0.0:
            raise ValueError("uncontested_on_point_coef must be >= 0")
        if majority_on_point_distribute not in ("on_point", "uniform"):
            raise ValueError(
                "majority_on_point_distribute must be 'on_point' or 'uniform', "
                f"got {majority_on_point_distribute!r}"
            )
        if uncontested_on_point_distribute not in ("on_point", "uniform"):
            raise ValueError(
                "uncontested_on_point_distribute must be 'on_point' or 'uniform', "
                f"got {uncontested_on_point_distribute!r}"
            )
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
        self._majority_on_point_alpha = float(majority_on_point_coef)
        self._majority_on_point_distribute = str(majority_on_point_distribute)
        self._uncontested_on_point_alpha = float(uncontested_on_point_coef)
        self._uncontested_on_point_distribute = str(uncontested_on_point_distribute)
        self._last_majority_on_point_metrics = self._empty_majority_on_point_metrics()
        self._last_uncontested_on_point_metrics = (
            self._empty_uncontested_on_point_metrics()
        )
        self._counter_reader = _CounterReader(per_agent=self._per_agent)
        self._prev = _EventCounters()
        self._clipper = _CumulativeClipper(shaping_clip)
        self._scalar_strategy = _ScalarRewardStrategy(self)
        self._per_agent_strategy = _PerAgentRewardStrategy(self)

        needs_obs_bufs = (
            self._distance_shaping_coef > 0.0
            or self._on_point_shaping_coef > 0.0
            or self._majority_on_point_alpha > 0.0
            or self._uncontested_on_point_alpha > 0.0
            or self._per_agent
        )
        if needs_obs_bufs:
            self._pos_slice = actor_field_slice("own_position")
            self._on_point_slice = actor_field_slice("self_on_point")
            self._obs_bufs = [np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32) for _ in range(_cpp.AGENTS_PER_MATCH)]
            self._obs_buf_a = self._obs_bufs[_TEAM_A_RANGER_SLOT]
            self._obs_buf_b = self._obs_bufs[_TEAM_B_RANGER_SLOT]
        else:
            self._pos_slice = self._on_point_slice = self._obs_bufs = self._obs_buf_a = self._obs_buf_b = None

    def reset(self, sim) -> None:
        self._prev = self._counter_reader.read(sim)
        self._clipper.reset()
        self._last_majority_on_point_metrics = self._empty_majority_on_point_metrics()
        self._last_uncontested_on_point_metrics = (
            self._empty_uncontested_on_point_metrics()
        )

    def set_team_spirit(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {value}")
        self._team_spirit = float(value)

    def set_majority_on_point_alpha(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"majority_on_point alpha must be >= 0, got {value}")
        self._majority_on_point_alpha = float(value)

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"uncontested_on_point alpha must be >= 0, got {value}")
        self._uncontested_on_point_alpha = float(value)

    @property
    def majority_on_point_alpha(self) -> float:
        return self._majority_on_point_alpha

    @property
    def uncontested_on_point_alpha(self) -> float:
        return self._uncontested_on_point_alpha

    @staticmethod
    def _empty_majority_on_point_metrics() -> dict[str, float]:
        return {
            "majority_on_point_alpha": 0.0,
            "majority_on_point_count_a": 0.0,
            "majority_on_point_count_b": 0.0,
            "majority_on_point_advantage_a": 0.0,
            "majority_on_point_advantage_b": 0.0,
            "majority_on_point_reward_a": 0.0,
            "majority_on_point_reward_b": 0.0,
        }

    def majority_on_point_metrics(self) -> dict[str, float]:
        return dict(self._last_majority_on_point_metrics)

    @staticmethod
    def _empty_uncontested_on_point_metrics() -> dict[str, float]:
        return {
            "uncontested_on_point_alpha": 0.0,
            "uncontested_on_point_count_a": 0.0,
            "uncontested_on_point_count_b": 0.0,
            "uncontested_on_point_reward_a": 0.0,
            "uncontested_on_point_reward_b": 0.0,
        }

    def uncontested_on_point_metrics(self) -> dict[str, float]:
        return dict(self._last_uncontested_on_point_metrics)

    def _distance_shaping_delta(self, sim) -> float:
        if self._distance_shaping_coef <= 0.0:
            return 0.0
        _cpp.build_actor_obs(sim, _TEAM_A_RANGER_SLOT, self._obs_buf_a)
        _cpp.build_actor_obs(sim, _TEAM_B_RANGER_SLOT, self._obs_buf_b)
        pos_a = self._obs_buf_a[self._pos_slice]
        pos_b = self._obs_buf_b[self._pos_slice]
        dist_a = float(np.hypot(pos_a[0], pos_a[1]))
        dist_b = float(np.hypot(pos_b[0], pos_b[1]))
        return -self._distance_shaping_coef * (dist_a - dist_b)

    def _on_point_shaping_delta(self, sim) -> float:
        if self._on_point_shaping_coef <= 0.0:
            return 0.0
        on_a = self._team_on_point_fraction(sim, (0, 1, 2))
        on_b = self._team_on_point_fraction(sim, (3, 4, 5))
        return self._on_point_shaping_coef * (on_a - on_b)

    def _time_penalty_delta(self) -> float:
        if self._time_penalty_per_second == 0.0:
            return 0.0
        return -self._time_penalty_per_second / float(TICK_HZ)

    def _decision_seconds(self, now: _EventCounters) -> float:
        delta_ticks = int(now.tick - self._prev.tick)
        if delta_ticks <= 0:
            delta_ticks = 1
        return float(delta_ticks) / float(TICK_HZ)

    def _damage_delta_by_slot(self, now: _EventCounters) -> tuple[np.ndarray, np.ndarray]:
        if self._damage_dealt_coef <= 0.0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        damage_delta_slot = (now.damage_dealt_by_slot - self._prev.damage_dealt_by_slot).astype(np.float32)
        per_hp = self._damage_dealt_coef / _CENTI_HP_PER_HP
        return per_hp * damage_delta_slot[0:3], per_hp * damage_delta_slot[3:6]

    def _majority_on_point_by_team(
        self, sim, decision_seconds: float
    ) -> tuple[float, float]:
        if self._majority_on_point_alpha <= 0.0:
            metrics = self._empty_majority_on_point_metrics()
            metrics["majority_on_point_alpha"] = float(self._majority_on_point_alpha)
            self._last_majority_on_point_metrics = metrics
            return 0.0, 0.0
        values_a = self._slot_on_point_values(sim, (0, 1, 2))
        values_b = self._slot_on_point_values(sim, (3, 4, 5))
        count_a = float(values_a.sum()) if values_a is not None else 0.0
        count_b = float(values_b.sum()) if values_b is not None else 0.0
        advantage_a = max(0.0, count_a - count_b)
        advantage_b = max(0.0, count_b - count_a)
        reward_a = self._majority_on_point_alpha * advantage_a * decision_seconds
        reward_b = self._majority_on_point_alpha * advantage_b * decision_seconds
        self._last_majority_on_point_metrics = {
            "majority_on_point_alpha": float(self._majority_on_point_alpha),
            "majority_on_point_count_a": count_a,
            "majority_on_point_count_b": count_b,
            "majority_on_point_advantage_a": advantage_a,
            "majority_on_point_advantage_b": advantage_b,
            "majority_on_point_reward_a": reward_a,
            "majority_on_point_reward_b": reward_b,
        }
        return reward_a, reward_b

    def _majority_on_point_shares(self, sim, slots: tuple[int, int, int]) -> np.ndarray:
        if self._majority_on_point_distribute == "uniform":
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return self._on_point_shares(sim, slots)

    def _uncontested_on_point_by_team(
        self, sim, decision_seconds: float
    ) -> tuple[float, float]:
        if self._uncontested_on_point_alpha <= 0.0:
            metrics = self._empty_uncontested_on_point_metrics()
            metrics["uncontested_on_point_alpha"] = float(
                self._uncontested_on_point_alpha
            )
            self._last_uncontested_on_point_metrics = metrics
            return 0.0, 0.0
        values_a = self._slot_on_point_values(sim, (0, 1, 2))
        values_b = self._slot_on_point_values(sim, (3, 4, 5))
        count_a = float(values_a.sum()) if values_a is not None else 0.0
        count_b = float(values_b.sum()) if values_b is not None else 0.0
        reward_a = (
            self._uncontested_on_point_alpha * decision_seconds
            if count_a > 0.0 and count_b <= 0.0
            else 0.0
        )
        reward_b = (
            self._uncontested_on_point_alpha * decision_seconds
            if count_b > 0.0 and count_a <= 0.0
            else 0.0
        )
        self._last_uncontested_on_point_metrics = {
            "uncontested_on_point_alpha": float(self._uncontested_on_point_alpha),
            "uncontested_on_point_count_a": count_a,
            "uncontested_on_point_count_b": count_b,
            "uncontested_on_point_reward_a": reward_a,
            "uncontested_on_point_reward_b": reward_b,
        }
        return reward_a, reward_b

    def _uncontested_on_point_shares(
        self, sim, slots: tuple[int, int, int]
    ) -> np.ndarray:
        if self._uncontested_on_point_distribute == "uniform":
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return self._on_point_shares(sim, slots)

    def step(self, sim):
        if self._per_agent:
            return self._per_agent_strategy.step(sim)
        return self._scalar_strategy.step(sim)

    def _step_scalar(self, sim) -> tuple[float, float]:
        now = self._counter_reader.read(sim)
        a_score_seconds, b_score_seconds, a_kills_delta, b_kills_delta = _CounterReader.scalar_delta(now, self._prev)
        raw_a = (
            self._score_per_second * a_score_seconds
            - self._score_per_second * b_score_seconds
            + self._kill_bonus * a_kills_delta
            - self._death_penalty * b_kills_delta
        )
        raw_a += self._distance_shaping_delta(sim)
        raw_a += self._on_point_shaping_delta(sim)
        raw_b = -raw_a
        majority_a, majority_b = self._majority_on_point_by_team(
            sim, self._decision_seconds(now)
        )
        majority_delta = majority_a - majority_b
        raw_a += majority_delta
        raw_b -= majority_delta
        uncontested_a, uncontested_b = self._uncontested_on_point_by_team(
            sim, self._decision_seconds(now)
        )
        uncontested_delta = uncontested_a - uncontested_b
        raw_a += uncontested_delta
        raw_b -= uncontested_delta
        tp = self._time_penalty_delta()
        raw_a += tp
        raw_b += tp
        reward_a = self._clipper.apply_clip(raw_a, "a")
        reward_b = self._clipper.apply_clip(raw_b, "b")
        self._prev = now
        return reward_a, reward_b

    def _step_per_agent(self, sim) -> tuple[np.ndarray, np.ndarray]:
        now = self._counter_reader.read(sim)
        a_score_seconds = (now.a_score_ticks - self._prev.a_score_ticks) / float(TICK_HZ)
        b_score_seconds = (now.b_score_ticks - self._prev.b_score_ticks) / float(TICK_HZ)
        kills_delta_slot = (now.kills_by_slot - self._prev.kills_by_slot).astype(np.float32)
        deaths_delta_slot = (now.deaths_by_slot - self._prev.deaths_by_slot).astype(np.float32)
        raw_a = np.zeros(3, dtype=np.float32)
        raw_b = np.zeros(3, dtype=np.float32)
        raw_a += self._kill_bonus * kills_delta_slot[0:3]
        raw_b += self._kill_bonus * kills_delta_slot[3:6]
        raw_a -= self._death_penalty * deaths_delta_slot[0:3]
        raw_b -= self._death_penalty * deaths_delta_slot[3:6]
        dmg_a, dmg_b = self._damage_delta_by_slot(now)
        raw_a += dmg_a
        raw_b += dmg_b
        if a_score_seconds != 0.0:
            shares_a = self._on_point_shares(sim, (0, 1, 2))
            raw_a += self._score_per_second * a_score_seconds * shares_a
            raw_b -= (self._score_per_second * a_score_seconds) / 3.0
        if b_score_seconds != 0.0:
            shares_b = self._on_point_shares(sim, (3, 4, 5))
            raw_b += self._score_per_second * b_score_seconds * shares_b
            raw_a -= (self._score_per_second * b_score_seconds) / 3.0
        majority_a, majority_b = self._majority_on_point_by_team(
            sim, self._decision_seconds(now)
        )
        if majority_a != 0.0:
            raw_a += majority_a * self._majority_on_point_shares(sim, (0, 1, 2))
            raw_b -= majority_a / 3.0
        if majority_b != 0.0:
            raw_b += majority_b * self._majority_on_point_shares(sim, (3, 4, 5))
            raw_a -= majority_b / 3.0
        uncontested_a, uncontested_b = self._uncontested_on_point_by_team(
            sim, self._decision_seconds(now)
        )
        if uncontested_a != 0.0:
            raw_a += uncontested_a * self._uncontested_on_point_shares(
                sim, (0, 1, 2)
            )
            raw_b -= uncontested_a / 3.0
        if uncontested_b != 0.0:
            raw_b += uncontested_b * self._uncontested_on_point_shares(
                sim, (3, 4, 5)
            )
            raw_a -= uncontested_b / 3.0
        dist = self._distance_shaping_delta(sim)
        onp = self._on_point_shaping_delta(sim)
        raw_a += dist + onp
        raw_b -= dist + onp
        tp = self._time_penalty_delta()
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

    def _on_point_shares(self, sim, slots: tuple[int, int, int]) -> np.ndarray:
        uniform = np.full(3, 1.0 / 3.0, dtype=np.float32)
        values = self._slot_on_point_values(sim, slots)
        if values is None:
            return uniform
        shares = values.astype(np.float32, copy=True)
        total = float(shares.sum())
        if total <= 1e-12:
            return uniform
        return shares / total

    def _team_on_point_fraction(self, sim, slots: tuple[int, int, int]) -> float:
        values = self._slot_on_point_values(sim, slots)
        if values is None:
            return 0.0
        return float(values.mean())

    def _slot_on_point_values(
        self, sim, slots: tuple[int, int, int]
    ) -> np.ndarray | None:
        fake_values = getattr(sim, "on_point_by_slot", None)
        if fake_values is not None:
            arr = np.asarray(fake_values, dtype=np.float32)
            if arr.shape[0] < _cpp.AGENTS_PER_MATCH:
                return None
            return arr[list(slots)].astype(np.float32, copy=True)
        if self._obs_bufs is None or self._on_point_slice is None:
            return None
        out = np.zeros(3, dtype=np.float32)
        for i, slot in enumerate(slots):
            try:
                _cpp.build_actor_obs(sim, slot, self._obs_bufs[slot])
            except Exception:
                return None
            out[i] = float(self._obs_bufs[slot][self._on_point_slice][0])
        return out
