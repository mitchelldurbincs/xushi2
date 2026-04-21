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

from dataclasses import dataclass

__all__ = [
    "RewardCalculator",
    "TICK_HZ",
    "SHAPING_CLIP_DEFAULT",
    "TERMINAL_WIN_DEFAULT",
    "TERMINAL_LOSS_DEFAULT",
    "KILL_BONUS_DEFAULT",
    "DEATH_PENALTY_DEFAULT",
    "SCORE_PER_SECOND_DEFAULT",
]

from . import xushi2_cpp as _cpp

TICK_HZ: int = _cpp.TICK_HZ

# Defaults from rl_design.md §5.
SHAPING_CLIP_DEFAULT: float = 3.0
TERMINAL_WIN_DEFAULT: float = 10.0
TERMINAL_LOSS_DEFAULT: float = -10.0
KILL_BONUS_DEFAULT: float = 0.25
DEATH_PENALTY_DEFAULT: float = 0.25
# +0.01 per own objective score point; since score ticks accumulate at
# 1 per sim tick while controlled, this equals 0.01/second while scoring.
SCORE_PER_SECOND_DEFAULT: float = 0.01


@dataclass
class _EventCounters:
    a_score_ticks: int = 0
    b_score_ticks: int = 0
    a_kills: int = 0
    b_kills: int = 0


def _read_counters(sim) -> _EventCounters:
    return _EventCounters(
        a_score_ticks=int(sim.team_a_score_ticks),
        b_score_ticks=int(sim.team_b_score_ticks),
        a_kills=int(sim.team_a_kills),
        b_kills=int(sim.team_b_kills),
    )


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


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
    ) -> None:
        if shaping_clip <= 0.0:
            raise ValueError("shaping_clip must be > 0")
        self._shaping_clip = float(shaping_clip)
        self._terminal_win = float(terminal_win)
        self._terminal_loss = float(terminal_loss)
        self._kill_bonus = float(kill_bonus)
        self._death_penalty = float(death_penalty)
        self._score_per_second = float(score_per_second)
        self._prev = _EventCounters()
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

    # --- public API ---

    def reset(self, sim) -> None:
        """Capture initial counters and zero cumulative shaping totals."""
        self._prev = _read_counters(sim)
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

    def step(self, sim) -> tuple[float, float]:
        """Return (team_a_reward, team_b_reward) for the just-stepped sim.

        Caller must have stepped the sim before this call. Multiple
        calls without a step in between return zero.
        """
        now = _read_counters(sim)

        a_score_delta = now.a_score_ticks - self._prev.a_score_ticks
        b_score_delta = now.b_score_ticks - self._prev.b_score_ticks
        a_kills_delta = now.a_kills - self._prev.a_kills
        b_kills_delta = now.b_kills - self._prev.b_kills

        # Convert score ticks to seconds for reward math.
        a_score_seconds = a_score_delta / float(TICK_HZ)
        b_score_seconds = b_score_delta / float(TICK_HZ)

        # Symmetrized: team_reward = own_events - enemy_events.
        # Team A own_events: own_score_gain + own_kills (= b_kills_delta is
        # wrong; kills are accumulated against the scorer, so
        # sim.team_a_kills increments when A kills a B hero. Use that.)
        raw_a = (
            self._score_per_second * a_score_seconds
            - self._score_per_second * b_score_seconds
            + self._kill_bonus * a_kills_delta
            - self._death_penalty * b_kills_delta
        )
        raw_b = -raw_a  # zero-sum on raw shaping by symmetrization

        reward_a = self._apply_clip(raw_a, "a")
        reward_b = self._apply_clip(raw_b, "b")

        self._prev = now
        return reward_a, reward_b

    def add_terminal(self, sim) -> tuple[float, float]:
        """Return (a_terminal, b_terminal) for the just-finished episode.

        Call only after ``sim.episode_over`` is True. Winner is read from
        ``sim.winner`` (a C++ enum); Team.Neutral means a draw and returns
        0 for both teams. Terminal rewards are not clipped.
        """
        if not sim.episode_over:
            raise RuntimeError(
                "add_terminal called before episode_over; step until "
                "terminal before querying terminal rewards"
            )
        winner = sim.winner
        if winner == _cpp.Team.A:
            return self._terminal_win, self._terminal_loss
        if winner == _cpp.Team.B:
            return self._terminal_loss, self._terminal_win
        return 0.0, 0.0  # draw

    # --- introspection for tests ---

    @property
    def cumulative_shaped_a(self) -> float:
        return self._cum_shaped_a

    @property
    def cumulative_shaped_b(self) -> float:
        return self._cum_shaped_b

    # --- internal ---

    def _apply_clip(self, raw_delta: float, team: str) -> float:
        """Clip the cumulative running total to [-clip, +clip] and return
        the step reward consistent with that cap."""
        if team == "a":
            old = self._cum_shaped_a
            new = _clip(old + raw_delta, -self._shaping_clip, self._shaping_clip)
            self._cum_shaped_a = new
        else:
            old = self._cum_shaped_b
            new = _clip(old + raw_delta, -self._shaping_clip, self._shaping_clip)
            self._cum_shaped_b = new
        return new - old
