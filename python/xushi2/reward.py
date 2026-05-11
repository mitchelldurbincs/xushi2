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
    "RewardCalculator",
    "TICK_HZ",
    "SHAPING_CLIP_DEFAULT",
    "TERMINAL_WIN_DEFAULT",
    "TERMINAL_LOSS_DEFAULT",
    "KILL_BONUS_DEFAULT",
    "DEATH_PENALTY_DEFAULT",
    "SCORE_PER_SECOND_DEFAULT",
    "DISTANCE_SHAPING_COEF_DEFAULT",
    "ON_POINT_SHAPING_COEF_DEFAULT",
    "TIME_PENALTY_PER_SECOND_DEFAULT",
    "DAMAGE_DEALT_COEF_DEFAULT",
]

from . import xushi2_cpp as _cpp
from .obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

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
# Opt-in per-decision distance-to-objective shaping. 0.0 disables; a small
# positive value (~0.01) provides a dense gradient toward the cap for probes
# where random exploration struggles to discover "sit on point". Zero-sum
# symmetrized: team A's per-step term is -coef*(dist_A - dist_B), team B is
# the negation. Not yet in rl_design.md §5 — probe/training-only augmentation.
DISTANCE_SHAPING_COEF_DEFAULT: float = 0.0
# Opt-in per-decision objective-contact shaping. Probe-only curriculum helper:
# rewards a team by coef * fraction_of_team_slots_on_point, symmetrized against
# the opposing team. Intended to bridge "walk near point" to "enter/hold point".
ON_POINT_SHAPING_COEF_DEFAULT: float = 0.0
# Per-second penalty applied to BOTH teams every tick. Intentionally
# breaks zero-sum to remove the deny-stalemate basin: with tps=0, a 0/0
# draw nets ~0 and PPO finds it as stable as scoring; with tps>0, a 30s
# draw nets -30*tps for both teams, so denying is strictly worse than
# attempting to score. 0.0 disables (default; backwards compatible).
TIME_PENALTY_PER_SECOND_DEFAULT: float = 0.0
# Per-HP-of-damage-dealt reward, credited to the attacker's slot. Used in
# the per-agent path only; gives PPO a per-slot signal for "your shot
# connected" so aim_delta gets a meaningful gradient. Damage is counted in
# centi-HP internally; the coef here is reward per HP (we divide by 100).
# 0.0 disables (default; backwards compatible).
DAMAGE_DEALT_COEF_DEFAULT: float = 0.0
_CENTI_HP_PER_HP: float = 100.0

_TEAM_A_RANGER_SLOT: int = 0
_TEAM_B_RANGER_SLOT: int = 3


@dataclass
class _EventCounters:
    a_score_ticks: int = 0
    b_score_ticks: int = 0
    a_kills: int = 0
    b_kills: int = 0
    # Per-slot lifetime kills/deaths. Length-6 arrays indexed by absolute
    # slot (0..2 = team A, 3..5 = team B). Populated only when the calculator
    # is in per_agent_rewards mode; on the scalar path these stay zero.
    kills_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )
    deaths_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )
    # Cumulative damage applied by each slot, in centi-HP. Read only on
    # the per-agent path when damage_dealt_coef > 0.
    damage_dealt_by_slot: np.ndarray = field(
        default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64)
    )


def _read_counters(sim, *, per_agent: bool = False) -> _EventCounters:
    out = _EventCounters(
        a_score_ticks=int(sim.team_a_score_ticks),
        b_score_ticks=int(sim.team_b_score_ticks),
        a_kills=int(sim.team_a_kills),
        b_kills=int(sim.team_b_kills),
    )
    if per_agent:
        out.kills_by_slot = np.asarray(sim.kills_by_slot, dtype=np.int64)
        out.deaths_by_slot = np.asarray(sim.deaths_by_slot, dtype=np.int64)
        # damage_dealt_by_slot may be absent on test fakes that predate the
        # field — fall back to zeros so existing tests don't break.
        damage_attr = getattr(sim, "damage_dealt_by_slot", None)
        if damage_attr is not None:
            out.damage_dealt_by_slot = np.asarray(damage_attr, dtype=np.int64)
    return out


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
        distance_shaping_coef: float = DISTANCE_SHAPING_COEF_DEFAULT,
        on_point_shaping_coef: float = ON_POINT_SHAPING_COEF_DEFAULT,
        time_penalty_per_second: float = TIME_PENALTY_PER_SECOND_DEFAULT,
        per_agent_rewards: bool = False,
        team_spirit: float = 0.0,
        damage_dealt_coef: float = DAMAGE_DEALT_COEF_DEFAULT,
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
        self._shaping_clip = float(shaping_clip)
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
        self._prev = _EventCounters()
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

        # Preallocate obs buffers when obs-derived shaping is on OR when the
        # per-agent path is on (the per-agent score split needs per-slot
        # on-point fractions). The obs builders are the only path to hero
        # positions / objective contact from Python.
        needs_obs_bufs = (
            self._distance_shaping_coef > 0.0
            or self._on_point_shaping_coef > 0.0
            or self._per_agent
        )
        if needs_obs_bufs:
            self._pos_slice = actor_field_slice("own_position")
            self._on_point_slice = actor_field_slice("self_on_point")
            self._obs_bufs = [
                np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32) for _ in range(_cpp.AGENTS_PER_MATCH)
            ]
            self._obs_buf_a = self._obs_bufs[_TEAM_A_RANGER_SLOT]
            self._obs_buf_b = self._obs_bufs[_TEAM_B_RANGER_SLOT]
        else:
            self._pos_slice = None
            self._on_point_slice = None
            self._obs_bufs = None
            self._obs_buf_a = None
            self._obs_buf_b = None

    # --- public API ---

    def reset(self, sim) -> None:
        """Capture initial counters and zero cumulative shaping totals."""
        self._prev = _read_counters(sim, per_agent=self._per_agent)
        self._cum_shaped_a = 0.0
        self._cum_shaped_b = 0.0

    def set_team_spirit(self, value: float) -> None:
        """Update the team_spirit interpolation factor.

        Trainer calls this once per update with the ramp value computed
        from training progress. Only meaningful on the per-agent path;
        a no-op otherwise."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {value}")
        self._team_spirit = float(value)

    def step(self, sim):
        """Return per-team rewards for the just-stepped sim.

        Caller must have stepped the sim before this call. Multiple
        calls without a step in between return zero.

        Returns:
            ``(float, float)`` when ``per_agent_rewards=False`` (default).
            ``(np.ndarray(3,), np.ndarray(3,))`` when
            ``per_agent_rewards=True`` — kill bonus credited to the
            killer's slot, death penalty to the victim's slot, score
            split among on-point teammates, enemy score subtracted
            uniformly across own slots. Sum-invariant with the scalar
            path when ``kill_bonus == death_penalty``.
        """
        if self._per_agent:
            return self._step_per_agent(sim)
        return self._step_scalar(sim)

    def _step_scalar(self, sim) -> tuple[float, float]:
        now = _read_counters(sim, per_agent=False)

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

        # Optional per-decision distance-to-cap shaping (opt-in).
        # Zero-sum symmetrized so raw_b = -raw_a still holds.
        if self._distance_shaping_coef > 0.0:
            _cpp.build_actor_obs(sim, _TEAM_A_RANGER_SLOT, self._obs_buf_a)
            _cpp.build_actor_obs(sim, _TEAM_B_RANGER_SLOT, self._obs_buf_b)
            pos_a = self._obs_buf_a[self._pos_slice]
            pos_b = self._obs_buf_b[self._pos_slice]
            dist_a = float(np.hypot(pos_a[0], pos_a[1]))
            dist_b = float(np.hypot(pos_b[0], pos_b[1]))
            raw_a += -self._distance_shaping_coef * (dist_a - dist_b)

        if self._on_point_shaping_coef > 0.0:
            on_a = self._team_on_point_fraction(sim, (0, 1, 2))
            on_b = self._team_on_point_fraction(sim, (3, 4, 5))
            raw_a += self._on_point_shaping_coef * (on_a - on_b)

        raw_b = -raw_a  # zero-sum on raw shaping by symmetrization

        # Asymmetric time penalty applied to BOTH teams. Not zero-sum on
        # purpose — see TIME_PENALTY_PER_SECOND_DEFAULT docstring for why.
        if self._time_penalty_per_second != 0.0:
            tp_step = -self._time_penalty_per_second / float(TICK_HZ)
            raw_a += tp_step
            raw_b += tp_step

        reward_a = self._apply_clip(raw_a, "a")
        reward_b = self._apply_clip(raw_b, "b")

        self._prev = now
        return reward_a, reward_b

    def _step_per_agent(self, sim) -> tuple[np.ndarray, np.ndarray]:
        now = _read_counters(sim, per_agent=True)

        a_score_seconds = (now.a_score_ticks - self._prev.a_score_ticks) / float(TICK_HZ)
        b_score_seconds = (now.b_score_ticks - self._prev.b_score_ticks) / float(TICK_HZ)
        kills_delta_slot = (now.kills_by_slot - self._prev.kills_by_slot).astype(np.float32)
        deaths_delta_slot = (now.deaths_by_slot - self._prev.deaths_by_slot).astype(np.float32)

        raw_a = np.zeros(3, dtype=np.float32)
        raw_b = np.zeros(3, dtype=np.float32)

        # Own kill credit and own death penalty, slot-attributed.
        raw_a += self._kill_bonus * kills_delta_slot[0:3]
        raw_b += self._kill_bonus * kills_delta_slot[3:6]
        raw_a -= self._death_penalty * deaths_delta_slot[0:3]
        raw_b -= self._death_penalty * deaths_delta_slot[3:6]

        # Damage-dealt credit (opt-in via damage_dealt_coef). Each slot
        # gets reward proportional to the HP they actually applied this
        # step. Asymmetric (no enemy-side mirror) — this is a per-team
        # shaping signal, not a zero-sum event credit. The team-sum clip
        # downstream still bounds total cumulative shaping at the
        # ±shaping_clip cap, so unbounded farming isn't possible.
        if self._damage_dealt_coef > 0.0:
            damage_delta_slot = (now.damage_dealt_by_slot - self._prev.damage_dealt_by_slot).astype(
                np.float32
            )
            per_hp = self._damage_dealt_coef / _CENTI_HP_PER_HP
            raw_a += per_hp * damage_delta_slot[0:3]
            raw_b += per_hp * damage_delta_slot[3:6]

        # Score: own split by per-slot on-point share; enemy subtracted
        # uniformly across own team's 3 slots.
        if a_score_seconds != 0.0:
            shares_a = self._on_point_shares(sim, (0, 1, 2))
            raw_a += self._score_per_second * a_score_seconds * shares_a
            raw_b -= (self._score_per_second * a_score_seconds) / 3.0
        if b_score_seconds != 0.0:
            shares_b = self._on_point_shares(sim, (3, 4, 5))
            raw_b += self._score_per_second * b_score_seconds * shares_b
            raw_a -= (self._score_per_second * b_score_seconds) / 3.0

        # Distance / on-point shaping: per-team scalar broadcast uniformly
        # across the 3 slots (no per-agent decomposition for these probes).
        if self._distance_shaping_coef > 0.0:
            _cpp.build_actor_obs(sim, _TEAM_A_RANGER_SLOT, self._obs_buf_a)
            _cpp.build_actor_obs(sim, _TEAM_B_RANGER_SLOT, self._obs_buf_b)
            pos_a = self._obs_buf_a[self._pos_slice]
            pos_b = self._obs_buf_b[self._pos_slice]
            dist_a = float(np.hypot(pos_a[0], pos_a[1]))
            dist_b = float(np.hypot(pos_b[0], pos_b[1]))
            shaping = -self._distance_shaping_coef * (dist_a - dist_b)
            raw_a += shaping
            raw_b -= shaping

        if self._on_point_shaping_coef > 0.0:
            on_a = self._team_on_point_fraction(sim, (0, 1, 2))
            on_b = self._team_on_point_fraction(sim, (3, 4, 5))
            shaping = self._on_point_shaping_coef * (on_a - on_b)
            raw_a += shaping
            raw_b -= shaping

        # Time penalty: uniform across all 6 slots, breaks zero-sum on purpose.
        if self._time_penalty_per_second != 0.0:
            tp_step = -self._time_penalty_per_second / float(TICK_HZ)
            raw_a += tp_step
            raw_b += tp_step

        # team_spirit interpolation: r_i ← (1-τ)·r_i + τ·mean(r_team).
        # Sum-invariant (mean is fixed under this affine combination),
        # so the team-sum clip below sees the same value before and after.
        if self._team_spirit > 0.0:
            tau = self._team_spirit
            mean_a = float(raw_a.mean())
            raw_a = (1.0 - tau) * raw_a + tau * mean_a
            mean_b = float(raw_b.mean())
            raw_b = (1.0 - tau) * raw_b + tau * mean_b

        # Cumulative clip on team sum (preserves today's ±shaping_clip
        # invariant; when kill_bonus == death_penalty, raw_a.sum() equals
        # today's scalar step reward exactly).
        self._scale_to_clipped_sum(raw_a, "a")
        self._scale_to_clipped_sum(raw_b, "b")

        self._prev = now
        return raw_a, raw_b

    def add_terminal(self, sim):
        """Return per-team terminal rewards for the just-finished episode.

        Call only after ``sim.episode_over`` is True. Winner is read from
        ``sim.winner`` (a C++ enum); Team.Neutral means a draw. Terminal
        rewards are not clipped.

        Returns ``(float, float)`` in scalar mode and uniform shape-(3,)
        ``np.float32`` arrays in ``per_agent_rewards`` mode (terminal is
        a team outcome by definition; team_spirit does not apply).
        """
        if not sim.episode_over:
            raise RuntimeError(
                "add_terminal called before episode_over; step until "
                "terminal before querying terminal rewards"
            )
        winner = sim.winner
        if winner == _cpp.Team.A:
            ta, tb = self._terminal_win, self._terminal_loss
        elif winner == _cpp.Team.B:
            ta, tb = self._terminal_loss, self._terminal_win
        else:
            ta, tb = 0.0, 0.0  # draw
        if self._per_agent:
            return (
                np.full(3, ta, dtype=np.float32),
                np.full(3, tb, dtype=np.float32),
            )
        return ta, tb

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

    def _scale_to_clipped_sum(self, raw: np.ndarray, team: str) -> None:
        """Clip the per-team cumulative sum, scaling raw in-place so its
        new sum matches the clipped step reward. Preserves today's
        ±shaping_clip cumulative cap on the team-sum metric."""
        team_step = float(raw.sum())
        clipped_step = self._apply_clip(team_step, team)
        if abs(team_step) > 1e-12:
            if clipped_step != team_step:
                raw *= clipped_step / team_step
            return
        # team_step ~ 0; clip can't shift a zero step (cumulative was already
        # within bounds), so raw is already correct.

    def _on_point_shares(self, sim, slots: tuple[int, int, int]) -> np.ndarray:
        """Return per-slot share (length 3) of the team's on-point presence.

        Sums to 1.0 when at least one slot is on point; otherwise returns
        uniform 1/3 each. Falls back to uniform when obs buffers are not
        allocated (e.g. unit-test FakeSim) or when build_actor_obs fails
        for a slot."""
        uniform = np.full(3, 1.0 / 3.0, dtype=np.float32)
        if self._obs_bufs is None or self._on_point_slice is None:
            return uniform
        shares = np.zeros(3, dtype=np.float32)
        for i, slot in enumerate(slots):
            try:
                _cpp.build_actor_obs(sim, slot, self._obs_bufs[slot])
            except Exception:
                return uniform
            shares[i] = float(self._obs_bufs[slot][self._on_point_slice][0])
        total = float(shares.sum())
        if total <= 1e-12:
            return uniform
        return shares / total

    def _team_on_point_fraction(self, sim, slots: tuple[int, int, int]) -> float:
        assert self._obs_bufs is not None
        assert self._on_point_slice is not None
        present = 0
        on_point = 0.0
        for slot in slots:
            try:
                _cpp.build_actor_obs(sim, slot, self._obs_bufs[slot])
            except Exception:
                continue
            present += 1
            on_point += float(self._obs_bufs[slot][self._on_point_slice][0])
        if present == 0:
            return 0.0
        return on_point / float(present)
