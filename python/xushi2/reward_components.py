from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import xushi2_cpp as _cpp
from .obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

TICK_HZ: int = _cpp.TICK_HZ
_CENTI_HP_PER_HP: float = 100.0
_TEAM_A_RANGER_SLOT: int = 0
_TEAM_B_RANGER_SLOT: int = 3


@dataclass
class EventCounters:
    a_score_ticks: int = 0
    b_score_ticks: int = 0
    a_kills: int = 0
    b_kills: int = 0
    kills_by_slot: np.ndarray = field(default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64))
    deaths_by_slot: np.ndarray = field(default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64))
    damage_dealt_by_slot: np.ndarray = field(default_factory=lambda: np.zeros(_cpp.AGENTS_PER_MATCH, dtype=np.int64))


class EventDeltaExtractor:
    def __init__(self, *, per_agent: bool) -> None:
        self._per_agent = bool(per_agent)

    def read(self, sim) -> EventCounters:
        out = EventCounters(
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
    def scalar_delta(now: EventCounters, prev: EventCounters) -> tuple[float, float, int, int]:
        return (
            (now.a_score_ticks - prev.a_score_ticks) / float(TICK_HZ),
            (now.b_score_ticks - prev.b_score_ticks) / float(TICK_HZ),
            now.a_kills - prev.a_kills,
            now.b_kills - prev.b_kills,
        )


class ObsAccessor:
    def __init__(self, *, enabled: bool) -> None:
        if enabled:
            self.pos_slice = actor_field_slice("own_position")
            self.on_point_slice = actor_field_slice("self_on_point")
            self.owner_slice = actor_field_slice("objective_owner_onehot")
            self.cap_team_slice = actor_field_slice("cap_team_onehot")
            self.cap_progress_slice = actor_field_slice("cap_progress")
            self.obs_bufs = [np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32) for _ in range(_cpp.AGENTS_PER_MATCH)]
            self.obs_buf_a = self.obs_bufs[_TEAM_A_RANGER_SLOT]
            self.obs_buf_b = self.obs_bufs[_TEAM_B_RANGER_SLOT]
        else:
            self.pos_slice = self.on_point_slice = self.obs_bufs = self.obs_buf_a = self.obs_buf_b = None
            self.owner_slice = self.cap_team_slice = self.cap_progress_slice = None

    def objective_conversion_state(self, sim) -> tuple[float, float, float] | None:
        """Read (owner_sign_a, cap_sign_a, cap_progress) from Team A's POV.

        Signs are +1 when the field favors Team A ("Us" from slot 0's view),
        -1 for Team B, 0 for Neutral. cap_progress is the [0, 1] fraction of
        objective_capture_ticks. The fields are global objective state, so any
        slot's obs carries them; slot 0 is used so "Us" means Team A.
        """
        if self.obs_buf_a is None or self.owner_slice is None:
            return None
        _cpp.build_actor_obs(sim, _TEAM_A_RANGER_SLOT, self.obs_buf_a)
        owner = self.obs_buf_a[self.owner_slice]
        cap_team = self.obs_buf_a[self.cap_team_slice]
        owner_sign = float(owner[1]) - float(owner[2])
        cap_sign = float(cap_team[1]) - float(cap_team[2])
        progress = float(self.obs_buf_a[self.cap_progress_slice][0])
        return owner_sign, cap_sign, progress

    def distance_term(self, sim, coef: float) -> float:
        if coef <= 0.0:
            return 0.0
        # Fake-sim hook (same pattern as on_point_by_slot /
        # objective_conversion_state): the SimPool feature view supplies the
        # per-slot team-frame center distances directly, so no actor obs
        # need rebuilding here.
        fake = getattr(sim, "dist_to_center_by_slot", None)
        if fake is not None:
            dist = np.asarray(fake, dtype=np.float32)
            return -coef * (
                float(dist[_TEAM_A_RANGER_SLOT]) - float(dist[_TEAM_B_RANGER_SLOT])
            )
        if self.obs_buf_a is None or self.obs_buf_b is None or self.pos_slice is None:
            return 0.0
        _cpp.build_actor_obs(sim, _TEAM_A_RANGER_SLOT, self.obs_buf_a)
        _cpp.build_actor_obs(sim, _TEAM_B_RANGER_SLOT, self.obs_buf_b)
        pos_a = self.obs_buf_a[self.pos_slice]
        pos_b = self.obs_buf_b[self.pos_slice]
        return -coef * (float(np.hypot(pos_a[0], pos_a[1])) - float(np.hypot(pos_b[0], pos_b[1])))

    def on_point_term(self, sim, coef: float) -> float:
        if coef <= 0.0:
            return 0.0
        return coef * (self.team_on_point_fraction(sim, (0, 1, 2)) - self.team_on_point_fraction(sim, (3, 4, 5)))

    def team_on_point_fraction(self, sim, slots: tuple[int, int, int]) -> float:
        # Fake-sim hook, mirroring RewardCalculator._slot_on_point_values.
        fake = getattr(sim, "on_point_by_slot", None)
        if fake is not None:
            values = np.asarray(fake, dtype=np.float32)
            return float(values[list(slots)].sum()) / float(len(slots))
        if self.obs_bufs is None or self.on_point_slice is None:
            return 0.0
        on_point = 0.0
        for slot in slots:
            _cpp.build_actor_obs(sim, slot, self.obs_bufs[slot])
            on_point += float(self.obs_bufs[slot][self.on_point_slice][0])
        return on_point / float(len(slots))


class ShapingTerms:
    @staticmethod
    def score_kill_death_scalar(a_score_seconds: float, b_score_seconds: float, a_kills_delta: int, b_kills_delta: int, *, score_per_second: float, kill_bonus: float, death_penalty: float) -> float:
        return score_per_second * a_score_seconds - score_per_second * b_score_seconds + kill_bonus * a_kills_delta - death_penalty * b_kills_delta

    @staticmethod
    def time_penalty_per_tick(tps: float) -> float:
        return 0.0 if tps == 0.0 else (-tps / float(TICK_HZ))

    @staticmethod
    def damage_by_slot(now: EventCounters, prev: EventCounters, coef: float) -> tuple[np.ndarray, np.ndarray]:
        if coef <= 0.0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        delta = (now.damage_dealt_by_slot - prev.damage_dealt_by_slot).astype(np.float32)
        per_hp = coef / _CENTI_HP_PER_HP
        return per_hp * delta[0:3], per_hp * delta[3:6]


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class CumulativeClipper:
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
