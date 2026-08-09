"""Phase-1 reward calculator."""

from __future__ import annotations

import numpy as np

__all__ = [
    "CAPTURE_COMPLETED_BONUS_DEFAULT",
    "CAP_PROGRESS_POTENTIAL_COEF_DEFAULT",
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
from .reward_components import (
    CumulativeClipper,
    EventCounters,
    EventDeltaExtractor,
    ObsAccessor,
    ShapingTerms,
    TICK_HZ,
)

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
CAP_PROGRESS_POTENTIAL_COEF_DEFAULT: float = 0.0
CAPTURE_COMPLETED_BONUS_DEFAULT: float = 0.0


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
        cap_progress_potential_coef: float = CAP_PROGRESS_POTENTIAL_COEF_DEFAULT,
        capture_completed_bonus: float = CAPTURE_COMPLETED_BONUS_DEFAULT,
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
        if damage_dealt_coef > 0.0 and not per_agent_rewards:
            # Damage shaping is attributed per slot and is only applied by the
            # per-agent step path. Silently accepting it on the scalar path
            # meant several committed self-play configs asked for damage
            # shaping and got none, with no metric revealing it.
            raise ValueError(
                "damage_dealt_coef > 0 requires per_agent_rewards=True; the "
                "scalar reward path has no per-slot damage attribution. Either "
                "enable per-agent rewards or remove damage_dealt_coef."
            )
        if majority_on_point_coef < 0.0:
            raise ValueError("majority_on_point_coef must be >= 0")
        if uncontested_on_point_coef < 0.0:
            raise ValueError("uncontested_on_point_coef must be >= 0")
        if cap_progress_potential_coef < 0.0:
            raise ValueError("cap_progress_potential_coef must be >= 0")
        if capture_completed_bonus < 0.0:
            raise ValueError("capture_completed_bonus must be >= 0")
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
        self._cap_progress_potential_coef = float(cap_progress_potential_coef)
        self._capture_completed_bonus = float(capture_completed_bonus)
        self._last_majority_on_point_metrics = self._empty_majority_on_point_metrics()
        self._last_uncontested_on_point_metrics = (
            self._empty_uncontested_on_point_metrics()
        )
        self._prev_conversion_phi_a = 0.0
        self._prev_owner_sign_a = 0.0
        self._captures_a = 0
        self._captures_b = 0
        self._last_objective_conversion_metrics = (
            self._empty_objective_conversion_metrics()
        )
        self._extractor = EventDeltaExtractor(per_agent=self._per_agent)
        self._prev = EventCounters()
        self._prev_tick = 0
        self._clipper = CumulativeClipper(shaping_clip)
        obs_enabled = (
            self._distance_shaping_coef > 0.0
            or self._on_point_shaping_coef > 0.0
            or self._majority_on_point_alpha > 0.0
            or self._uncontested_on_point_alpha > 0.0
            or self._cap_progress_potential_coef > 0.0
            or self._capture_completed_bonus > 0.0
            or self._per_agent
        )
        self._obs = ObsAccessor(enabled=obs_enabled)

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

    @staticmethod
    def _empty_uncontested_on_point_metrics() -> dict[str, float]:
        return {
            "uncontested_on_point_alpha": 0.0,
            "uncontested_on_point_count_a": 0.0,
            "uncontested_on_point_count_b": 0.0,
            "uncontested_on_point_reward_a": 0.0,
            "uncontested_on_point_reward_b": 0.0,
        }

    @staticmethod
    def _empty_objective_conversion_metrics() -> dict[str, float]:
        return {
            "cap_progress_potential_coef": 0.0,
            "capture_completed_bonus": 0.0,
            "conversion_phi_a": 0.0,
            "cap_progress_potential_reward_a": 0.0,
            "capture_completed_reward_a": 0.0,
            "captures_a": 0.0,
            "captures_b": 0.0,
        }

    def reset(self, sim) -> None:
        self._prev = self._extractor.read(sim)
        self._prev_tick = int(getattr(sim, "tick", 0))
        self._clipper.reset()
        self._last_majority_on_point_metrics = self._empty_majority_on_point_metrics()
        self._last_uncontested_on_point_metrics = (
            self._empty_uncontested_on_point_metrics()
        )
        self._captures_a = 0
        self._captures_b = 0
        self._last_objective_conversion_metrics = (
            self._empty_objective_conversion_metrics()
        )
        state = self._objective_conversion_state(sim)
        if state is None:
            self._prev_conversion_phi_a = 0.0
            self._prev_owner_sign_a = 0.0
        else:
            owner_sign, cap_sign, progress = state
            self._prev_conversion_phi_a = owner_sign + cap_sign * progress
            self._prev_owner_sign_a = owner_sign

    def set_team_spirit(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"team_spirit must be in [0, 1], got {value}")
        if value > 0.0 and not self._per_agent:
            # team_spirit blends per-agent rewards toward the team mean, which
            # only the per-agent step path computes. Accepting a non-zero value
            # on the scalar path would silently discard a configured ramp.
            # Zero is still accepted: the trainer pushes it every update
            # regardless of whether a ramp is configured.
            raise ValueError(
                "team_spirit > 0 requires per_agent_rewards=True; the scalar "
                "reward path has no per-agent rewards to blend."
            )
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

    def majority_on_point_metrics(self) -> dict[str, float]:
        return dict(self._last_majority_on_point_metrics)

    def uncontested_on_point_metrics(self) -> dict[str, float]:
        return dict(self._last_uncontested_on_point_metrics)

    def objective_conversion_metrics(self) -> dict[str, float]:
        return dict(self._last_objective_conversion_metrics)

    def _objective_conversion_state(self, sim) -> tuple[float, float, float] | None:
        """Return (owner_sign_a, cap_sign_a, cap_progress) or None.

        Tests may fake this with a sim attribute ``objective_conversion_state``
        holding the same tuple; otherwise the values are read from Team A's
        actor observation (global objective fields).
        """
        if (
            self._cap_progress_potential_coef <= 0.0
            and self._capture_completed_bonus <= 0.0
        ):
            return None
        fake = getattr(sim, "objective_conversion_state", None)
        if fake is not None:
            owner_sign, cap_sign, progress = fake
            return float(owner_sign), float(cap_sign), float(progress)
        return self._obs.objective_conversion_state(sim)

    def _objective_conversion_term(self, sim) -> float:
        """Team-A-signed conversion shaping for this step (B receives the negation).

        Two parts:
        - Potential-based shaping on the objective state machine with
          Phi_A = owner_sign + cap_sign * cap_progress_fraction. The step
          reward is coef * (Phi' - Phi), so accruing capture progress and
          gaining ownership pay incrementally, while letting progress decay
          or losing ownership costs the same amount back. Potential-based
          terms do not change the optimal policy, so this can stay enabled
          permanently (no anneal needed).
        - A one-time event bonus when objective ownership flips to a team
          (the capture-completion event the score chain depends on).
        """
        state = self._objective_conversion_state(sim)
        if state is None:
            metrics = self._empty_objective_conversion_metrics()
            metrics["cap_progress_potential_coef"] = self._cap_progress_potential_coef
            metrics["capture_completed_bonus"] = self._capture_completed_bonus
            self._last_objective_conversion_metrics = metrics
            return 0.0
        owner_sign, cap_sign, progress = state
        phi_a = owner_sign + cap_sign * progress
        pbrs_a = self._cap_progress_potential_coef * (
            phi_a - self._prev_conversion_phi_a
        )
        bonus_a = 0.0
        if owner_sign > 0.5 and self._prev_owner_sign_a <= 0.5:
            self._captures_a += 1
            bonus_a += self._capture_completed_bonus
        elif owner_sign < -0.5 and self._prev_owner_sign_a >= -0.5:
            self._captures_b += 1
            bonus_a -= self._capture_completed_bonus
        self._prev_conversion_phi_a = phi_a
        self._prev_owner_sign_a = owner_sign
        self._last_objective_conversion_metrics = {
            "cap_progress_potential_coef": self._cap_progress_potential_coef,
            "capture_completed_bonus": self._capture_completed_bonus,
            "conversion_phi_a": phi_a,
            "cap_progress_potential_reward_a": pbrs_a,
            "capture_completed_reward_a": bonus_a,
            "captures_a": float(self._captures_a),
            "captures_b": float(self._captures_b),
        }
        return pbrs_a + bonus_a

    def _decision_seconds(self, sim) -> float:
        now_tick = int(getattr(sim, "tick", self._prev_tick))
        delta_ticks = now_tick - self._prev_tick
        self._prev_tick = now_tick
        if delta_ticks <= 0:
            delta_ticks = 1
        return float(delta_ticks) / float(TICK_HZ)

    @staticmethod
    def _scale_raw_terms(raw_a: float, raw_b: float, coef: float, decision_seconds: float) -> tuple[float, float]:
        if coef <= 0.0 or decision_seconds <= 0.0:
            return raw_a, raw_b
        shaped = coef * decision_seconds
        return raw_a * shaped, raw_b * shaped

    def step(self, sim):
        return self._step_per_agent(sim) if self._per_agent else self._step_scalar(sim)

    def _step_scalar(self, sim) -> tuple[float, float]:
        now = self._extractor.read(sim)
        a_score_seconds, b_score_seconds, a_kills_delta, b_kills_delta = self._extractor.scalar_delta(now, self._prev)
        decision_seconds = self._decision_seconds(sim)

        # Each team's score/kill/death term is computed from ITS OWN counters.
        # Mirroring Team A's term (raw_b = -raw_a) is only equivalent when
        # kill_bonus == death_penalty; otherwise Team B ends up being paid
        # death_penalty per kill and charged kill_bonus per death, i.e. its
        # opponent's coefficients. That matters most under self-play, where a
        # single shared policy sees both teams' rewards.
        raw_a = ShapingTerms.score_kill_death_scalar(
            a_score_seconds,
            b_score_seconds,
            a_kills_delta,
            b_kills_delta,
            score_per_second=self._score_per_second,
            kill_bonus=self._kill_bonus,
            death_penalty=self._death_penalty,
        )
        raw_b = ShapingTerms.score_kill_death_scalar(
            b_score_seconds,
            a_score_seconds,
            b_kills_delta,
            a_kills_delta,
            score_per_second=self._score_per_second,
            kill_bonus=self._kill_bonus,
            death_penalty=self._death_penalty,
        )

        # Positional shaping is genuinely zero-sum: both terms are already
        # computed as an A-minus-B difference, so B takes the negation.
        zero_sum_a = self._obs.distance_term(sim, self._distance_shaping_coef)
        zero_sum_a += self._obs.on_point_term(sim, self._on_point_shaping_coef)
        raw_a += zero_sum_a
        raw_b -= zero_sum_a

        majority_a, majority_b = self._majority_on_point_by_team(
            sim, decision_seconds
        )
        raw_a += majority_a - majority_b
        raw_b -= majority_a - majority_b

        uncontested_a, uncontested_b = self._uncontested_on_point_by_team(
            sim, decision_seconds
        )
        raw_a += uncontested_a - uncontested_b
        raw_b -= uncontested_a - uncontested_b

        conversion_a = self._objective_conversion_term(sim)
        raw_a += conversion_a
        raw_b -= conversion_a

        tp = ShapingTerms.time_penalty_per_tick(self._time_penalty_per_second)
        raw_a += tp
        raw_b += tp

        reward_a = self._clipper.apply_clip(raw_a, "a")
        reward_b = self._clipper.apply_clip(raw_b, "b")
        self._prev = now
        return reward_a, reward_b

    def _step_per_agent(self, sim) -> tuple[np.ndarray, np.ndarray]:
        now = self._extractor.read(sim)
        decision_seconds = self._decision_seconds(sim)
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
        dmg_a, dmg_b = ShapingTerms.damage_by_slot(
            now, self._prev, self._damage_dealt_coef
        )
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
            sim, decision_seconds
        )
        if majority_a != 0.0:
            raw_a += majority_a * self._majority_on_point_shares(sim, (0, 1, 2))
            raw_b -= majority_a / 3.0
        if majority_b != 0.0:
            raw_b += majority_b * self._majority_on_point_shares(sim, (3, 4, 5))
            raw_a -= majority_b / 3.0

        uncontested_a, uncontested_b = self._uncontested_on_point_by_team(
            sim, decision_seconds
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

        conversion_a = self._objective_conversion_term(sim)
        if conversion_a > 0.0:
            # Team A gained potential/ownership: credit A's on-point members.
            raw_a += conversion_a * self._on_point_shares(sim, (0, 1, 2))
            raw_b -= conversion_a / 3.0
        elif conversion_a < 0.0:
            # Team B gained (or A's progress decayed): credit B's on-point
            # members and charge A uniformly.
            raw_b += (-conversion_a) * self._on_point_shares(sim, (3, 4, 5))
            raw_a += conversion_a / 3.0

        raw_a += self._obs.distance_term(sim, self._distance_shaping_coef)
        raw_b -= self._obs.distance_term(sim, self._distance_shaping_coef)
        raw_a += self._obs.on_point_term(sim, self._on_point_shaping_coef)
        raw_b -= self._obs.on_point_term(sim, self._on_point_shaping_coef)

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
            metrics["uncontested_on_point_alpha"] = float(self._uncontested_on_point_alpha)
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

    def _uncontested_on_point_shares(self, sim, slots: tuple[int, int, int]) -> np.ndarray:
        if self._uncontested_on_point_distribute == "uniform":
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return self._on_point_shares(sim, slots)

    def _on_point_shares(self, sim, slots: tuple[int, int, int]) -> np.ndarray:
        values = self._slot_on_point_values(sim, slots)
        if values is None:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        total = float(values.sum())
        if total <= 1e-12:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        return values / total

    def _slot_on_point_values(
        self, sim, slots: tuple[int, int, int]
    ) -> np.ndarray | None:
        fake_values = getattr(sim, "on_point_by_slot", None)
        if fake_values is not None:
            arr = np.asarray(fake_values, dtype=np.float32)
            if arr.shape[0] < _cpp.AGENTS_PER_MATCH:
                return None
            return arr[list(slots)].astype(np.float32, copy=True)
        if self._obs.on_point_slice is None or self._obs.obs_bufs is None:
            return None
        out = np.zeros(3, dtype=np.float32)
        for i, slot in enumerate(slots):
            # Deliberately unguarded. This used to catch Exception and return
            # None, which made _on_point_shares fall back to a uniform 1/3 --
            # silently replacing per-agent credit assignment with an even team
            # split, which is exactly what the team_spirit work exists to
            # study. build_actor_obs raises only ValueError, for a malformed
            # buffer; every other failure inside the builder aborts and cannot
            # be caught anyway. So the catch could not catch what it was
            # written for, and could catch real bugs.
            _cpp.build_actor_obs(sim, slot, self._obs.obs_bufs[slot])
            out[i] = float(self._obs.obs_bufs[slot][self._obs.on_point_slice][0])
        return out

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
