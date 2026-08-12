"""View helpers over the SimPool reward feature block.

``RewardFeatureView`` presents one env's feature vector (layout:
``obs_manifest.REWARD_FEATURE_FIELDS``, written by the C++
``write_reward_features``) through the exact interface ``RewardCalculator``
reads from a live ``Sim``:

- the properties ``EventDeltaExtractor.read`` consumes (`tick`,
  ``team_*_score_ticks``, ``team_*_kills``, ``*_by_slot`` arrays,
  ``episode_over``, ``winner``), and
- the fake-sim hooks the calculator already honors for tests
  (``on_point_by_slot``, ``objective_conversion_state``,
  ``dist_to_center_by_slot``).

Reusing the per-env ``RewardCalculator`` through this view — instead of a
re-derived batched reward implementation — keeps SimPool reward math
line-for-line identical to the legacy path; the only thing that changed is
where the numbers come from (one packed block per step instead of ~30 FFI
property reads plus rebuilt actor observations).
"""

from __future__ import annotations

import numpy as np

from . import xushi2_cpp as _cpp
from .obs_manifest import REWARD_FEATURE_DIM, reward_feature_slice

__all__ = ["REWARD_FEATURE_DIM", "RewardFeatureView"]

_TICK = reward_feature_slice("tick").start
_A_SCORE = reward_feature_slice("team_a_score_ticks").start
_B_SCORE = reward_feature_slice("team_b_score_ticks").start
_A_KILLS = reward_feature_slice("team_a_kills").start
_B_KILLS = reward_feature_slice("team_b_kills").start
_OWNER_SIGN = reward_feature_slice("owner_sign_a").start
_CAP_SIGN = reward_feature_slice("cap_sign_a").start
_CAP_FRACTION = reward_feature_slice("cap_progress_fraction").start
_EPISODE_OVER = reward_feature_slice("episode_over").start
_WINNER_SIGN = reward_feature_slice("winner_sign").start
_KILLS_SLOT = reward_feature_slice("kills_by_slot")
_DEATHS_SLOT = reward_feature_slice("deaths_by_slot")
_DAMAGE_SLOT = reward_feature_slice("damage_centi_by_slot")
_ON_POINT_SLOT = reward_feature_slice("on_point_by_slot")
_DIST_SLOT = reward_feature_slice("dist_to_center_by_slot")


class RewardFeatureView:
    """Sim-shaped adapter over one env's reward feature vector."""

    __slots__ = ("_f",)

    def __init__(self, features: np.ndarray) -> None:
        f = np.asarray(features, dtype=np.float32).reshape(-1)
        if f.shape[0] != REWARD_FEATURE_DIM:
            raise ValueError(
                f"features must have {REWARD_FEATURE_DIM} entries, got {f.shape}"
            )
        self._f = f

    def update(self, features: np.ndarray) -> None:
        """Point the view at a fresh feature vector (no copy)."""
        f = np.asarray(features, dtype=np.float32).reshape(-1)
        if f.shape[0] != REWARD_FEATURE_DIM:
            raise ValueError(
                f"features must have {REWARD_FEATURE_DIM} entries, got {f.shape}"
            )
        self._f = f

    # --- EventDeltaExtractor / RewardCalculator property surface ------------
    @property
    def tick(self) -> int:
        return int(self._f[_TICK])

    @property
    def team_a_score_ticks(self) -> int:
        return int(self._f[_A_SCORE])

    @property
    def team_b_score_ticks(self) -> int:
        return int(self._f[_B_SCORE])

    @property
    def team_a_kills(self) -> int:
        return int(self._f[_A_KILLS])

    @property
    def team_b_kills(self) -> int:
        return int(self._f[_B_KILLS])

    @property
    def kills_by_slot(self) -> np.ndarray:
        return self._f[_KILLS_SLOT].astype(np.int64)

    @property
    def deaths_by_slot(self) -> np.ndarray:
        return self._f[_DEATHS_SLOT].astype(np.int64)

    @property
    def damage_dealt_by_slot(self) -> np.ndarray:
        return self._f[_DAMAGE_SLOT].astype(np.int64)

    @property
    def episode_over(self) -> bool:
        return bool(self._f[_EPISODE_OVER] > 0.5)

    @property
    def winner(self):
        sign = float(self._f[_WINNER_SIGN])
        if sign > 0.5:
            return _cpp.Team.A
        if sign < -0.5:
            return _cpp.Team.B
        return _cpp.Team.Neutral

    # --- Fake-sim hooks RewardCalculator / ObsAccessor already honor --------
    @property
    def on_point_by_slot(self) -> np.ndarray:
        return self._f[_ON_POINT_SLOT]

    @property
    def dist_to_center_by_slot(self) -> np.ndarray:
        return self._f[_DIST_SLOT]

    @property
    def objective_conversion_state(self) -> tuple[float, float, float]:
        return (
            float(self._f[_OWNER_SIGN]),
            float(self._f[_CAP_SIGN]),
            float(self._f[_CAP_FRACTION]),
        )
