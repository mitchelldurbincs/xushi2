"""Deterministic Phase-9 self-play match-type scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SelfPlayMatch:
    match_type: str
    group: str
    snapshot_path: str | None = None
    anchor_bot: str | None = None


class SelfPlaySchedule:
    """Sample current/snapshot/anchor match types from a compact config."""

    _ORDER = ("current", "snapshot", "anchor")

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        latest: Sequence[str | Path] = (),
        historical: Sequence[str | Path] = (),
        anchor: Sequence[str | Path] = (),
        anchor_bot: str | None = "noop",
    ) -> None:
        raw_weights = dict(weights or {"current": 0.7, "snapshot": 0.2, "anchor": 0.1})
        cleaned_weights: dict[str, float] = {}
        for key in self._ORDER:
            weight = float(raw_weights.get(key, 0.0))
            if weight < 0.0:
                raise ValueError("self-play schedule weights must be non-negative")
            if weight > 0.0:
                cleaned_weights[key] = weight
        if not cleaned_weights:
            raise ValueError("self-play schedule must have at least one positive weight")
        self.weights = cleaned_weights
        self.latest = tuple(str(Path(p)) for p in latest)
        self.historical = tuple(str(Path(p)) for p in historical)
        self.anchor = tuple(str(Path(p)) for p in anchor)
        self.anchor_bot = None if anchor_bot is None else str(anchor_bot)
        self.summary = ",".join(
            f"{key}:{self.weights[key]:.3g}" for key in self._ORDER if key in self.weights
        )

    @classmethod
    def from_config(
        cls,
        schedule_cfg: dict | None,
        league_cfg: dict | None = None,
    ) -> "SelfPlaySchedule":
        schedule_cfg = dict(schedule_cfg or {})
        league_cfg = dict(league_cfg or {})
        return cls(
            weights=dict(schedule_cfg.get("weights", {})),
            latest=tuple(league_cfg.get("latest", ())),
            historical=tuple(league_cfg.get("historical", ())),
            anchor=tuple(league_cfg.get("anchor", ())),
            anchor_bot=schedule_cfg.get("anchor_bot", "noop"),
        )

    def sample(self, seed: int) -> SelfPlayMatch:
        rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
        kinds = [k for k in self._ORDER if k in self.weights]
        weights = np.asarray([self.weights[k] for k in kinds], dtype=np.float64)
        kind = str(rng.choice(kinds, p=weights / weights.sum()))
        if kind == "current":
            return SelfPlayMatch(match_type="current", group="current")
        if kind == "anchor":
            if self.anchor:
                path = self.anchor[int(rng.integers(0, len(self.anchor)))]
                return SelfPlayMatch(
                    match_type="anchor", group="anchor", snapshot_path=path
                )
            if self.anchor_bot is None:
                raise ValueError("anchor match sampled but no anchor paths or bot exist")
            return SelfPlayMatch(
                match_type="anchor", group="anchor", anchor_bot=self.anchor_bot
            )

        paths: list[tuple[str, str]] = []
        paths.extend(("latest", p) for p in self.latest)
        paths.extend(("historical", p) for p in self.historical)
        if not paths:
            raise ValueError("snapshot match sampled but no snapshot paths exist")
        group, path = paths[int(rng.integers(0, len(paths)))]
        return SelfPlayMatch(match_type="snapshot", group=group, snapshot_path=path)
