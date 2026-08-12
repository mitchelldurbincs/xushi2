"""Snapshot retention manifest helpers for Phase 9 self-play."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class SnapshotRetention:
    """Maintain a compact snapshot-league manifest for Phase-9 self-play."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        max_latest: int = 20,
        preserve_best: int = 3,
        anchor_paths: Sequence[str | Path] = (),
        weights: dict[str, float] | None = None,
    ) -> None:
        if max_latest <= 0:
            raise ValueError("max_latest must be positive")
        if preserve_best < 0:
            raise ValueError("preserve_best must be non-negative")
        self.manifest_path = Path(manifest_path)
        self.max_latest = int(max_latest)
        self.preserve_best = int(preserve_best)
        self.anchor_paths = tuple(str(Path(p)) for p in anchor_paths)
        self.weights = dict(weights or {"latest": 0.7, "historical": 0.2, "anchor": 0.1})
        # Load any existing history. Starting empty meant the first
        # record_checkpoint of a resumed run overwrote the manifest with a
        # single record, discarding the prior run's whole league.
        self._records: list[dict[str, Any]] = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"snapshot manifest {self.manifest_path} exists but could not be read: {exc}. "
                "Refusing to start from an empty league and silently discard it; move or "
                "delete the file to start fresh."
            ) from exc
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            raise ValueError(
                f"snapshot manifest {self.manifest_path} has a non-list 'records' entry"
            )
        return [dict(record) for record in records]

    def record_checkpoint(
        self,
        path: str | Path,
        *,
        update: int,
        score: float,
        matrix_score: float | None = None,
        matrix_gate_passed: bool | None = None,
        matrix_rows: int | None = None,
    ) -> dict:
        resolved = str(Path(path))
        self._records = [r for r in self._records if str(r["path"]) != resolved]
        record: dict[str, Any] = {
            "path": resolved,
            "update": int(update),
            "score": float(score),
        }
        if matrix_score is not None:
            record["matrix_score"] = float(matrix_score)
        if matrix_gate_passed is not None:
            record["matrix_gate_passed"] = bool(matrix_gate_passed)
        if matrix_rows is not None:
            record["matrix_rows"] = int(matrix_rows)
        self._records.append(record)
        self.write()
        return self.manifest()

    def manifest(self) -> dict:
        by_update = sorted(self._records, key=lambda r: int(r["update"]))
        latest = by_update[-self.max_latest :]
        best = sorted(
            self._records,
            key=lambda r: (
                int(bool(r.get("matrix_gate_passed", False))),
                int("matrix_score" in r),
                float(r.get("matrix_score", r["score"])),
                float(r["score"]),
                int(r["update"]),
            ),
            reverse=True,
        )[: self.preserve_best]
        return {
            "latest": [str(r["path"]) for r in latest],
            "historical": [str(r["path"]) for r in best],
            "anchor": list(self.anchor_paths),
            "weights": dict(self.weights),
            "records": list(by_update),
        }

    def write(self) -> None:
        """Write the manifest atomically.

        write_text truncates then writes, so a crash mid-write left a truncated
        file that the next run could not parse. Rename from a temp file so a
        reader sees either the old manifest or the new one.
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, self.manifest_path)
        finally:
            tmp.unlink(missing_ok=True)
