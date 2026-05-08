"""Check a MAPPO matchup-matrix JSON file against compact gate thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xushi2.mappo_matrix_gate import check_matrix_gate


def _parse_thresholds(values: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"threshold must be opponent_type=value, got {raw!r}")
        key, value = raw.split("=", 1)
        out[str(key)] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a MAPPO matrix JSON artifact")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-win-rate", action="append", default=[])
    parser.add_argument("--max-draw-rate", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.matrix.read_text(encoding="utf-8"))
    summary = check_matrix_gate(
        rows,
        {
            "min_rows": int(args.min_rows),
            "min_win_rate": _parse_thresholds([str(v) for v in args.min_win_rate]),
            "max_draw_rate": _parse_thresholds([str(v) for v in args.max_draw_rate]),
        },
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if summary["passed"] else "FAIL"
    print(
        f"[mappo_matrix_gate] {status} rows={summary['row_count']} "
        f"types={summary['counts_by_type']}"
    )
    for failure in summary["failures"]:
        print(f"[mappo_matrix_gate] failure: {failure}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
