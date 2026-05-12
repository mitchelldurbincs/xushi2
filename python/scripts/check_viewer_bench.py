#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def regressions(curr: dict, base: dict, tolerance_pct: float) -> list[str]:
    keys = ["avg_ms", "p50_ms", "p95_ms", "p99_ms"]
    fails: list[str] = []
    for k in keys:
        c = float(curr[k])
        b = float(base[k])
        limit = b * (1.0 + tolerance_pct / 100.0)
        if c > limit:
            fails.append(f"{k}: current={c:.3f}ms baseline={b:.3f}ms limit={limit:.3f}ms")
    cfps = float(curr["fps"])
    bfps = float(base["fps"])
    floor = bfps * (1.0 - tolerance_pct / 100.0)
    if cfps < floor:
        fails.append(f"fps: current={cfps:.3f} baseline={bfps:.3f} floor={floor:.3f}")
    return fails


def main() -> int:
    p = argparse.ArgumentParser(description="Check viewer benchmark JSON against baseline")
    p.add_argument("--result", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--tolerance-pct", type=float, default=15.0)
    args = p.parse_args()

    result = load_json(Path(args.result))
    baseline = load_json(Path(args.baseline))
    failures = regressions(result, baseline, args.tolerance_pct)
    if failures:
        print("Viewer benchmark regression detected:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"Viewer benchmark OK (tolerance {args.tolerance_pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
