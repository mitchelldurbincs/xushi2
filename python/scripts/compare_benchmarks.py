"""Compare a benchmark run against a stored baseline.

Default behavior is warning-oriented: regressions beyond warn thresholds do not
fail the process so noisy environments (e.g., shared CI runners) do not
immediately block all contributors.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineMetric:
    name: str
    direction: str
    median: float


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_metric_map(baseline_payload: dict) -> dict[str, BaselineMetric]:
    out: dict[str, BaselineMetric] = {}
    for entry in baseline_payload.get("benchmarks", []):
        out[entry["name"]] = BaselineMetric(
            name=entry["name"],
            direction=entry["direction"],
            median=float(entry["median"]),
        )
    return out


def _median_from_entry(entry: dict) -> float:
    if "median" in entry:
        return float(entry["median"])
    values = entry.get("values", [])
    if not values:
        raise ValueError(f"Metric '{entry.get('name', '<unknown>')}' has no median or values")
    return float(statistics.median(float(v) for v in values))


def _regression_pct(direction: str, baseline: float, current: float) -> float:
    if math.isclose(baseline, 0.0):
        return 0.0 if math.isclose(current, 0.0) else math.inf
    if direction == "lower_is_better":
        return ((current - baseline) / baseline) * 100.0
    if direction == "higher_is_better":
        return ((baseline - current) / baseline) * 100.0
    raise ValueError(f"Unsupported direction: {direction}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline JSON")
    parser.add_argument("--current", type=Path, required=True, help="Path to current benchmark JSON")
    parser.add_argument(
        "--warn-threshold-pct",
        type=float,
        default=None,
        help="Warn threshold override; defaults to baseline.thresholds.warn_regression_pct (or 12)",
    )
    parser.add_argument(
        "--fail-threshold-pct",
        type=float,
        default=None,
        help=(
            "Optional fail threshold override. If omitted, script never fails for regressions and only warns."
        ),
    )
    args = parser.parse_args()

    baseline_payload = _load_json(args.baseline)
    current_payload = _load_json(args.current)

    baseline_metrics = _to_metric_map(baseline_payload)
    current_entries = current_payload.get("benchmarks", [])
    current_map = {entry["name"]: entry for entry in current_entries}

    warn_threshold = args.warn_threshold_pct
    if warn_threshold is None:
        warn_threshold = float(
            baseline_payload.get("thresholds", {}).get("warn_regression_pct", 12.0)
        )
    fail_threshold = args.fail_threshold_pct
    if fail_threshold is None and baseline_payload.get("thresholds"):
        fail_threshold = baseline_payload["thresholds"].get("fail_regression_pct")

    print(f"Comparing benchmarks using warn threshold {warn_threshold:.2f}%")
    if fail_threshold is None:
        print("Fail threshold disabled (warning-only mode)")
    else:
        print(f"Fail threshold enabled at {float(fail_threshold):.2f}%")

    warned = 0
    failed = 0

    for metric_name, baseline_metric in baseline_metrics.items():
        if metric_name not in current_map:
            print(f"[WARN] Missing current metric: {metric_name}")
            warned += 1
            continue

        current_median = _median_from_entry(current_map[metric_name])
        regression = _regression_pct(
            baseline_metric.direction,
            baseline_metric.median,
            current_median,
        )
        print(
            f"- {metric_name}: baseline={baseline_metric.median:.6g}, "
            f"current={current_median:.6g}, regression={regression:.2f}%"
        )

        if regression >= warn_threshold:
            print(
                f"  [WARN] Regression >= warn threshold ({warn_threshold:.2f}%): "
                f"{regression:.2f}%"
            )
            warned += 1
        if fail_threshold is not None and regression >= float(fail_threshold):
            print(
                f"  [FAIL] Regression >= fail threshold ({float(fail_threshold):.2f}%): "
                f"{regression:.2f}%"
            )
            failed += 1

    if warned:
        print(f"Summary: {warned} warning(s), {failed} failure(s)")
    else:
        print("Summary: no regressions above thresholds")

    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
