#!/usr/bin/env python3
"""Run benchmark_sim and write CI-friendly artifacts.

This script does not enforce performance thresholds; it only records outputs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--artifact-dir", default="artifacts/benchmarks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dir = pathlib.Path(args.build_dir)
    artifact_dir = pathlib.Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    benchmark_bin = build_dir / "src" / "sim" / "tools" / "benchmark_sim"
    if not benchmark_bin.exists():
        print(f"error: benchmark binary not found: {benchmark_bin}", file=sys.stderr)
        print("hint: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        return 2

    json_out = subprocess.run(
        [str(benchmark_bin), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    csv_out = subprocess.run(
        [str(benchmark_bin), "--csv"],
        check=True,
        capture_output=True,
        text=True,
    )

    json_path = artifact_dir / "sim_benchmark.json"
    csv_path = artifact_dir / "sim_benchmark.csv"

    parsed = json.loads(json_out.stdout)
    json_path.write_text(json.dumps(parsed, indent=2) + "\n")
    csv_path.write_text(csv_out.stdout)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
