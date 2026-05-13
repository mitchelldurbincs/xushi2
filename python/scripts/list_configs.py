#!/usr/bin/env python3
"""Discover experiment configs and print canonical train commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path("experiments/configs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Config root directory.")
    parser.add_argument("--phase", help="Filter metadata.phase (e.g., phase4).")
    parser.add_argument("--purpose", help="Filter metadata.purpose (e.g., baseline/smoke/probe).")
    parser.add_argument("--status", help="Filter metadata.status (e.g., active/legacy).")
    parser.add_argument(
        "--format",
        choices=("table", "commands"),
        default="table",
        help="Output mode: metadata table or only canonical commands.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def discover_configs(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.yaml")):
        data = load_yaml(path)
        if not data:
            continue
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            continue
        rel_path = path.as_posix()
        records.append(
            {
                "path": rel_path,
                "phase": str(meta.get("phase", "")),
                "purpose": str(meta.get("purpose", "")),
                "status": str(meta.get("status", "")),
                "expected_runtime": str(meta.get("expected_runtime", "")),
                "gate_relevance": meta.get("gate_relevance", []),
                "lineage": str(meta.get("lineage", "")),
                "command": f"python -m train.train --config {rel_path}",
            }
        )
    return records


def apply_filters(records: list[dict[str, Any]], phase: str | None, purpose: str | None, status: str | None) -> list[dict[str, Any]]:
    result = records
    if phase:
        result = [r for r in result if r["phase"] == phase]
    if purpose:
        result = [r for r in result if r["purpose"] == purpose]
    if status:
        result = [r for r in result if r["status"] == status]
    return result


def print_table(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No configs matched filters.")
        return
    for r in records:
        gate = ",".join(r["gate_relevance"]) if isinstance(r["gate_relevance"], list) else str(r["gate_relevance"])
        print(f"{r['path']}")
        print(f"  phase={r['phase']} purpose={r['purpose']} status={r['status']} runtime={r['expected_runtime']}")
        print(f"  gate_relevance={gate}")
        print(f"  lineage={r['lineage']}")
        print(f"  cmd: {r['command']}")


def main() -> int:
    args = parse_args()
    records = discover_configs(args.root)
    filtered = apply_filters(records, args.phase, args.purpose, args.status)
    if args.format == "commands":
        for r in filtered:
            print(r["command"])
        return 0
    print_table(filtered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
