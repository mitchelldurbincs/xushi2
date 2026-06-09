"""Standalone benchmark harness for eval paths.

Stable output schema:
{
  "suite_version": "v1",
  "benchmarks": [...],
  "system_info": {...}
}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xushi2.runner import run_episode


SUITE_VERSION = "v1"


@dataclass
class BenchResult:
    mode: str
    repeat_idx: int
    episodes: int
    warmup_episodes: int
    seed: int
    total_wall_time_sec: float
    episodes_per_sec: float
    mean_ms_per_episode: float
    mean_ms_per_step: float | None
    step_count: int | None


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return proc.stdout.strip() or None


def _system_info() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def _run_phase0_scripted(args: argparse.Namespace, seed: int, episodes: int) -> tuple[float, int]:
    sim_cfg = {
        "seed": seed,
        "round_length_seconds": args.round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "mechanics": {
            "revolver_damage_centi_hp": args.revolver_damage_centi_hp,
            "revolver_fire_cooldown_ticks": args.revolver_fire_cooldown_ticks,
            "revolver_hitbox_radius": args.revolver_hitbox_radius,
            "respawn_ticks": args.respawn_ticks,
        },
    }
    t0 = time.perf_counter()
    total_steps = 0
    for ep_idx in range(episodes):
        result = run_episode(sim_cfg, args.bot_a, args.bot_b, seed_override=seed + ep_idx)
        total_steps += int(result.final_tick)
    elapsed = time.perf_counter() - t0
    return elapsed, total_steps


def _run_mode(args: argparse.Namespace, seed: int, episodes: int) -> tuple[float, int | None]:
    if args.mode == "phase0_scripted":
        return _run_phase0_scripted(args, seed=seed, episodes=episodes)
    raise ValueError(f"unsupported mode: {args.mode!r}")


def _to_payload(args: argparse.Namespace, rows: list[BenchResult]) -> dict[str, Any]:
    return {
        "suite_version": SUITE_VERSION,
        "benchmarks": [
            {
                **asdict(row),
                "config": {
                    "bot_a": args.bot_a,
                    "bot_b": args.bot_b,
                    "round_length_seconds": args.round_length_seconds,
                    "revolver_damage_centi_hp": args.revolver_damage_centi_hp,
                    "revolver_fire_cooldown_ticks": args.revolver_fire_cooldown_ticks,
                    "revolver_hitbox_radius": args.revolver_hitbox_radius,
                    "respawn_ticks": args.respawn_ticks,
                },
            }
            for row in rows
        ],
        "system_info": _system_info(),
    }


def _write_csv(path: str, rows: list[BenchResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark eval harness")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("phase0_scripted",),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--csv-out", type=str, default=None)

    parser.add_argument("--bot-a", type=str, default="basic")
    parser.add_argument("--bot-b", type=str, default="basic")
    parser.add_argument("--round-length-seconds", type=int, default=30)
    parser.add_argument("--revolver-damage-centi-hp", type=int, default=600)
    parser.add_argument("--revolver-fire-cooldown-ticks", type=int, default=15)
    parser.add_argument("--revolver-hitbox-radius", type=float, default=0.75)
    parser.add_argument("--respawn-ticks", type=int, default=120)
    args = parser.parse_args()

    rows: list[BenchResult] = []
    for r in range(args.repeat):
        run_seed = args.seed + r * 100_000
        if args.warmup_episodes > 0:
            _run_mode(args, seed=run_seed, episodes=args.warmup_episodes)

        elapsed, step_count = _run_mode(args, seed=run_seed, episodes=args.episodes)
        eps_per_sec = float(args.episodes) / elapsed if elapsed > 0 else float("inf")
        ms_per_ep = (elapsed * 1_000.0) / float(args.episodes)
        ms_per_step = (elapsed * 1_000.0) / float(step_count) if step_count and step_count > 0 else None
        rows.append(
            BenchResult(
                mode=args.mode,
                repeat_idx=r,
                episodes=args.episodes,
                warmup_episodes=args.warmup_episodes,
                seed=run_seed,
                total_wall_time_sec=elapsed,
                episodes_per_sec=eps_per_sec,
                mean_ms_per_episode=ms_per_ep,
                mean_ms_per_step=ms_per_step,
                step_count=step_count,
            )
        )

    payload = _to_payload(args, rows)
    print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        _write_csv(args.csv_out, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
