"""Evaluation entrypoint.

Runs a policy (or snapshot) against scripted-bot baselines and reports:
- win rate per baseline
- behavioral metrics (game-design §13, rl-design §11)
- per-episode replay IDs for post-hoc inspection
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 evaluation entrypoint")
    parser.add_argument("--policy", type=str, help="Policy checkpoint path (unused in Phase 0)")
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()

    print(f"[xushi2] eval entrypoint invoked (policy={args.policy}, episodes={args.episodes})")
    print("[xushi2] Phase 0 — eval harness not yet implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
