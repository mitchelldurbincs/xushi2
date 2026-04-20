"""Training entrypoint.

Phase 0: this is a skeleton. Real Phase-1 PPO lands alongside the first
flat-observation env wrapper.

Usage:
    python -m train.train --config experiments/configs/phase0_determinism.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 training entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training config YAML under experiments/configs/",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    phase = config.get("phase", "unknown")

    print(f"[xushi2] training entrypoint invoked (phase={phase})")
    print("[xushi2] Phase 0 — trainer not yet implemented.")
    print("[xushi2] Next step: Phase 1 curriculum (feedforward PPO, flat obs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
