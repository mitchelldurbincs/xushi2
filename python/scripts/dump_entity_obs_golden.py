"""Regenerate the per-platform entity-obs golden fixtures.

Thin wrapper over the scenario definitions in
tests/test_entity_obs_golden.py so the scenarios cannot drift from what the
regression test replays. Run from the python/ directory:

    .venv/bin/python scripts/dump_entity_obs_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PY_ROOT))
sys.path.insert(0, str(_PY_ROOT / "tests"))

from test_entity_obs_golden import regenerate_all  # noqa: E402


def main() -> int:
    for path in regenerate_all():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
