"""Path resolution helpers for the Python test suite.

Tests historically opened config files via relative paths like
``../experiments/configs/foo.yaml``, which only resolved when pytest was
invoked from the ``python/`` directory. The helpers here compute paths
relative to the repo root derived from this file's location, so config
resolution works regardless of pytest's cwd.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Absolute path under the repo root, joined from ``parts``."""
    return REPO_ROOT.joinpath(*parts)


def config_path(rel: str) -> Path:
    """Absolute path to a file under ``experiments/configs/``."""
    return REPO_ROOT / "experiments" / "configs" / rel
