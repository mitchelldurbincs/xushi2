"""Regression tests for the path-resolution helpers in `_paths`.

The helpers exist so config-file tests work regardless of pytest's cwd —
historically tests opened ``../experiments/configs/foo.yaml`` and only
passed when pytest was invoked from ``python/``.
"""

from __future__ import annotations

from pathlib import Path

from _paths import REPO_ROOT, config_path, repo_path


def test_repo_root_contains_experiments_configs() -> None:
    assert (REPO_ROOT / "experiments" / "configs").is_dir()


def test_config_path_resolves_real_file() -> None:
    # phase0_determinism.yaml has existed since Phase 0; canary for the resolver.
    p = config_path("phase0_determinism.yaml")
    assert p.is_file(), f"expected file at {p}"


def test_repo_path_joins_under_root() -> None:
    assert repo_path("python", "tests", "_paths.py").is_file()


def test_config_path_independent_of_cwd(tmp_path: Path, monkeypatch) -> None:
    expected = config_path("phase0_determinism.yaml")
    monkeypatch.chdir(tmp_path)
    assert config_path("phase0_determinism.yaml") == expected
    assert config_path("phase0_determinism.yaml").is_file()
