"""Smoke test for xushi2-eval --dump-obs / --dump-reward (Phase 1b).

We call eval.main() with a fresh argv and verify the CSVs land with
sensible shapes. This only exercises the env-mode dump — the Phase-0
hash-dump path stays covered by tests/test_phase0_determinism.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eval import eval as eval_mod
from xushi2.obs_manifest import ACTOR_PHASE1_DIM


_BASE_MECH_ARGS: list[str] = [
    "--revolver-damage-centi-hp", "7500",
    "--revolver-fire-cooldown-ticks", "15",
    "--revolver-hitbox-radius", "0.75",
    "--respawn-ticks", "240",
]


def _run_eval(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["xushi2-eval"] + argv)
    exit_code = eval_mod.main()
    assert exit_code == 0


def test_dump_obs_writes_csv_with_expected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_csv = tmp_path / "obs.csv"
    _run_eval(
        _BASE_MECH_ARGS
        + [
            "--dump-obs", str(out_csv),
            "--opponent-bot", "noop",
            "--learner-team", "A",
            "--round-length-seconds", "3",
        ],
        monkeypatch,
    )
    assert out_csv.exists()
    with out_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    # 1 tick col + ACTOR_PHASE1_DIM obs cols.
    assert len(header) == 1 + ACTOR_PHASE1_DIM
    # At 10 Hz decisions over 3s we expect ~30 rows, ±a couple for
    # terminal timing. Require > 0 and < 200 (sanity upper bound).
    assert 0 < len(rows) < 200


def test_dump_reward_writes_csv_with_expected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_csv = tmp_path / "reward.csv"
    _run_eval(
        _BASE_MECH_ARGS
        + [
            "--dump-reward", str(out_csv),
            "--opponent-bot", "noop",
            "--learner-team", "A",
            "--round-length-seconds", "3",
        ],
        monkeypatch,
    )
    assert out_csv.exists()
    with out_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["tick", "step_reward_learner", "reward_team_a", "reward_team_b"]
    assert 0 < len(rows) < 200


def test_dump_obs_without_opponent_bot_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_csv = tmp_path / "obs.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["xushi2-eval"] + _BASE_MECH_ARGS
        + [
            "--dump-obs", str(out_csv),
            "--round-length-seconds", "3",
        ],
    )
    # argparse's parser.error() raises SystemExit(2).
    with pytest.raises(SystemExit):
        eval_mod.main()
