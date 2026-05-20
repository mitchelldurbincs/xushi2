"""Unit tests for Phase-2 memory-toy eval harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import eval_memory_toy as cli_mod
from eval.memory_toy_gate import (
    AblationResult,
    GateAggregateResult,
    evaluate_gate_thresholds,
    load_memory_toy_gate_thresholds,
)


def test_load_thresholds_rejects_extra_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text(
        "\n".join(
            [
                "version: memory_toy_gate/vx",
                "normal_mean_min: -0.15",
                "zero_mean_range: [-1.2, -0.8]",
                "random_mean_range: [-1.5, -0.8]",
                "normal_zero_gap_min: 0.5",
                "unexpected: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected keys"):
        load_memory_toy_gate_thresholds(cfg)


def test_evaluate_gate_thresholds_uses_configurable_values() -> None:
    thresholds = load_memory_toy_gate_thresholds(cli_mod.DEFAULT_THRESHOLDS)
    results = {
        "normal": AblationResult(mean=-0.05, ci95=0.01, n=16),
        "zero_every_tick": AblationResult(mean=-1.0, ci95=0.01, n=16),
        "random_every_tick": AblationResult(mean=-1.1, ci95=0.01, n=16),
    }
    ok, failures = evaluate_gate_thresholds(results, thresholds)
    assert ok is True
    assert failures == []


def test_cli_prints_threshold_metadata(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"stub")

    monkeypatch.setattr(cli_mod, "load_checkpoint", lambda _p: (object(), {}))

    def _fake_eval(**_kwargs):
        thresholds = load_memory_toy_gate_thresholds(cli_mod.DEFAULT_THRESHOLDS)
        return GateAggregateResult(
            per_mode={
                "normal": AblationResult(mean=-0.05, ci95=0.01, n=2),
                "zero_every_tick": AblationResult(mean=-1.0, ci95=0.01, n=2),
                "random_every_tick": AblationResult(mean=-1.1, ci95=0.01, n=2),
            },
            gap_normal_minus_zero=0.95,
            passed=True,
            failure_reasons=[],
            thresholds=thresholds,
        )

    monkeypatch.setattr(cli_mod, "evaluate_memory_toy_gate", _fake_eval)
    monkeypatch.setattr(
        "sys.argv",
        ["eval_memory_toy", "--checkpoint", str(ckpt), "--episodes", "2", "--seed", "7"],
    )

    assert cli_mod.main() == 0
    out = capsys.readouterr().out
    assert "thresholds:" in out
    assert "version=memory_toy_gate/v1" in out
    assert "source=" in out
