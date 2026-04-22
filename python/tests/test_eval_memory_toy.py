"""Unit tests for Phase-2 memory-toy eval harness."""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from eval import eval_memory_toy as eval_mod


def _call_with_known_kwargs(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the kwargs its signature accepts."""
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)


def _mk_result(mean: float, *, ci95: float = 0.01, n: int = 32) -> Any:
    """Build an ``AblationResult`` instance robustly across minor API drift."""
    cls = eval_mod.AblationResult
    sig = inspect.signature(cls)

    values: dict[str, Any] = {
        "mean": mean,
        "ci95": ci95,
        "n": n,
        "num_episodes": n,
        "episodes": n,
        "mode": "test",
        "seed": 0,
    }
    kwargs = {
        name: values[name]
        for name in sig.parameters
        if name != "self" and name in values
    }

    if is_dataclass(cls):
        for f in fields(cls):
            if f.name not in kwargs:
                if f.default is not inspect._empty:  # type: ignore[attr-defined]
                    kwargs[f.name] = f.default
                elif f.default_factory is not inspect._empty:  # type: ignore[attr-defined]
                    kwargs[f.name] = f.default_factory()

    return cls(**kwargs)


def test_ablation_modes_are_behaviorally_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert mode routing is distinct (not all aliased to one code path)."""
    called_modes: list[str] = []

    def _fake_run_ablation(*_: Any, **kwargs: Any) -> Any:
        mode = kwargs["mode"]
        called_modes.append(mode)
        means = {
            "normal": -0.05,
            "zero_every_tick": -1.0,
            "random_every_tick": -1.1,
        }
        return _mk_result(mean=means[mode])

    monkeypatch.setattr(eval_mod, "run_ablation", _fake_run_ablation)

    out = _call_with_known_kwargs(
        eval_mod.ablation_modes_differ,
        model=object(),
        config={},
        num_episodes=8,
        seed=123,
    )

    assert out is True
    assert set(called_modes) == {"normal", "zero_every_tick", "random_every_tick"}


@pytest.mark.parametrize(
    ("normal", "zero", "random_", "expected_fragment"),
    [
        (-0.20, -1.00, -1.10, "normal_mean"),
        (-0.05, -1.25, -1.10, "zero_every_tick_mean"),
        (-0.05, -1.00, -1.60, "random_every_tick_mean"),
        (-0.40, -0.10, -1.10, "gap normal-zero"),
    ],
)
def test_check_gate_reports_each_failure_condition(
    normal: float,
    zero: float,
    random_: float,
    expected_fragment: str,
) -> None:
    ok, failures = eval_mod._check_gate(
        _mk_result(normal),
        _mk_result(zero),
        _mk_result(random_),
    )

    assert ok is False
    assert any(expected_fragment in msg for msg in failures)


def test_check_gate_passes_for_expected_band() -> None:
    ok, failures = eval_mod._check_gate(
        _mk_result(-0.05),
        _mk_result(-1.0),
        _mk_result(-1.1),
    )

    assert ok is True
    assert failures == []


@pytest.mark.parametrize(
    ("normal_mean", "expected_rc"),
    [
        (-0.05, 0),
        (-0.30, 1),
    ],
)
def test_cli_return_code_from_synthesized_ablation_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    normal_mean: float,
    expected_rc: int,
) -> None:
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"stub")

    monkeypatch.setattr(eval_mod, "load_checkpoint", lambda _p: (object(), {}))

    means = {
        "normal": normal_mean,
        "zero_every_tick": -1.0,
        "random_every_tick": -1.1,
    }

    def _fake_run_ablation(*_: Any, **kwargs: Any) -> Any:
        return _mk_result(mean=means[kwargs["mode"]], n=16)

    monkeypatch.setattr(eval_mod, "run_ablation", _fake_run_ablation)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_memory_toy",
            "--checkpoint",
            str(ckpt),
            "--episodes",
            "16",
            "--seed",
            "7",
        ],
    )

    assert eval_mod.main() == expected_rc
