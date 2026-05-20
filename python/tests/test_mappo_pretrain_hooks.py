from __future__ import annotations

from types import SimpleNamespace

from train.mappo_pretrain_hooks import (
    maybe_run_bc_pretrain,
    maybe_run_composition_pretrain,
    maybe_warm_start,
)


def _context(**run_overrides):
    run_cfg = {
        "composition_pretrain": False,
        "bc_pretrain_steps": 0,
        **run_overrides,
    }
    return SimpleNamespace(run_cfg=run_cfg, phase_label="phase4")


def test_warm_start_noops_without_checkpoint() -> None:
    maybe_warm_start(_context(), SimpleNamespace(model=object()))


def test_composition_pretrain_noops_when_disabled() -> None:
    result = maybe_run_composition_pretrain(
        _context(composition_pretrain=False),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True


def test_bc_pretrain_noops_when_steps_zero() -> None:
    result = maybe_run_bc_pretrain(
        _context(bc_pretrain_steps=0),
        SimpleNamespace(model=object()),
        SimpleNamespace(),
    )

    assert result is True
