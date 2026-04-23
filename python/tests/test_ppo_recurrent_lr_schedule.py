from __future__ import annotations

import math

import pytest

from train.ppo_recurrent import lr_for_update


def test_lr_for_update_constant_schedule_stays_at_base_lr() -> None:
    base_lr = 3.0e-4

    assert lr_for_update(1, 10, base_lr=base_lr) == pytest.approx(base_lr)
    assert lr_for_update(5, 10, base_lr=base_lr) == pytest.approx(base_lr)
    assert lr_for_update(10, 10, base_lr=base_lr) == pytest.approx(base_lr)


def test_lr_for_update_linear_schedule_hits_first_and_last_update() -> None:
    base_lr = 3.0e-4

    assert lr_for_update(
        1,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.15,
    ) == pytest.approx(base_lr)
    assert lr_for_update(
        10,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.15,
    ) == pytest.approx(base_lr * 0.15)


def test_lr_for_update_cosine_schedule_hits_expected_midpoint() -> None:
    base_lr = 3.0e-4
    expected_ratio = 0.2 + 0.5 * (1.0 - 0.2) * (1.0 + math.cos(math.pi * 0.5))

    assert lr_for_update(
        1,
        9,
        base_lr=base_lr,
        schedule="cosine",
        lr_final_ratio=0.2,
    ) == pytest.approx(base_lr)
    assert lr_for_update(
        5,
        9,
        base_lr=base_lr,
        schedule="cosine",
        lr_final_ratio=0.2,
    ) == pytest.approx(base_lr * expected_ratio)
    assert lr_for_update(
        9,
        9,
        base_lr=base_lr,
        schedule="cosine",
        lr_final_ratio=0.2,
    ) == pytest.approx(base_lr * 0.2)


def test_lr_for_update_warmup_ramps_then_starts_schedule_at_base_lr() -> None:
    base_lr = 3.0e-4

    assert lr_for_update(
        1,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.2,
        warmup_updates=3,
    ) == pytest.approx(base_lr / 3.0)
    assert lr_for_update(
        2,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.2,
        warmup_updates=3,
    ) == pytest.approx(2.0 * base_lr / 3.0)
    assert lr_for_update(
        3,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.2,
        warmup_updates=3,
    ) == pytest.approx(base_lr)
    assert lr_for_update(
        4,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.2,
        warmup_updates=3,
    ) == pytest.approx(base_lr)
    assert lr_for_update(
        10,
        10,
        base_lr=base_lr,
        schedule="linear",
        lr_final_ratio=0.2,
        warmup_updates=3,
    ) == pytest.approx(base_lr * 0.2)


def test_lr_for_update_rejects_unknown_schedule_name() -> None:
    with pytest.raises(ValueError, match="unknown lr schedule"):
        lr_for_update(1, 10, base_lr=3.0e-4, schedule="triangle")
