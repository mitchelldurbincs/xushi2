"""Tests for the entropy-bonus and log_std anneals: schedule math."""

from __future__ import annotations

import pytest

from train.mappo_model import compute_entropy_scale, compute_log_std_offset


def test_disabled_holds_one():
    assert compute_entropy_scale(update=250, anneal_updates=0, final_scale=0.0) == 1.0


def test_midpoint_linear():
    assert compute_entropy_scale(
        update=200, anneal_updates=400, final_scale=0.0
    ) == pytest.approx(0.5)


def test_reaches_and_holds_final():
    assert compute_entropy_scale(update=400, anneal_updates=400, final_scale=0.0) == 0.0
    assert compute_entropy_scale(update=999, anneal_updates=400, final_scale=0.0) == 0.0


def test_nonzero_final_scale():
    assert compute_entropy_scale(
        update=100, anneal_updates=200, final_scale=0.5
    ) == pytest.approx(0.75)


def test_update_zero_is_full_scale():
    assert compute_entropy_scale(update=0, anneal_updates=400, final_scale=0.0) == 1.0


def test_log_std_offset_disabled_is_zero():
    assert compute_log_std_offset(update=200, anneal_updates=0, final_offset=-1.386) == 0.0


def test_log_std_offset_midpoint_linear():
    assert compute_log_std_offset(
        update=150, anneal_updates=300, final_offset=-1.386
    ) == pytest.approx(-0.693)


def test_log_std_offset_reaches_and_holds_final():
    assert compute_log_std_offset(
        update=300, anneal_updates=300, final_offset=-1.386
    ) == pytest.approx(-1.386)
    assert compute_log_std_offset(
        update=999, anneal_updates=300, final_offset=-1.386
    ) == pytest.approx(-1.386)
