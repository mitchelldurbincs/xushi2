"""Tests for the Phase-1 observation manifest.

The manifest is the source of truth for the actor- and critic-side flat
observation tensors. Field widths sum to the declared total, and the slice
lookup helpers match the declared field ordering.
"""

from __future__ import annotations

import pytest

from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    ACTOR_PHASE1_FIELDS,
    CRITIC_PHASE1_DIM,
    CRITIC_PHASE1_FIELDS,
    actor_field_slice,
    critic_field_slice,
)


def test_actor_dim_matches_field_widths():
    assert ACTOR_PHASE1_DIM == sum(width for _, width, _ in ACTOR_PHASE1_FIELDS)


def test_actor_dim_matches_spec_value():
    # See docs/observation_spec.md §Phase 1. Also must equal
    # src/sim/include/xushi2/sim/obs.h::kActorObsPhase1Dim.
    assert ACTOR_PHASE1_DIM == 31


def test_critic_dim_matches_field_widths():
    assert CRITIC_PHASE1_DIM == sum(width for _, width, _ in CRITIC_PHASE1_FIELDS)


def test_critic_dim_matches_spec_value():
    # Must equal src/sim/include/xushi2/sim/obs.h::kCriticObsPhase1Dim.
    assert CRITIC_PHASE1_DIM == 45


def test_critic_is_at_least_as_wide_as_actor():
    # Critic obs is a superset; the minimum invariant is >= actor dim.
    assert CRITIC_PHASE1_DIM >= ACTOR_PHASE1_DIM


def test_actor_field_names_are_unique():
    names = [name for name, _, _ in ACTOR_PHASE1_FIELDS]
    assert len(names) == len(set(names))


def test_critic_field_names_are_unique():
    names = [name for name, _, _ in CRITIC_PHASE1_FIELDS]
    assert len(names) == len(set(names))


def test_actor_slice_own_hp_first():
    # own_hp is the first declared field, width 1.
    s = actor_field_slice("own_hp")
    assert s.start == 0
    assert s.stop == 1


def test_actor_slice_own_position_width_two():
    s = actor_field_slice("own_position")
    assert s.stop - s.start == 2


def test_actor_slice_objective_owner_onehot_width_three():
    s = actor_field_slice("objective_owner_onehot")
    assert s.stop - s.start == 3


def test_actor_slices_cover_full_tensor_contiguously():
    # Walking the fields in order yields non-overlapping, contiguous slices
    # that together span [0, ACTOR_PHASE1_DIM).
    cursor = 0
    for name, width, _ in ACTOR_PHASE1_FIELDS:
        s = actor_field_slice(name)
        assert s.start == cursor
        assert s.stop == cursor + width
        cursor = s.stop
    assert cursor == ACTOR_PHASE1_DIM


def test_critic_slices_cover_full_tensor_contiguously():
    cursor = 0
    for name, width, _ in CRITIC_PHASE1_FIELDS:
        s = critic_field_slice(name)
        assert s.start == cursor
        assert s.stop == cursor + width
        cursor = s.stop
    assert cursor == CRITIC_PHASE1_DIM


def test_unknown_actor_field_raises_key_error():
    with pytest.raises(KeyError):
        actor_field_slice("does_not_exist")


def test_unknown_critic_field_raises_key_error():
    with pytest.raises(KeyError):
        critic_field_slice("does_not_exist")


def test_actor_expected_field_count_equals_spec():
    # Checkpoint: observation_spec.md §Phase 1 enumerates the fields below.
    # Changes here are breaking — bump a version and update C++ obs.h.
    expected = {
        "own_hp",
        "own_velocity",
        "own_aim_unit",
        "own_position",
        "own_ammo",
        "own_reloading",
        "own_combat_roll_cd",
        "enemy_alive",
        "enemy_respawn_timer",
        "enemy_relative_position",
        "enemy_hp",
        "enemy_velocity",
        "objective_owner_onehot",
        "cap_team_onehot",
        "cap_progress",
        "contested",
        "objective_unlocked",
        "own_score",
        "enemy_score",
        "self_on_point",
        "enemy_on_point",
        "round_timer",
    }
    got = {name for name, _, _ in ACTOR_PHASE1_FIELDS}
    assert got == expected
