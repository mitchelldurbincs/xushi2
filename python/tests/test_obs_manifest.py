"""Tests for the observation manifest.

The manifest is the source of truth for the actor- and critic-side flat
observation tensors. Field widths sum to the declared total, and the slice
lookup helpers match the declared field ordering.
"""

from __future__ import annotations

import pytest

from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    ACTOR_PHASE1_FIELDS,
    CRITIC_DIM,
    CRITIC_FIELDS,
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
    assert CRITIC_DIM == sum(width for _, width, _ in CRITIC_FIELDS)


def test_critic_dim_is_135():
    # Must equal src/sim/include/xushi2/sim/obs.h::kCriticObsDim.
    assert CRITIC_DIM == 135


def test_critic_is_at_least_as_wide_as_actor():
    # Critic obs is a superset; the minimum invariant is >= actor dim.
    assert CRITIC_DIM >= ACTOR_PHASE1_DIM


def test_actor_field_names_are_unique():
    names = [name for name, _, _ in ACTOR_PHASE1_FIELDS]
    assert len(names) == len(set(names))


def test_critic_field_names_are_unique():
    names = [name for name, _, _ in CRITIC_FIELDS]
    assert len(names) == len(set(names))


def test_critic_fields_have_expected_layout():
    # First 3*len(actor_fields) are slotN/<actor_field> for N in 0,1,2.
    actor_names = [name for name, _, _ in ACTOR_PHASE1_FIELDS]
    expected_prefix: list[str] = []
    for slot in range(3):
        for name in actor_names:
            expected_prefix.append(f"slot{slot}/{name}")

    actual_names = [name for name, _, _ in CRITIC_FIELDS]
    assert actual_names[: len(expected_prefix)] == expected_prefix

    # Then enemyN/<world block> for N in 0,1,2 (9 fields each).
    enemy_field_names = [
        "world_position", "world_velocity", "world_aim_unit",
        "hp_normalized", "alive_flag", "respawn_timer",
        "ammo", "reloading", "combat_roll_cd",
    ]
    cursor = len(expected_prefix)
    for enemy in range(3):
        for name in enemy_field_names:
            assert actual_names[cursor] == f"enemy{enemy}/{name}", (
                f"at index {cursor}, got {actual_names[cursor]!r}"
            )
            cursor += 1

    # Then objective + seed unprefixed.
    tail = [
        "cap_progress_ticks", "team_a_score_ticks",
        "team_b_score_ticks", "tick_raw", "seed_hi", "seed_lo",
    ]
    assert actual_names[cursor:] == tail


def test_critic_phase1_symbols_removed():
    import xushi2.obs_manifest as m
    assert not hasattr(m, "CRITIC_PHASE1_FIELDS")
    assert not hasattr(m, "CRITIC_PHASE1_DIM")


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
    cursor = 0
    for name, width, _ in ACTOR_PHASE1_FIELDS:
        s = actor_field_slice(name)
        assert s.start == cursor
        assert s.stop == cursor + width
        cursor = s.stop
    assert cursor == ACTOR_PHASE1_DIM


def test_critic_slices_cover_full_tensor_contiguously():
    cursor = 0
    for name, width, _ in CRITIC_FIELDS:
        s = critic_field_slice(name)
        assert s.start == cursor
        assert s.stop == cursor + width
        cursor = s.stop
    assert cursor == CRITIC_DIM


def test_unknown_actor_field_raises_key_error():
    with pytest.raises(KeyError):
        actor_field_slice("does_not_exist")


def test_unknown_critic_field_raises_key_error():
    with pytest.raises(KeyError):
        critic_field_slice("does_not_exist")


def test_actor_expected_field_count_equals_spec():
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
