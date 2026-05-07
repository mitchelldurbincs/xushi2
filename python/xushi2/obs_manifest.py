"""Observation manifest — canonical field order for the actor- and
critic-side flat observation tensors.

This module is the single source of truth for observation layout on the
Python side. The C++ side mirrors these constants in
`src/sim/include/xushi2/sim/obs.h`; the two files MUST stay in lockstep.
Changing field order or width is a breaking change: old checkpoints become
invalid, and downstream reward / env / replay code that indexes by position
must be updated in the same commit.

Actor layout is documented in `docs/observation_spec.md` §Phase 1.
Phase-4 critic layout: 3 own-team actor mirrors (slot{0,1,2}/<actor_field>)
followed by 3 enemy world-frame blocks (enemy{0,1,2}/<world_field>) followed
by raw objective counters and seed bits. See
`docs/plans/2026-05-07-phase4-critic-obs-design.md` for layout rationale.

Two invariants hold (see `docs/observation_spec.md` §"Observation invariants"):

1. The actor and critic builders are separate top-level functions with
   separate manifests — they may share pure low-level utilities, but no
   function that iterates hidden enemy state is reachable from the actor
   obs builder.
2. All actor spatial features are in a team-relative frame (Team A as-is,
   Team B mirrored across map center). The critic emits actor mirrors in
   the same team frame, but enemy-team blocks are emitted in WORLD frame
   so the critic sees absolute positions for both sides without a
   per-side mirror collapse.
"""

from __future__ import annotations

__all__ = [
    "ACTOR_PHASE1_DIM",
    "ACTOR_PHASE1_FIELDS",
    "CRITIC_DIM",
    "CRITIC_FIELDS",
    "actor_field_slice",
    "critic_field_slice",
]

# Each entry: (field_name, width_in_floats, short_description).
# Order here IS the on-tensor order — do not reorder without updating the
# C++ obs builder in the same commit.
ACTOR_PHASE1_FIELDS: tuple[tuple[str, int, str], ...] = (
    ("own_hp",                  1, "own HP / max, in [0, 1]"),
    ("own_velocity",            2, "own velocity in team-frame, [-1, 1]"),
    ("own_aim_unit",            2, "own aim direction as (sin, cos)"),
    ("own_position",            2, "own position in team-frame, normalized to map extent"),
    ("own_ammo",                1, "own revolver magazine / 6"),
    ("own_reloading",           1, "1 if currently reloading else 0"),
    ("own_combat_roll_cd",      1, "ability_1 cooldown ticks / max, in [0, 1]"),
    ("enemy_alive",             1, "1 if enemy is alive (no fog at Phase 1) else 0"),
    ("enemy_respawn_timer",     1, "enemy respawn ticks remaining / max, 0 when alive"),
    ("enemy_relative_position", 2, "enemy minus own position in team-frame"),
    ("enemy_hp",                1, "enemy HP / max; 0 if dead"),
    ("enemy_velocity",          2, "enemy velocity in team-frame; (0,0) if dead"),
    ("objective_owner_onehot",  3, "objective owner: {Neutral, Us, Them}"),
    ("cap_team_onehot",         3, "team currently accruing capture progress: {None, Us, Them}"),
    ("cap_progress",            1, "capture progress in [0, 1]"),
    ("contested",               1, "1 if both teams present on point"),
    ("objective_unlocked",      1, "1 if past the 15s unlock window"),
    ("own_score",               1, "own score ticks / win threshold"),
    ("enemy_score",             1, "enemy score ticks / win threshold"),
    ("self_on_point",           1, "1 if viewer is inside the objective circle"),
    ("enemy_on_point",          1, "1 if enemy is inside the objective circle (public, no fog at Phase 1)"),
    ("round_timer",             1, "sim ticks elapsed / round length ticks"),
)

ACTOR_PHASE1_DIM: int = sum(width for _, width, _ in ACTOR_PHASE1_FIELDS)


def _slot_prefixed_actor_fields(slot: int) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (f"slot{slot}/{name}", width, desc)
        for name, width, desc in ACTOR_PHASE1_FIELDS
    )


_ENEMY_WORLD_BLOCK: tuple[tuple[str, int, str], ...] = (
    ("world_position",   2, "world-frame position (no mirror)"),
    ("world_velocity",   2, "world-frame velocity (no mirror)"),
    ("world_aim_unit",   2, "world-frame aim as (sin, cos) of aim_angle"),
    ("hp_normalized",    1, "health_centi_hp / max_health_centi_hp"),
    ("alive_flag",       1, "1 if alive else 0"),
    ("respawn_timer",    1, "(respawn_tick - now) / max, 0 when alive"),
    ("ammo",             1, "weapon.magazine / kRangerMaxMagazine"),
    ("reloading",        1, "1 if reloading else 0"),
    ("combat_roll_cd",   1, "cd_ability_1 / max"),
)


def _enemy_block_for(enemy: int) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (f"enemy{enemy}/{name}", width, desc)
        for name, width, desc in _ENEMY_WORLD_BLOCK
    )


# Phase-4 critic obs layout. Mirrors C++ `kCriticObsDim` in obs.h.
CRITIC_FIELDS: tuple[tuple[str, int, str], ...] = (
    *_slot_prefixed_actor_fields(0),
    *_slot_prefixed_actor_fields(1),
    *_slot_prefixed_actor_fields(2),
    *_enemy_block_for(0),
    *_enemy_block_for(1),
    *_enemy_block_for(2),
    ("cap_progress_ticks", 1, "raw capture progress tick counter"),
    ("team_a_score_ticks", 1, "raw Team A score tick counter"),
    ("team_b_score_ticks", 1, "raw Team B score tick counter"),
    ("tick_raw",           1, "raw match tick counter"),
    ("seed_hi",            1, "top 32 bits of seed, normalized [-1, 1]"),
    ("seed_lo",            1, "bottom 32 bits of seed, normalized [-1, 1]"),
)

CRITIC_DIM: int = sum(width for _, width, _ in CRITIC_FIELDS)
assert CRITIC_DIM == 135, (
    f"CRITIC_DIM drifted to {CRITIC_DIM}; expected 135. "
    "Did the C++ kCriticObsDim get updated to match?"
)


def _build_slice_table(fields: tuple[tuple[str, int, str], ...]) -> dict[str, slice]:
    table: dict[str, slice] = {}
    cursor = 0
    for name, width, _ in fields:
        table[name] = slice(cursor, cursor + width)
        cursor += width
    return table


_ACTOR_SLICES: dict[str, slice] = _build_slice_table(ACTOR_PHASE1_FIELDS)
_CRITIC_SLICES: dict[str, slice] = _build_slice_table(CRITIC_FIELDS)


def actor_field_slice(name: str) -> slice:
    """Return the `slice` into the actor obs tensor for the named field."""
    try:
        return _ACTOR_SLICES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown actor obs field {name!r}; known: {sorted(_ACTOR_SLICES)}"
        ) from exc


def critic_field_slice(name: str) -> slice:
    """Return the `slice` into the critic obs tensor for the named field."""
    try:
        return _CRITIC_SLICES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown critic obs field {name!r}; known: {sorted(_CRITIC_SLICES)}"
        ) from exc
