# Phase 4 — Centralized Critic Observation: Design

**Date:** 2026-05-07
**Status:** Design approved. Ready for implementation.
**Slice:** Audit item 2 (foundation step) from `2026-04-24-phase4-prep.md`.

## Goals & Scope

Widen the C++ centralized-critic observation builder from its 1v1 (Phase-1)
shape to the 3v3 (Phase-4) shape required by MAPPO/CTDE. This is the
foundation slice in the Phase-4 implementation order — every downstream
Phase-4 component (3v3 env, vector env, rollout buffer, MAPPOTrainer) reads
from it.

### In scope

1. New C++ builder `build_critic_obs(sim, team_perspective, out, capacity)`
   emitting `kCriticObsDim = 135` floats.
2. Minimal sim-side prerequisite: `MatchConfig.team_size ∈ {1, 3}` plus a 3v3
   spawn path in `reset_state` and `respawn_tick_update`. `team_size=1` stays
   bit-identical to today.
3. Drop `kCriticObsPhase1Dim`, `build_critic_obs_phase1`, `CRITIC_PHASE1_*`.
   Single un-suffixed `build_critic_obs` / `kCriticObsDim` / `CRITIC_FIELDS`
   from this slice forward.
4. Python bindings, Python manifest, C++ tests, Python tests all updated to
   the new shape and dim.

### Out of scope (deferred to later Phase-4 slices)

- Any *consumption* of the critic obs: centralized critic forward pass,
  MAPPOTrainer, rollout buffer agent/team axes — Phase-4 audit items 5–7.
- The 3v3 env (`phase4_mappo.py`), `XushiVectorEnv`, opponent pool —
  audit items 3, 4, 10.
- Per-role fields for second heroes (Phase 10+).

### Success criteria

- Build green, all 114 currently-passing tests still pass.
- `kCriticObsDim == 135` and `CRITIC_DIM == 135` enforced at compile/import
  time on both sides.
- New tests verify the critic obs layout end-to-end against a real 3v3 sim
  state.

## Layout — `kCriticObsDim = 135`

```
[0  ..  31)   actor_obs(team_perspective, own slot 0)   — 31 floats, team-frame
[31 ..  62)   actor_obs(team_perspective, own slot 1)   — 31 floats, team-frame
[62 ..  93)   actor_obs(team_perspective, own slot 2)   — 31 floats, team-frame

[93 .. 105)   enemy_world_block(enemy slot 0)           — 12 floats, world-frame
[105.. 117)   enemy_world_block(enemy slot 1)           — 12 floats, world-frame
[117.. 129)   enemy_world_block(enemy slot 2)           — 12 floats, world-frame

[129..133)    cap_progress_ticks, team_a_score_ticks,
              team_b_score_ticks, tick                  — 4 floats, raw ints
[133..135)    seed_hi, seed_lo                          — 2 floats, normalized
```

**Per-enemy block (12 floats), all world-frame, no team mirroring:**

```
world_position    (2)
world_velocity    (2)
world_aim_unit    (2)   // (sin(aim_angle), cos(aim_angle))
hp_normalized     (1)   // health_centi_hp / max_health_centi_hp
alive_flag        (1)
respawn_timer     (1)   // (respawn_tick - now) / max, 0 when alive
ammo              (1)   // weapon.magazine / kRangerMaxMagazine
reloading         (1)
combat_roll_cd    (1)   // cd_ability_1 / max
```

### Layout rationale

The first 93 floats give the centralized critic each own-team agent's
exact actor surface. The CTDE "critic sees ≥ actor" contract is then
trivial to assert in tests (`critic[i*31:(i+1)*31] == actor_obs(slot_i)`).
The 36-float enemy block carries world-frame ground truth for the
opponent team, which the actor never sees (no enemy aim, ammo, reloading,
combat-roll CD on the actor side under fog rules — those would be hidden
state under Phase 7, so the critic must source them privileged).

The trailing 6 floats are unchanged from Phase-1: raw objective tick
counters and seed bits. They were already team-agnostic.

### Trade-offs accepted

- **~30 floats of redundancy** across the three actor mirrors (objective
  fields, score, round timer, etc. are agent-invariant). Cost is one
  extra cache line into a small MLP — negligible.
- **Slot-identity coupled to offset** (slot 0 always at byte 0). The sim
  does not permute slot identities, so this is theoretical drift — not a
  current correctness issue.
- **Mixed coordinate frames**: actor blocks are team-frame mirrored,
  privileged tail is world-frame. Critic must learn the mapping, exactly
  as it does today under Phase-1.

## Sim prerequisite — minimal 3v3 spawning

The sim already has `kTeamSize=3`, `kAgentsPerMatch=6`, and a 6-slot
`heroes` array. But `reset_state` only spawns slots 0 and 3 today — all
others stay `present=false`. Without this prerequisite the new critic
builder cannot be exercised end-to-end at 3v3.

### Changes

**`src/sim/include/xushi2/sim/sim.h`** — `MatchConfig` gains:

```cpp
std::uint32_t team_size = 1;  // 1 (Phase 1–3) or 3 (Phase 4+)
```

`Sim` ctor adds `team_size ∈ {1, 3}` validation alongside existing
config checks.

**`src/sim/src/internal/sim_spawn_reset.cpp`**:

- `reset_state` dispatches on `cfg.team_size`:
  - `1`: existing path unchanged. Slot 0 (Team A) and slot 3 (Team B) at
    today's spawn points. Bit-identical to today.
  - `3`: spawns slots 0,1,2 along the Team A spawn line and 3,4,5 along
    the Team B spawn line, x-offset by `dx = 0.15 * (max_x − min_x)` so
    they don't overlap. y-coordinates: 10% of `span_y` for Team A, 90% for
    Team B. Aim angles: `+π/2` for Team A, `−π/2` for Team B. Entity ids
    1..6 in slot order.
- `respawn_tick_update` extended to look up the per-slot spawn point for
  the team_size==3 path. team_size==1 path unchanged.

### Why bundle respawn now

Critic-obs tests don't trigger respawn, but leaving respawn 1v1-only
would be a latent crash for the next slice (the 3v3 env). Cost is small:
six hardcoded spawn points instead of two.

### Determinism baseline

`state_hash()` covers `present` and per-hero state. Existing determinism
golden hashes are pinned at `team_size=1` and remain valid since that
path is bit-identical. New determinism baselines for the 3v3 path land
in step 5 (tests) — they're separate goldens, not a rebaseline of the
existing ones.

## Implementation steps

Each step keeps the build green.

### 1. Sim 3v3 spawning

Add `MatchConfig.team_size`, dispatch in `reset_state`, extend
`respawn_tick_update`. Add a focused C++ test asserting:

- `team_size=3` produces 6 present heroes at 6 distinct positions,
  3 per team, all alive at full HP.
- `team_size=1` produces a byte-identical `MatchState` to before
  (compare via `state_hash()` against pinned baseline).

All 114 currently-passing tests still pass at this point.

### 2. Header + Python manifest contract (atomic commit)

**`src/sim/include/xushi2/sim/obs.h`**:
- Remove `kCriticObsPhase1Dim`, remove `build_critic_obs_phase1` decl.
- Add `inline constexpr std::uint32_t kCriticObsDim = 135;` and the new
  `build_critic_obs` decl. Update header doc-comment to describe the new
  layout: own-mirror × 3 → enemy-block × 3 → objective + seed. Retain
  the ban on calling from any actor obs path.

**`python/xushi2/obs_manifest.py`**:
- Drop `CRITIC_PHASE1_FIELDS`, `CRITIC_PHASE1_DIM`, the phase-1 critic
  slice table, and the `_phase1` references in module docstring.
- Add `CRITIC_FIELDS` and `CRITIC_DIM`. Field naming convention:
  `slotN/<actor_field>` for the actor mirrors (N ∈ 0,1,2),
  `enemyN/<field>` for the enemy world blocks (N ∈ 0,1,2), then the
  unprefixed objective + seed fields.
- Add `_assert_dim(CRITIC_FIELDS, expected=135)` at module load — fails
  loudly on any layout drift.
- Update `critic_field_slice` and `__all__`.

This commit will leave `critic_obs.cpp` not compiling until step 3 — that
is intentional. The two contracts (header + manifest) ship together.

### 3. New C++ builder

**`src/sim/src/critic_obs.cpp`** is rewritten:

- Replace `find_team_ranger_slot` (returns one) with
  `find_team_ranger_slots(state, team) -> std::array<std::uint32_t, 3>`
  returning the three slot indices in ascending order. Asserts all three
  are alive-or-respawning (i.e. `present`); if `present` ever flips off
  on death (verify against `sim_combat.cpp` during implementation) the
  assertion softens to slot-index range rather than presence.
- Body of `build_critic_obs(sim, team_perspective, out, capacity)`:
  1. Resolve `own_slots = find_team_ranger_slots(state, team_perspective)`.
  2. For each own slot, call
     `build_actor_obs_phase1(sim, slot, out + cursor, kActorObsPhase1Dim)`;
     advance cursor by 31. Total 93 floats.
  3. Resolve `enemy_slots = find_team_ranger_slots(state, opposite(team_perspective))`.
  4. For each enemy slot, emit a 12-float block via a private helper
     `emit_enemy_world_block(Writer&, HeroState const&, MatchConfig const&)`.
     World-frame, no mirror, regardless of team_perspective.
  5. Emit raw objective counters: `cap_progress_ticks`,
     `team_a_score_ticks`, `team_b_score_ticks`, `tick`. Same as Phase 1.
  6. Emit `norm_u32(seed_hi)`, `norm_u32(seed_lo)`. Same as Phase 1.
  7. `X2_ENSURE(cursor == kCriticObsDim, …)`.

### 4. Python binding

**`src/python_bindings/module.cpp`** (~lines 199–225):
- Update the `m.def("build_critic_obs", …)` lambda to call the new C++
  symbol and validate `out.shape(0) >= kCriticObsDim`. Update the
  docstring to point at the new layout.

### 5. Tests

**`tests/observations/test_critic_obs.cpp`**:
- All cases construct `MatchConfig` with `team_size = 3`.
- Drop the existing `PrefixMatchesActorObsForSameTeam` (slot-0-only).
  Replace with `PrefixMatchesActorObsForAllThreeOwnSlots` looping
  slots 0,1,2 and asserting each 31-float chunk equals
  `build_actor_obs_phase1(sim, slot, …)`. Run for both Team A and
  Team B perspectives.
- New `EnemyWorldBlockMatchesHeroStateRaw`: assert each enemy block
  matches `state.heroes[enemy_slot]` raw fields (position/velocity
  world, sin/cos of aim, hp, alive, respawn_timer, ammo, reloading,
  combat_roll_cd).
- Keep `RawTickCountersAtFreshStateAreZero`,
  `SeedBitsAreStableAcrossSteps`, `RawTickAdvancesAfterSteps` —
  update offsets: privileged-tail base for raw counters is now
  `93 + 36 = 129`; seed bits are at `133`/`134`.
- New `IdleSteps3v3DoesNotCrash`: 100 idle ticks on a `team_size=3`
  sim, no `X2_ENSURE` aborts.

**`tests/observations/test_obs_dims.cpp`**: assert
`kCriticObsDim == 135`.

**`python/tests/test_bindings_obs.py`**: bump expected dim to 135;
construct sim with `team_size=3`; smoke-test finiteness and shape.

**`python/tests/test_obs_manifest.py`**: assert `CRITIC_DIM == 135`;
assert C++ `kCriticObsDim` (via binding) equals Python `CRITIC_DIM`.

### 6. Docs

**`docs/observation_spec.md`**:
- Update §Phase 1 to note the 1v1 critic builder has been retired —
  `build_critic_obs` now requires `team_size=3`.
- Add §Phase 4 documenting the new critic layout: own-mirror × 3,
  enemy-block × 3, objective + seed. Field-by-field catalog matching
  `obs_manifest.py CRITIC_FIELDS`.

## Risks

1. **`present` semantics on death.** Current code in `sim_combat.cpp`
   appears to keep `present=true` and only flip `alive=false` so respawn
   can find the slot, but this needs verification in step 3. If wrong,
   `find_team_ranger_slots`' assertion must soften to slot-range only.
2. **Combat / objective / movement subsystems** loop over all 6 slots
   gated by `present`. With 3v3 spawning enabled, previously-empty slots
   become live. For this slice the only thing stepping the sim is
   critic-obs tests doing idle actions, which won't trigger combat — the
   `IdleSteps3v3DoesNotCrash` test is the cheap insurance against
   `X2_ENSURE` aborts in tick subsystems.
3. **Manifest/header drift.** The Python manifest and C++ header are
   both single-sources-of-truth for the same layout. The
   `_assert_dim(…, expected=135)` and `kCriticObsDim` constant are the
   guardrails; the cross-binding test in `test_obs_manifest.py` is the
   integration check. Any field reordering must touch both files in the
   same commit.

## Order of Phase-4 work after this slice

This design lands audit item 2. Remaining order from the audit:

3. 3v3 env (`python/envs/phase4_mappo.py`) with per-agent obs/action/
   reward and a `build_critic_obs` hook. Scripted opponents.
4. Custom `XushiVectorEnv` (sync-only) returning `(actor_obs,
   critic_obs, …)` tuple.
5. MAPPO rollout buffer with agent + team axes.
6. Split `models.py` into `Actor` + `Critic`.
7. `MAPPOTrainer`.
8. Wire into orchestration; per-agent greedy eval.
9. Smoke-train Phase-4 vs scripted bots.
10. Self-play opponent pool.
11. (Later) async vector env once rollout dominates wall clock.

The viewer/replay debug gate (audit's reordered item 1) lands in
parallel rather than as a hard prerequisite.
