# Phase 4 Critic Obs — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 1v1 (Phase-1) centralized-critic observation builder with a 3v3 (Phase-4) builder that emits 135 floats — 3 own-team actor mirrors + 3 per-enemy world blocks + objective + seed — and bundle the minimal `MatchConfig.team_size` plumbing the sim needs to be exercised at 3v3.

**Architecture:** First land the sim 3v3 spawn path (gated on a new `MatchConfig.team_size` field, default `1` so existing tests are bit-identical). Then atomically update `obs.h` and `obs_manifest.py` to the new contract — that intentionally breaks the C++ link until the new `build_critic_obs` body is written. Tests update last, asserting the new layout end-to-end against a real 3v3 sim.

**Tech Stack:** C++17 (sim, GoogleTest), CMake build, pybind11 Python bindings, Python 3.10+ (pytest, numpy).

**Reference:** All layout decisions and rationale live in `docs/plans/2026-05-07-phase4-critic-obs-design.md`. This plan executes that design — when in doubt about field order, frame conventions, or rationale, consult the design doc rather than re-deriving.

**Worktree:** `.worktrees/phase4-critic-obs` on branch `phase4-critic-obs`. All commands below are relative to that worktree root unless stated otherwise.

**Commit cadence:** Per user preference, do NOT commit per task. The user batches the whole delta at end-of-feature. Treat each task's "verify pass" step as the checkpoint.

---

## Task 0: Worktree baseline check

**Goal:** Confirm the worktree builds clean and the existing test suite passes before any change. Without this, mid-plan failures are ambiguous (new bug vs. pre-existing).

**Files:** none modified.

**Step 1: Configure CMake**

```powershell
cmake -S . -B build -DXUSHI2_BUILD_VIEWER=OFF -DXUSHI2_BUILD_TESTS=ON -DXUSHI2_BUILD_PYTHON_MODULE=ON
```

Expected: configures cleanly, generates `build/`.

**Step 2: Build C++ and Python extension**

```powershell
cmake --build build --parallel
```

Expected: builds without errors. Python extension `xushi2_cpp.*.pyd` lands somewhere under `python/xushi2/`.

**Step 3: Run C++ tests**

```powershell
ctest --test-dir build --output-on-failure
```

Expected: all C++ tests PASS. Note the count.

**Step 4: Run Python tests**

```powershell
cd python; python -m pytest tests -q
```

Expected: 114 tests PASS (per Phase 3 result writeup `c482235`).

**Step 5: If any baseline test fails**

Stop and report. Do NOT continue with implementation — fix the baseline first or ask the user.

---

## Task 1: Add `MatchConfig.team_size` field with validation

**Goal:** Introduce the 1-or-3 toggle field. Default `1` so all existing tests are unaffected. Sim ctor rejects out-of-range values.

**Files:**
- Modify: `src/sim/include/xushi2/sim/sim.h` (add `team_size` field to `MatchConfig`)
- Modify: `src/sim/src/sim.cpp` (add ctor validation alongside existing checks)
- Test: extend `tests/sim/test_sim_config.cpp` if it exists; otherwise add to `tests/observations/test_critic_obs.cpp` as a smoke case (preferable: dedicated config-validation test file already exists or will be obvious from `tests/sim/`)

**Step 1: Locate the existing config-validation tests**

```powershell
Get-ChildItem -Recurse -File tests | Select-String -Pattern "MatchConfig|kMaxRoundLength|invalid.*config" -List | Select-Object -First 5
```

Use the first hit's file as the home for the new test. If no obvious file, create `tests/sim/test_match_config.cpp`.

**Step 2: Write failing tests**

```cpp
// In the chosen test file:
TEST(MatchConfig, DefaultTeamSizeIsOne) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    EXPECT_EQ(cfg.team_size, 1u);
}

TEST(MatchConfig, TeamSizeThreeIsAccepted) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    EXPECT_NO_THROW(Sim sim(cfg));
}

TEST(MatchConfig, TeamSizeTwoIsRejected) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 2;
    EXPECT_ANY_THROW(Sim sim(cfg));
}
```

**Step 3: Run; verify FAIL**

```powershell
cmake --build build --parallel; ctest --test-dir build -R MatchConfig --output-on-failure
```

Expected: compile fails (no `team_size` field) or tests fail.

**Step 4: Add the field and validation**

In `src/sim/include/xushi2/sim/sim.h`, inside `struct MatchConfig`:

```cpp
// Phase-4 toggle. team_size==1: single Ranger per team (Phase 1–3 path,
// slots 0 and 3). team_size==3: full 3v3 (Phase 4+, slots 0–2 + 3–5).
// Other values are rejected by the Sim ctor.
std::uint32_t team_size = 1;
```

In `src/sim/src/sim.cpp`, inside the existing config-validation block in `Sim::Sim`:

```cpp
X2_REQUIRE(config.team_size == 1 || config.team_size == 3,
           common::ErrorCode::CorruptState);
```

**Step 5: Run; verify PASS**

```powershell
cmake --build build --parallel; ctest --test-dir build -R MatchConfig --output-on-failure
```

Expected: all 3 new MatchConfig tests PASS. Run the full C++ suite to confirm no regressions:

```powershell
ctest --test-dir build --output-on-failure
```

Expected: all previously-passing tests still PASS.

---

## Task 2: 3v3 spawning in `reset_state`

**Goal:** When `team_size == 3`, populate slots 0,1,2 (Team A) and 3,4,5 (Team B) at six distinct positions. When `team_size == 1`, behave bit-identical to today.

**Files:**
- Modify: `src/sim/src/internal/sim_spawn_reset.cpp` (`reset_state` function)
- Test: same file as Task 1's tests

**Step 1: Write failing tests**

```cpp
TEST(Spawn3v3, ProducesSixPresentHeroesAtDistinctPositions) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    Sim sim(cfg);
    const auto& heroes = sim.state().heroes;

    // All 6 slots present and alive.
    for (std::uint32_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(heroes[i].present) << "slot " << i;
        EXPECT_TRUE(heroes[i].alive) << "slot " << i;
    }

    // Slots 0–2 are Team A, 3–5 are Team B.
    for (std::uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(heroes[i].team, common::Team::A) << "slot " << i;
    }
    for (std::uint32_t i = 3; i < 6; ++i) {
        EXPECT_EQ(heroes[i].team, common::Team::B) << "slot " << i;
    }

    // No two heroes share a position.
    for (std::uint32_t i = 0; i < 6; ++i) {
        for (std::uint32_t j = i + 1; j < 6; ++j) {
            const float dx = heroes[i].position.x - heroes[j].position.x;
            const float dy = heroes[i].position.y - heroes[j].position.y;
            EXPECT_GT(dx*dx + dy*dy, 1e-3F) << "slots " << i << " and " << j;
        }
    }
}

TEST(Spawn3v3, TeamSizeOnePathIsUnchanged) {
    // Pin: with team_size=1 (default), slot 0 is Team A and slot 3 is Team B,
    // both at fixed Phase-1 spawn points; other slots are absent.
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto& heroes = sim.state().heroes;

    EXPECT_TRUE(heroes[0].present);
    EXPECT_TRUE(heroes[3].present);
    for (std::uint32_t i : {1u, 2u, 4u, 5u}) {
        EXPECT_FALSE(heroes[i].present) << "slot " << i;
    }
}
```

**Step 2: Run; verify FAIL**

```powershell
cmake --build build --parallel; ctest --test-dir build -R Spawn3v3 --output-on-failure
```

Expected: `ProducesSixPresentHeroesAtDistinctPositions` fails (slots 1,2,4,5 not present); `TeamSizeOnePathIsUnchanged` passes (no behavior change yet).

**Step 3: Implement 3v3 branch in `reset_state`**

In `src/sim/src/internal/sim_spawn_reset.cpp`, replace the body of `reset_state` after the `state.objective.*` initialization with:

```cpp
const float cx = 0.5F * (config.map.min_x + config.map.max_x);
const float span_y = config.map.max_y - config.map.min_y;
const float team_a_y = config.map.min_y + 0.1F * span_y;
const float team_b_y = config.map.min_y + 0.9F * span_y;

if (config.team_size == 1) {
    // Phase 1–3 path: one Ranger per team, slots 0 and 3.
    spawn_ranger(state.heroes[0], common::Team::A, 1,
                 common::Vec2{cx, team_a_y}, 0.5F * common::kPi);
    spawn_ranger(state.heroes[3], common::Team::B, 2,
                 common::Vec2{cx, team_b_y}, -0.5F * common::kPi);
} else {
    // Phase 4 path: 3v3, slots 0–2 (A) and 3–5 (B), x-offset by ±dx.
    const float dx = 0.15F * (config.map.max_x - config.map.min_x);
    const float xs[3] = {cx - dx, cx, cx + dx};
    for (std::uint32_t i = 0; i < 3; ++i) {
        spawn_ranger(state.heroes[i], common::Team::A,
                     static_cast<common::EntityId>(i + 1),
                     common::Vec2{xs[i], team_a_y},
                     0.5F * common::kPi);
        spawn_ranger(state.heroes[3 + i], common::Team::B,
                     static_cast<common::EntityId>(4 + i),
                     common::Vec2{xs[i], team_b_y},
                     -0.5F * common::kPi);
    }
}
```

**Step 4: Run; verify PASS**

```powershell
cmake --build build --parallel; ctest --test-dir build -R Spawn3v3 --output-on-failure
```

Expected: both Spawn3v3 tests PASS.

**Step 5: Verify no regressions in 1v1 path**

```powershell
ctest --test-dir build --output-on-failure
```

Expected: full suite passes. The `TeamSizeOnePathIsUnchanged` test plus existing determinism tests cover the 1v1 byte-identical claim.

---

## Task 3: Respawn at correct slot for 3v3

**Goal:** When a hero dies and respawns under `team_size == 3`, return them to their own slot's spawn point — not slot 0/3's. Cheap insurance against latent crashes in the next slice.

**Files:**
- Modify: `src/sim/src/internal/sim_spawn_reset.cpp` (`respawn_tick_update`)
- Test: same file as prior tasks

**Step 1: Write failing test**

```cpp
TEST(Spawn3v3, RespawnReturnsToOwnSlotPoint) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    Sim sim(cfg);

    // Capture each Team-A slot's original spawn position.
    std::array<common::Vec2, 3> origins;
    for (std::uint32_t i = 0; i < 3; ++i) {
        origins[i] = sim.state().heroes[i].position;
    }

    // Force-kill slot 1, advance enough ticks to respawn, check position.
    // (Use a test-support helper if one exists; otherwise zero HP and
    // step until alive again. The respawn window is config.respawn_ticks.)
    auto& mutable_h = const_cast<HeroState&>(sim.state().heroes[1]);
    mutable_h.alive = false;
    mutable_h.health_centi_hp = 0;
    mutable_h.respawn_tick = sim.state().tick + cfg.respawn_ticks;

    std::array<Action, kAgentsPerMatch> idle{};
    for (std::uint32_t k = 0; k < cfg.respawn_ticks + 4; ++k) {
        sim.step_decision(idle);
        if (sim.state().heroes[1].alive) break;
    }

    ASSERT_TRUE(sim.state().heroes[1].alive);
    EXPECT_NEAR(sim.state().heroes[1].position.x, origins[1].x, 1e-4F);
    EXPECT_NEAR(sim.state().heroes[1].position.y, origins[1].y, 1e-4F);
}
```

NOTE: if `tests/observations/...` cannot easily mutate `HeroState`, move the test to a sim-internal test directory (e.g. `tests/sim/test_respawn.cpp`) which already does this kind of mutation. Look for existing patterns:

```powershell
Get-ChildItem -Recurse -File tests | Select-String -Pattern "respawn|alive\s*=\s*false" -List | Select-Object -First 5
```

**Step 2: Run; verify FAIL**

```powershell
cmake --build build --parallel; ctest --test-dir build -R RespawnReturnsToOwnSlot --output-on-failure
```

Expected: hero respawns at slot-0's position (cx, team_a_y), not slot-1's (cx − dx, team_a_y).

**Step 3: Implement 3v3-aware respawn**

In `src/sim/src/internal/sim_spawn_reset.cpp`, replace the body of `respawn_tick_update` (after the early returns) with code that picks a per-slot spawn point. The function only receives `HeroState& h` and `MatchConfig const& config`, not the slot index — so add a slot-index parameter and update the call site (which is in `sim.cpp` or a tick-pipeline file; find it via `Get-ChildItem ... | Select-String "respawn_tick_update"`).

```cpp
void respawn_tick_update(HeroState& h, std::uint32_t slot,
                         common::Tick now, const MatchConfig& config) {
    if (h.alive || !h.present) return;
    if (now < h.respawn_tick) return;

    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;
    const float team_a_y = config.map.min_y + 0.1F * span_y;
    const float team_b_y = config.map.min_y + 0.9F * span_y;

    common::Vec2 spawn_pos{};
    float aim_angle = 0.0F;
    if (config.team_size == 1) {
        // Phase 1–3: slot 0 and slot 3 at the canonical points.
        if (h.team == common::Team::A) {
            spawn_pos = common::Vec2{cx, team_a_y};
            aim_angle = 0.5F * common::kPi;
        } else {
            spawn_pos = common::Vec2{cx, team_b_y};
            aim_angle = -0.5F * common::kPi;
        }
    } else {
        // Phase 4: per-slot offset along x.
        const float dx = 0.15F * (config.map.max_x - config.map.min_x);
        const std::uint32_t local = (h.team == common::Team::A) ? slot : (slot - 3);
        const float xs[3] = {cx - dx, cx, cx + dx};
        spawn_pos = common::Vec2{xs[local], (h.team == common::Team::A) ? team_a_y : team_b_y};
        aim_angle = (h.team == common::Team::A) ? 0.5F * common::kPi : -0.5F * common::kPi;
    }

    const std::uint32_t preserved_kills = h.kills;
    const std::uint32_t preserved_deaths = h.deaths;
    spawn_ranger(h, h.team, h.id, spawn_pos, aim_angle);
    h.kills = preserved_kills;
    h.deaths = preserved_deaths;
}
```

Update the header `src/sim/src/internal/sim_spawn_reset.h` and the call site to pass the slot index (the call site loops over heroes, so the index is already in scope).

**Step 4: Run; verify PASS**

```powershell
cmake --build build --parallel; ctest --test-dir build -R RespawnReturnsToOwnSlot --output-on-failure
```

Expected: PASS. Then full suite:

```powershell
ctest --test-dir build --output-on-failure
```

Expected: full PASS.

---

## Task 4: Replace Python critic manifest

**Goal:** Drop `CRITIC_PHASE1_*`. Add `CRITIC_FIELDS`, `CRITIC_DIM`. Field naming: `slotN/<actor_field>` for the three actor mirrors, `enemyN/<field>` for the enemy world blocks, then bare names for objective + seed. Assert `CRITIC_DIM == 135` at module load.

**Files:**
- Modify: `python/xushi2/obs_manifest.py`
- Test: `python/tests/test_obs_manifest.py`

**Step 1: Write failing test**

In `python/tests/test_obs_manifest.py`, add (or replace):

```python
def test_critic_dim_is_135():
    from xushi2.obs_manifest import CRITIC_DIM
    assert CRITIC_DIM == 135

def test_critic_fields_have_expected_layout():
    from xushi2.obs_manifest import CRITIC_FIELDS, ACTOR_PHASE1_FIELDS

    # First 3*len(actor_fields) are slotN/<actor_field> for N in 0,1,2.
    actor_names = [name for name, _, _ in ACTOR_PHASE1_FIELDS]
    expected_prefix = []
    for slot in range(3):
        for name in actor_names:
            expected_prefix.append(f"slot{slot}/{name}")

    actual_names = [name for name, _, _ in CRITIC_FIELDS]
    assert actual_names[: len(expected_prefix)] == expected_prefix

    # Then enemyN/<world block> for N in 0,1,2 (12 fields each).
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
```

Drop any existing `test_critic_phase1_*` tests in this file — they reference symbols we're deleting.

**Step 2: Run; verify FAIL**

```powershell
cd python; python -m pytest tests/test_obs_manifest.py -v
```

Expected: import errors / attribute errors / assertion failures on the new tests.

**Step 3: Update `python/xushi2/obs_manifest.py`**

Replace the existing `CRITIC_PHASE1_FIELDS`, `CRITIC_PHASE1_DIM`, `_CRITIC_SLICES` block with:

```python
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

_CRITIC_SLICES: dict[str, slice] = _build_slice_table(CRITIC_FIELDS)
```

Update `__all__` to drop `CRITIC_PHASE1_FIELDS`, `CRITIC_PHASE1_DIM` and add `CRITIC_FIELDS`, `CRITIC_DIM`. Update module docstring to reference Phase 4 layout.

**Step 4: Run; verify PASS**

```powershell
cd python; python -m pytest tests/test_obs_manifest.py -v
```

Expected: 3 new tests PASS. Note: `test_bindings_obs.py` may be temporarily broken since the C++ binding still expects the old size — that's handled in Task 7.

---

## Task 5: Update C++ obs.h header (intentionally breaks link)

**Goal:** Atomic swap of the C++ contract: drop `kCriticObsPhase1Dim` and `build_critic_obs_phase1`; add `kCriticObsDim = 135` and the new `build_critic_obs` decl. The implementation file (Task 6) will catch up immediately.

**Files:**
- Modify: `src/sim/include/xushi2/sim/obs.h`

**Step 1: Edit `src/sim/include/xushi2/sim/obs.h`**

Replace lines 30–34 (the `kCriticObsPhase1Dim` constant and its surrounding comment):

```cpp
// Phase-4 critic obs tensor width. Mirrors CRITIC_DIM in
// python/xushi2/obs_manifest.py. Layout: 3 own-team actor mirrors
// (3 × kActorObsPhase1Dim = 93 floats) + 3 enemy world blocks
// (12 floats each = 36) + 4 raw objective counters + 2 seed bits = 135.
inline constexpr std::uint32_t kCriticObsDim = 135;
```

Replace lines 54–71 (the `build_critic_obs_phase1` declaration and its contract comment):

```cpp
// Build the Phase-4 centralized-critic observation for the given team
// perspective into the caller-provided `out_buffer` of capacity
// `out_capacity` float32 entries.
//
// Contract:
//  - Writes exactly `kCriticObsDim` floats starting at out_buffer[0].
//  - Requires `out_capacity >= kCriticObsDim`; aborts otherwise.
//  - Requires the `Sim` to have been constructed with
//    `MatchConfig::team_size == 3`. With team_size == 1 the builder
//    will assert because the team has fewer than 3 present Rangers.
//  - `team_perspective` must be `Team::A` or `Team::B`.
//  - Layout: for the team's three Ranger slots in ascending order,
//    emit the 31-float actor obs of each (team-frame). Then for the
//    enemy team's three slots, emit a 12-float world-frame block
//    (position, velocity, aim_unit, hp, alive, respawn_timer, ammo,
//    reloading, combat_roll_cd). Then 4 raw objective tick counters
//    and 2 normalized seed bits. See python/xushi2/obs_manifest.py
//    CRITIC_FIELDS for the field-by-field catalog.
//  - The critic may iterate full sim state. It MUST NOT be called
//    from any code path that builds an actor obs.
void build_critic_obs(const Sim& sim,
                      common::Team team_perspective,
                      float* out_buffer,
                      std::uint32_t out_capacity) noexcept;
```

**Step 2: Attempt build; expect link/compile failure**

```powershell
cmake --build build --parallel
```

Expected: `critic_obs.cpp` fails to compile (still defines `build_critic_obs_phase1`, references `kCriticObsPhase1Dim`). This is intentional — Task 6 lands the matching implementation. Do not commit between Task 5 and Task 6.

---

## Task 6: Implement new `build_critic_obs` in `critic_obs.cpp`

**Goal:** Replace the file's body with the 3v3 builder.

**Files:**
- Modify: `src/sim/src/critic_obs.cpp`

**Step 1: Rewrite `critic_obs.cpp`**

Replace the entire file body (after the `#include`s) with:

```cpp
namespace xushi2::sim {

namespace {

struct Writer {
    float* out;
    std::uint32_t cursor;
    void push1(float v) noexcept {
        X2_ENSURE(std::isfinite(v), common::ErrorCode::NonFiniteFloat);
        out[cursor++] = v;
    }
    void push2(float a, float b) noexcept { push1(a); push1(b); }
};

float norm_u32(std::uint32_t v) noexcept {
    constexpr float kTwoPow32 = 4294967296.0F;
    return 2.0F * (static_cast<float>(v) / kTwoPow32) - 1.0F;
}

std::array<std::uint32_t, 3> find_team_ranger_slots(
        const MatchState& s, common::Team team) noexcept {
    std::array<std::uint32_t, 3> slots{
        static_cast<std::uint32_t>(s.heroes.size()),
        static_cast<std::uint32_t>(s.heroes.size()),
        static_cast<std::uint32_t>(s.heroes.size()),
    };
    std::uint32_t found = 0;
    for (std::uint32_t i = 0; i < s.heroes.size() && found < 3; ++i) {
        const auto& h = s.heroes[i];
        if (h.present && h.team == team) {
            slots[found++] = i;
        }
    }
    X2_REQUIRE(found == 3, common::ErrorCode::InvalidHeroId);
    return slots;
}

void emit_enemy_world_block(Writer& w,
                            const HeroState& h,
                            const MatchConfig& cfg) noexcept {
    // World-frame position and velocity, no mirror.
    w.push2(h.position.x, h.position.y);
    w.push2(h.velocity.x, h.velocity.y);

    // World-frame aim as (sin, cos) — no mirror, raw aim_angle.
    w.push2(std::sin(h.aim_angle), std::cos(h.aim_angle));

    // hp_normalized.
    const float hp = (h.max_health_centi_hp > 0)
        ? (static_cast<float>(h.health_centi_hp) /
           static_cast<float>(h.max_health_centi_hp))
        : 0.0F;
    w.push1(hp);

    // alive_flag.
    w.push1(h.alive ? 1.0F : 0.0F);

    // respawn_timer normalized to [0, 1] using cfg.respawn_ticks.
    float respawn_norm = 0.0F;
    if (!h.alive && cfg.respawn_ticks > 0) {
        // Mirrors the actor's enemy_respawn_timer convention.
        respawn_norm = static_cast<float>(h.respawn_tick) /
                       static_cast<float>(cfg.respawn_ticks);
        if (respawn_norm > 1.0F) respawn_norm = 1.0F;
        if (respawn_norm < 0.0F) respawn_norm = 0.0F;
    }
    w.push1(respawn_norm);

    // ammo.
    const float ammo = (common::kRangerMaxMagazine > 0)
        ? (static_cast<float>(h.weapon.magazine) /
           static_cast<float>(common::kRangerMaxMagazine))
        : 0.0F;
    w.push1(ammo);

    // reloading: same convention the actor builder uses.
    // (Inspect actor_obs.cpp for the exact predicate; mirror it here.)
    w.push1(h.weapon.reload_remaining_ticks > 0 ? 1.0F : 0.0F);

    // combat_roll_cd normalized.
    const float roll_cd = (common::kRangerCombatRollCooldownTicks > 0)
        ? (static_cast<float>(h.cd_ability_1) /
           static_cast<float>(common::kRangerCombatRollCooldownTicks))
        : 0.0F;
    w.push1(roll_cd);
}

}  // namespace

void build_critic_obs(const Sim& sim,
                      common::Team team_perspective,
                      float* out_buffer,
                      std::uint32_t out_capacity) noexcept {
    X2_REQUIRE(out_buffer != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(out_capacity >= kCriticObsDim,
               common::ErrorCode::CapacityExceeded);
    X2_REQUIRE(team_perspective == common::Team::A ||
                   team_perspective == common::Team::B,
               common::ErrorCode::InvalidHeroId);

    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();
    X2_REQUIRE(cfg.team_size == 3, common::ErrorCode::CorruptState);

    // 1) Three own-team actor mirrors, each kActorObsPhase1Dim = 31 floats.
    const auto own_slots = find_team_ranger_slots(s, team_perspective);
    for (std::uint32_t i = 0; i < 3; ++i) {
        build_actor_obs_phase1(sim, own_slots[i],
                               out_buffer + i * kActorObsPhase1Dim,
                               kActorObsPhase1Dim);
    }

    Writer w{out_buffer, 3 * kActorObsPhase1Dim};

    // 2) Three enemy-team world blocks, 12 floats each.
    const common::Team enemy_team =
        (team_perspective == common::Team::A) ? common::Team::B
                                              : common::Team::A;
    const auto enemy_slots = find_team_ranger_slots(s, enemy_team);
    for (std::uint32_t i = 0; i < 3; ++i) {
        emit_enemy_world_block(w, s.heroes[enemy_slots[i]], cfg);
    }

    // 3) Raw objective counters.
    w.push1(static_cast<float>(s.objective.cap_progress_ticks));
    w.push1(static_cast<float>(s.objective.team_a_score_ticks));
    w.push1(static_cast<float>(s.objective.team_b_score_ticks));
    w.push1(static_cast<float>(s.tick));

    // 4) Seed bits.
    const std::uint64_t seed = cfg.seed;
    const std::uint32_t seed_hi =
        static_cast<std::uint32_t>((seed >> 32) & 0xFFFFFFFFULL);
    const std::uint32_t seed_lo =
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL);
    w.push1(norm_u32(seed_hi));
    w.push1(norm_u32(seed_lo));

    X2_ENSURE(w.cursor == kCriticObsDim,
              common::ErrorCode::CapacityExceeded);
}

}  // namespace xushi2::sim
```

**NOTE on `weapon.reload_remaining_ticks` and the reloading predicate:** open `src/sim/src/actor_obs.cpp` and copy the *exact* predicate the actor builder uses for `own_reloading`. The placeholder above (`reload_remaining_ticks > 0`) may or may not match; mismatched predicates would fail Task 8's `EnemyWorldBlockMatchesHeroStateRaw` test. Verify before moving on.

**NOTE on `kRangerCombatRollCooldownTicks`:** confirm this constant name exists in `src/common/include/xushi2/common/limits.hpp` (or equivalent). The actor builder normalizes `cd_ability_1` against some constant — find that constant, mirror it here.

**Step 2: Build**

```powershell
cmake --build build --parallel
```

Expected: builds clean.

**Step 3: Run existing critic obs tests**

```powershell
ctest --test-dir build -R CriticObs --output-on-failure
```

Expected: most existing tests fail because the layout changed. Collect the failures — they're all expected and will be replaced in Task 8. Do NOT delete or "fix" them yet.

---

## Task 7: Update Python binding for new dim

**Goal:** Bring `m.def("build_critic_obs", …)` in line with the new `kCriticObsDim`.

**Files:**
- Modify: `src/python_bindings/module.cpp` (lines ~199–225)

**Step 1: Edit the binding lambda**

Replace the body of the `build_critic_obs` `m.def`:

```cpp
m.def(
    "build_critic_obs",
    [](const xushi2::sim::Sim& sim,
       xushi2::common::Team team_perspective,
       py::array_t<float, py::array::c_style | py::array::forcecast> out) {
        if (out.ndim() != 1) {
            throw std::invalid_argument("out must be a 1D float32 array");
        }
        if (static_cast<std::uint32_t>(out.shape(0)) <
            xushi2::sim::kCriticObsDim) {
            throw std::invalid_argument(
                "out has insufficient capacity for kCriticObsDim");
        }
        if (team_perspective != xushi2::common::Team::A &&
            team_perspective != xushi2::common::Team::B) {
            throw std::invalid_argument(
                "team_perspective must be Team.A or Team.B "
                "(Team.Neutral is not a valid critic side)");
        }
        xushi2::sim::build_critic_obs(
            sim, team_perspective, out.mutable_data(0),
            static_cast<std::uint32_t>(out.shape(0)));
    },
    py::arg("sim"), py::arg("team_perspective"), py::arg("out"),
    "Build the Phase-4 critic obs (kCriticObsDim floats) into out.");
```

Also expose the new dim constant if not already: search for `kCriticObsPhase1Dim` in `module.cpp` and replace with `kCriticObsDim`.

**Step 2: Build**

```powershell
cmake --build build --parallel
```

Expected: builds clean.

**Step 3: Run the bindings smoke test (will likely fail until Task 9)**

```powershell
cd python; python -m pytest tests/test_bindings_obs.py -v
```

Expected: failures because the test still expects `CRITIC_PHASE1_DIM = 45`. That's fixed in Task 9.

---

## Task 8: Rewrite C++ critic obs tests for new layout

**Goal:** Replace `tests/observations/test_critic_obs.cpp` cases to match the new layout. All cases construct `MatchConfig` with `team_size = 3`.

**Files:**
- Modify: `tests/observations/test_critic_obs.cpp`
- Modify: `tests/observations/test_obs_dims.cpp` (bump dim assert to 135)

**Step 1: Update `test_obs_dims.cpp`**

Replace any `EXPECT_EQ(kCriticObsPhase1Dim, 45)` with `EXPECT_EQ(kCriticObsDim, 135)`. Drop references to `kCriticObsPhase1Dim`.

**Step 2: Rewrite `test_critic_obs.cpp`**

Replace the file with:

```cpp
#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <xushi2/sim/obs.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

namespace {

using xushi2::common::Action;
using xushi2::common::Team;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::kCriticObsDim;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

constexpr float kEps = 1e-5F;

MatchConfig make_3v3_config() {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    return cfg;
}

std::array<float, kCriticObsDim> build_critic(const Sim& sim, Team team) {
    std::array<float, kCriticObsDim> out{};
    xushi2::sim::build_critic_obs(
        sim, team, out.data(), static_cast<std::uint32_t>(out.size()));
    return out;
}

TEST(CriticObs, ShapeAndAllFinite) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

TEST(CriticObs, PrefixMatchesActorObsForAllThreeOwnSlotsTeamA) {
    Sim sim(make_3v3_config());
    auto critic = build_critic(sim, Team::A);
    for (std::uint32_t slot = 0; slot < 3; ++slot) {
        std::array<float, kActorObsPhase1Dim> actor{};
        xushi2::sim::build_actor_obs_phase1(
            sim, slot, actor.data(),
            static_cast<std::uint32_t>(actor.size()));
        for (std::uint32_t i = 0; i < kActorObsPhase1Dim; ++i) {
            EXPECT_FLOAT_EQ(critic[slot * kActorObsPhase1Dim + i], actor[i])
                << "slot " << slot << " field " << i;
        }
    }
}

TEST(CriticObs, PrefixMatchesActorObsForAllThreeOwnSlotsTeamB) {
    Sim sim(make_3v3_config());
    auto critic = build_critic(sim, Team::B);
    const std::uint32_t team_b_slots[3] = {3, 4, 5};
    for (std::uint32_t i = 0; i < 3; ++i) {
        std::array<float, kActorObsPhase1Dim> actor{};
        xushi2::sim::build_actor_obs_phase1(
            sim, team_b_slots[i], actor.data(),
            static_cast<std::uint32_t>(actor.size()));
        for (std::uint32_t j = 0; j < kActorObsPhase1Dim; ++j) {
            EXPECT_FLOAT_EQ(critic[i * kActorObsPhase1Dim + j], actor[j])
                << "team B slot index " << i << " field " << j;
        }
    }
}

TEST(CriticObs, EnemyWorldBlockMatchesHeroStateRawTeamA) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3 * kActorObsPhase1Dim;  // = 93
    const std::uint32_t enemy_slots[3] = {3, 4, 5};
    for (std::uint32_t i = 0; i < 3; ++i) {
        const auto& h = sim.state().heroes[enemy_slots[i]];
        const std::uint32_t off = base + i * 12;
        EXPECT_NEAR(obs[off + 0], h.position.x, kEps);
        EXPECT_NEAR(obs[off + 1], h.position.y, kEps);
        EXPECT_NEAR(obs[off + 2], h.velocity.x, kEps);
        EXPECT_NEAR(obs[off + 3], h.velocity.y, kEps);
        EXPECT_NEAR(obs[off + 4], std::sin(h.aim_angle), kEps);
        EXPECT_NEAR(obs[off + 5], std::cos(h.aim_angle), kEps);
        // hp at off+6, alive at off+7, respawn at off+8, ammo at off+9,
        // reloading at off+10, roll_cd at off+11. Spot-check hp + alive.
        EXPECT_FLOAT_EQ(obs[off + 7], h.alive ? 1.0F : 0.0F);
    }
}

TEST(CriticObs, RawTickCountersAtFreshStateAreZero) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3 * kActorObsPhase1Dim + 3 * 12;  // = 129
    EXPECT_NEAR(obs[base + 0], 0.0F, kEps);  // cap_progress_ticks
    EXPECT_NEAR(obs[base + 1], 0.0F, kEps);  // team_a_score_ticks
    EXPECT_NEAR(obs[base + 2], 0.0F, kEps);  // team_b_score_ticks
    EXPECT_NEAR(obs[base + 3], 0.0F, kEps);  // tick_raw
}

TEST(CriticObs, SeedBitsAreStableAcrossSteps) {
    MatchConfig cfg = make_3v3_config();
    cfg.seed = 0xD1CEDA7A0BADF00DULL;
    Sim sim(cfg);
    auto a = build_critic(sim, Team::A);
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step_decision(idle);
    sim.step_decision(idle);
    auto b = build_critic(sim, Team::A);
    EXPECT_FLOAT_EQ(a[kCriticObsDim - 2], b[kCriticObsDim - 2]);
    EXPECT_FLOAT_EQ(a[kCriticObsDim - 1], b[kCriticObsDim - 1]);
}

TEST(CriticObs, RawTickAdvancesAfterSteps) {
    Sim sim(make_3v3_config());
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step_decision(idle);
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3 * kActorObsPhase1Dim + 3 * 12;
    EXPECT_GT(obs[base + 3], 0.0F);  // tick_raw advanced
}

TEST(CriticObs, IdleSteps3v3DoesNotCrash) {
    Sim sim(make_3v3_config());
    std::array<Action, kAgentsPerMatch> idle{};
    for (std::uint32_t k = 0; k < 100; ++k) {
        sim.step_decision(idle);
    }
    auto obs = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

}  // namespace
```

**Step 3: Build + run**

```powershell
cmake --build build --parallel; ctest --test-dir build -R CriticObs --output-on-failure
```

Expected: all critic obs tests PASS.

**Step 4: Run the full C++ suite**

```powershell
ctest --test-dir build --output-on-failure
```

Expected: full PASS, including all original tests.

---

## Task 9: Update Python binding tests

**Files:**
- Modify: `python/tests/test_bindings_obs.py`

**Step 1: Edit test file**

Find the existing `test_critic_obs_*` cases. Update:
- Replace `CRITIC_PHASE1_DIM` references with `CRITIC_DIM` from `xushi2.obs_manifest`.
- Construct `Sim` via `MatchConfig(team_size=3, …)` (or whatever the binding exposes — search the binding for how `MatchConfig` is constructed in tests).
- Buffer size: `np.zeros(135, dtype=np.float32)` (or use `CRITIC_DIM`).

The shape of the test should be:
1. Build a 3v3 sim.
2. Allocate buffer of size `CRITIC_DIM`.
3. Call `_cpp.build_critic_obs(sim, _cpp.Team.A, buf)`.
4. Assert `np.all(np.isfinite(buf))` and `buf.shape == (CRITIC_DIM,)`.
5. Assert that `team_perspective == Team.Neutral` raises.
6. Assert that an undersized buffer raises.

**Step 2: Run**

```powershell
cd python; python -m pytest tests/test_bindings_obs.py -v
```

Expected: PASS.

---

## Task 10: Final verification — full test sweep

**Files:** none modified.

**Step 1: Full C++ test suite**

```powershell
cmake --build build --parallel; ctest --test-dir build --output-on-failure
```

Expected: full PASS, count ≥ baseline from Task 0.

**Step 2: Full Python test suite**

```powershell
cd python; python -m pytest tests -q
```

Expected: full PASS, 114 + however many tests we added.

**Step 3: Spot-check the cross-language dim invariant**

```powershell
cd python; python -c "from xushi2.obs_manifest import CRITIC_DIM; from xushi2 import _cpp; import numpy as np; assert CRITIC_DIM == 135; print('OK CRITIC_DIM=', CRITIC_DIM)"
```

Expected: `OK CRITIC_DIM= 135`.

---

## Task 11: Update `docs/observation_spec.md`

**Files:**
- Modify: `docs/observation_spec.md`

**Step 1: Read the current §Phase 1 section**

Identify where the Phase-1 critic obs is documented. Note its structure (the prose pattern, the field-by-field catalog).

**Step 2: Add §Phase 4 section**

After §Phase 1, add a new §Phase 4 documenting the new critic layout. Include:
- A short prose description: 3 own-team actor mirrors → 3 enemy world blocks → objective + seed.
- A field-by-field catalog matching `obs_manifest.py CRITIC_FIELDS`.
- A note on coordinate frames: actor mirrors are team-frame, enemy blocks are world-frame.
- A pointer to `docs/plans/2026-05-07-phase4-critic-obs-design.md` for layout rationale.

**Step 3: Update §Phase 1 critic note**

Add: "The 1v1 critic builder (`build_critic_obs_phase1`) has been retired. `build_critic_obs` now requires `MatchConfig::team_size == 3`. The single-Ranger phases (2, 3) do not use a centralized critic — they share the actor's trunk."

---

## Final hand-off

When Task 10 reports green, all the work is in the worktree on branch `phase4-critic-obs`. Per user preference, do NOT auto-commit. Stop here and ask the user how they want to integrate (merge / PR / cherry-pick).

**Followups to flag for the user (next slices, not this one):**
- 3v3 env (`python/envs/phase4_mappo.py`) — depends on this slice landing.
- Centralized critic forward pass — first consumer of the new obs.
- Determinism golden hashes for `team_size=3` — separate goldens, not a rebaseline of the 1v1 ones.
