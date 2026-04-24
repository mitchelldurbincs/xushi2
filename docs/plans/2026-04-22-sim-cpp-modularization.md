> **Status:** Completed 2026-04-22 (commit d0e6634).

# Sim.cpp Modularization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the monolithic `src/sim/src/sim.cpp` into domain-grouped internal modules while keeping the public `Sim` API (`include/xushi2/sim/sim.h`) and deterministic tick behavior byte-identical.

**Architecture:** Internal headers + cpps live under `src/sim/src/internal/` in the `xushi2::sim::internal` namespace. Cross-module helpers go through that namespace; intra-file-only helpers stay `static` (no anonymous namespace). `sim.cpp` shrinks to ctor/reset/step/state_hash glue that delegates to the new modules.

**Tech Stack:** C++17, CMake (Visual Studio 17 2022 generator), GoogleTest, existing `build/` directory in Release mode.

**User memory note:** Per saved feedback, no per-task commits — single commit + push at the end.

---

## Module Map (final state)

| Internal module | Header exposes | TU-private (`static`) |
|---|---|---|
| `sim_hash` | `compute_state_hash(MatchState const&)` returning `std::uint64_t` | `kFnvOffset/kFnvPrime`, `hash_bytes/i32/u32/u8`, `hash_weapon/hero/objective/rng` |
| `sim_spawn_reset` | `spawn_ranger`, `reset_state`, `respawn_tick_update` | — |
| `sim_weapon_ranger` | `weapon_on_fire_success`, `weapon_on_combat_roll`, `weapon_tick_update` | — |
| `sim_combat` | `DamageEvent`, `DamageBuffer`, `resolve_revolver_fire`, `apply_damage_buffer`, `process_deaths` | `ray_circle_hit_t` |
| `sim_objective` | `objective_tick_update` | `arena_center`, `inside_objective` |
| `sim_tick_pipeline` | `apply_one_tick` | `hero_speed`, `kVanguardSpeed/kRangerSpeed/kMenderSpeed`, `maybe_combat_roll`, the per-stage `stage_*` functions |

`validate_mechanics` stays in `sim.cpp` (ctor-only concern, not domain logic).

All public-from-internal helpers live in `namespace xushi2::sim::internal`.

---

## Task 1 — Branch + baseline verification

**Files:** none (verification only).

**Step 1.1 — Create the work branch:**
```
git checkout -b refactor/sim-internal-modules
```

**Step 1.2 — Confirm baseline build is clean (current `main`):**
```
cmake --build build --config Release
```
Expected: builds to completion, no errors. (Warnings are OK if pre-existing.)

**Step 1.3 — Confirm baseline tests pass:**
```
ctest --test-dir build -C Release --output-on-failure
```
Expected: all tests pass. **Record the count and any pre-existing failures** — that becomes the post-refactor target.

**Step 1.4 — Capture a baseline state hash for a manual sanity check** (the `test_golden_replay` test already does this, but having a quick eyeball value helps):
```
ctest --test-dir build -C Release -R test_golden_replay -V
```
Expected: PASS.

If any of 1.2/1.3/1.4 fails, **stop and report** before refactoring — we need a green baseline.

---

## Task 2 — Extract `sim_hash`

**Files:**
- Create: `src/sim/src/internal/sim_hash.h`
- Create: `src/sim/src/internal/sim_hash.cpp`
- Modify: `src/sim/src/sim.cpp` (remove hash helpers, replace `Sim::state_hash` body with one-line delegate)
- Modify: `src/sim/CMakeLists.txt` (add `src/internal/sim_hash.cpp`)

**Step 2.1 — Write the header.** Path: `src/sim/src/internal/sim_hash.h`

```cpp
#pragma once

#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Deterministic FNV-1a 64 hash of the full match state. Manifest of included
// fields lives in docs/determinism_rules.md §"state_hash() manifest".
std::uint64_t compute_state_hash(const MatchState& state);

}  // namespace xushi2::sim::internal
```

**Step 2.2 — Write the cpp.** Path: `src/sim/src/internal/sim_hash.cpp`

Move verbatim from `sim.cpp`:
- `kFnvOffset`, `kFnvPrime`
- `hash_bytes`, `hash_i32`, `hash_u32`, `hash_u8`
- `hash_weapon`, `hash_hero`, `hash_objective`, `hash_rng`
- The body of `Sim::state_hash` becomes a free function `compute_state_hash`.

Use `static` (not anonymous namespace) for the file-local helpers. Includes: `<cstdint>`, `<cmath>`, `<sstream>`, `<string>`, `<random>`, `xushi2/common/math.hpp` (for `quantize_pos`), `xushi2/sim/sim.h`, plus this module's own header.

**Step 2.3 — Replace `Sim::state_hash` in `sim.cpp`:**
```cpp
std::uint64_t Sim::state_hash() const noexcept {
    return internal::compute_state_hash(state_);
}
```
And delete the now-orphaned hash helpers + FNV constants from `sim.cpp`.

**Step 2.4 — Update `src/sim/CMakeLists.txt`** to add `src/internal/sim_hash.cpp` to `xushi2_sim` sources. Keep alphabetical/grouped order with existing entries.

**Step 2.5 — Build + test:**
```
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```
Expected: still green. **`test_golden_replay` and `test_determinism` are the load-bearing checks here** — if either flips, the move was not byte-identical (likely a missed include or a copy that altered hashing order).

---

## Task 3 — Extract `sim_weapon_ranger`

**Files:**
- Create: `src/sim/src/internal/sim_weapon_ranger.h`
- Create: `src/sim/src/internal/sim_weapon_ranger.cpp`
- Modify: `src/sim/src/sim.cpp` (remove the three weapon functions)
- Modify: `src/sim/CMakeLists.txt`

**Step 3.1 — Header:**
```cpp
#pragma once

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Narrow update functions — the ONLY way RangerWeaponState mutates.
// See docs/game_design.md §6 "Reload behavior."
void weapon_on_fire_success(RangerWeaponState& w, const Phase1MechanicsConfig& m);
void weapon_on_combat_roll(RangerWeaponState& w);
void weapon_tick_update(RangerWeaponState& w);

}  // namespace xushi2::sim::internal
```

**Step 3.2 — Cpp:** move the three functions verbatim from `sim.cpp` into `xushi2::sim::internal`. Includes: `<cstdint>`, `<limits>`, `xushi2/common/assert.hpp`, `xushi2/common/limits.hpp`, this module's header. Use `common::ErrorCode`, `common::kRangerMaxMagazine`, `common::kAutoReloadDelayTicks`, `common::kReloadDurationTicks` qualified inline.

**Step 3.3 — Delete those three functions from `sim.cpp`.**

**Step 3.4 — Update CMakeLists.**

**Step 3.5 — Build + test.** (At this point `sim.cpp` won't compile yet because `maybe_combat_roll` and `resolve_revolver_fire` still reference these unqualified — fix by adding `#include "internal/sim_weapon_ranger.h"` at the top of `sim.cpp` and adjusting call sites to `internal::weapon_on_fire_success(...)` etc.)

Expected: green tests.

---

## Task 4 — Extract `sim_spawn_reset`

**Files:**
- Create: `src/sim/src/internal/sim_spawn_reset.h`
- Create: `src/sim/src/internal/sim_spawn_reset.cpp`
- Modify: `src/sim/src/sim.cpp`
- Modify: `src/sim/CMakeLists.txt`

**Step 4.1 — Header:**
```cpp
#pragma once

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

void spawn_ranger(HeroState& h, common::Team team, common::EntityId id,
                  common::Vec2 position, float aim_angle);

void reset_state(MatchState& state, const MatchConfig& config);

void respawn_tick_update(HeroState& h, common::Tick now, const MatchConfig& config);

}  // namespace xushi2::sim::internal
```

**Step 4.2 — Cpp:** move `spawn_ranger`, `reset_state`, `respawn_tick_update` verbatim. Includes: `xushi2/common/limits.hpp`, header.

**Step 4.3 — Update `Sim::Sim`, `Sim::reset(...)` in `sim.cpp` to call `internal::reset_state(state_, config_);`.**

**Step 4.4 — Delete the moved functions from `sim.cpp`.**

**Step 4.5 — CMakeLists update + build + test.** Expected: green.

---

## Task 5 — Extract `sim_combat`

**Files:**
- Create: `src/sim/src/internal/sim_combat.h`
- Create: `src/sim/src/internal/sim_combat.cpp`
- Modify: `src/sim/src/sim.cpp`
- Modify: `src/sim/CMakeLists.txt`

**Step 5.1 — Header:**
```cpp
#pragma once

#include <array>
#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

struct DamageEvent {
    common::EntityId attacker_id = 0;
    std::uint32_t victim_slot = 0;
    std::uint32_t damage_centi_hp = 0;
};

using DamageBuffer = std::array<DamageEvent, kAgentsPerMatch>;

void resolve_revolver_fire(MatchState& state,
                           const std::array<common::Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
                           DamageBuffer& buf,
                           std::array<bool, kAgentsPerMatch>& has_damage);

void apply_damage_buffer(MatchState& state,
                         const DamageBuffer& buf,
                         const std::array<bool, kAgentsPerMatch>& has_damage);

void process_deaths(MatchState& state,
                    const DamageBuffer& buf,
                    const std::array<bool, kAgentsPerMatch>& has_damage,
                    const MatchConfig& config);

}  // namespace xushi2::sim::internal
```

**Step 5.2 — Cpp:** move `ray_circle_hit_t` (`static`), `DamageEvent`/`DamageBuffer` definitions (now in header), `resolve_revolver_fire`, `apply_damage_buffer`, `process_deaths`. `resolve_revolver_fire` calls `internal::weapon_on_fire_success` — include `sim_weapon_ranger.h`.

**Step 5.3 — Delete moved functions from `sim.cpp`.**

**Step 5.4 — Build + test.** Expected: green. **`test_combat` is the load-bearing check.**

---

## Task 6 — Extract `sim_objective`

**Files:**
- Create: `src/sim/src/internal/sim_objective.h`
- Create: `src/sim/src/internal/sim_objective.cpp`
- Modify: `src/sim/src/sim.cpp`
- Modify: `src/sim/CMakeLists.txt`

**Step 6.1 — Header:**
```cpp
#pragma once

#include <array>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

void objective_tick_update(ObjectiveState& obj,
                           const std::array<HeroState, kAgentsPerMatch>& heroes,
                           common::Tick now,
                           const MapBounds& map);

}  // namespace xushi2::sim::internal
```

**Step 6.2 — Cpp:** move `arena_center` (`static`), `inside_objective` (`static`), `objective_tick_update`. Includes: `xushi2/common/assert.hpp`, `xushi2/common/limits.hpp`, header.

**Step 6.3 — Delete moved functions from `sim.cpp`.**

**Step 6.4 — Build + test.** Expected: green. **`test_objective` is the load-bearing check.**

---

## Task 7 — Extract `sim_tick_pipeline` AND split `apply_one_tick` into stages

This is the largest task. Combine extraction with the stage-split per the user's amended request.

**Files:**
- Create: `src/sim/src/internal/sim_tick_pipeline.h`
- Create: `src/sim/src/internal/sim_tick_pipeline.cpp`
- Modify: `src/sim/src/sim.cpp`
- Modify: `src/sim/CMakeLists.txt`

**Step 7.1 — Header:**
```cpp
#pragma once

#include <array>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Apply one full simulation tick. Orchestrates stages 1–7 and 10–15 from
// game_design.md §11. Increments state.tick at the end. Stages are kept in
// strict declared order; their pre/post invariants are documented at the
// stage_* implementations in sim_tick_pipeline.cpp.
void apply_one_tick(MatchState& state,
                    const MatchConfig& config,
                    const std::array<common::Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed);

}  // namespace xushi2::sim::internal
```

**Step 7.2 — Cpp scaffolding.** Includes:
```
"internal/sim_combat.h", "internal/sim_objective.h",
"internal/sim_spawn_reset.h", "internal/sim_weapon_ranger.h",
"internal/sim_tick_pipeline.h",
<xushi2/common/action_canon.hpp>, <xushi2/common/assert.hpp>,
<xushi2/common/limits.hpp>, <xushi2/common/math.hpp>,
<algorithm>, <cmath>, <limits>
```

Top of file (TU-private):
```cpp
namespace xushi2::sim::internal {
namespace {

inline constexpr float kVanguardSpeed = 3.6F;
inline constexpr float kRangerSpeed   = 4.2F;
inline constexpr float kMenderSpeed   = 4.0F;

float hero_speed(common::HeroKind kind) { /* unchanged */ }
void  maybe_combat_roll(HeroState& h, const common::Action& a,
                        bool aim_consumed, const MapBounds& map) { /* unchanged, calls internal::weapon_on_combat_roll */ }

}  // namespace
```

**Step 7.3 — Define stage functions** in declared order. Each takes the same data the original loop body uses. All `static` (or anonymous-namespace) since they're only called by `apply_one_tick`.

```cpp
// Pre: actions canonicalized; aim_consumed[i] true iff this tick is
//      a non-first sub-tick of step_decision().
// Post: living heroes' aim_angle updated; positions/velocities untouched.
static void stage_validate_and_aim(MatchState& state,
                                   const std::array<common::Action, kAgentsPerMatch>& actions,
                                   const std::array<bool, kAgentsPerMatch>& aim_consumed);

// Pre: aim updated. Post: positions advanced by velocity*dt and clamped
//      to map bounds; velocity reflects the canonicalized move input.
static void stage_movement_and_bounds(MatchState& state,
                                      const MatchConfig& config,
                                      const std::array<common::Action, kAgentsPerMatch>& actions);

// Pre: positions stable. Post: per-hero ability cooldowns decremented;
//      living Ranger weapon state advanced (auto-reload bookkeeping).
//      MUST run before abilities so Combat Roll's cd check sees the
//      correct value, and before fire resolution so fire_cooldown_ticks
//      is current this tick.
static void stage_cooldowns_and_weapon_tick(MatchState& state);

// Pre: cooldowns decremented for THIS tick. Post: any qualifying Ranger
//      that requested ability_1 this decision-window is dashed and its
//      magazine refilled (instant reload). Only fires on the first
//      sub-tick (aim_consumed false) — impulse semantics.
static void stage_abilities_combat_roll(MatchState& state,
                                        const std::array<common::Action, kAgentsPerMatch>& actions,
                                        const std::array<bool, kAgentsPerMatch>& aim_consumed,
                                        const MapBounds& map);

// (Steps 8–9 — spatial index / fog — deferred to later phases.)

// Pre: positions + cooldowns current. Post: DamageBuffer populated;
//      attackers' magazines/fire-cooldowns updated. NO HP changes here.
//      Per-attacker tie-break is by victim slot index for determinism.
static void stage_fire_resolution(MatchState& state,
                                  const std::array<common::Action, kAgentsPerMatch>& actions,
                                  const Phase1MechanicsConfig& m,
                                  DamageBuffer& buf,
                                  std::array<bool, kAgentsPerMatch>& has_damage);

// Pre: DamageBuffer populated. Post: victim HP reduced. All damage from
//      this tick is applied SIMULTANEOUSLY (no kill-credit ordering bias)
//      — a victim already dead this tick is left at 0 HP; subsequent
//      damage to a dead victim is dropped.
static void stage_apply_damage(MatchState& state,
                               const DamageBuffer& buf,
                               const std::array<bool, kAgentsPerMatch>& has_damage);

// Pre: HP applied. Post: heroes at 0 HP marked dead with respawn_tick set;
//      death counters incremented; kill credit awarded to attackers whose
//      victim died this tick. MUST run after damage application so
//      simultaneous lethal trades both score.
static void stage_process_deaths(MatchState& state,
                                 const DamageBuffer& buf,
                                 const std::array<bool, kAgentsPerMatch>& has_damage,
                                 const MatchConfig& config);

// Pre: deaths processed. Post: any hero whose respawn_tick has elapsed is
//      respawned at its team's spawn point with full HP/magazine, kills
//      and deaths preserved. Order: respawn after death-processing so a
//      hero that died and revived in the same tick is impossible.
static void stage_respawn(MatchState& state, const MatchConfig& config);

// Pre: hero positions/alive flags reflect this tick's outcomes. Post:
//      ObjectiveState advanced by one tick of the 5-case state machine.
//      Score counters monotonically non-decreasing (asserted inside).
static void stage_objective(MatchState& state, const MapBounds& map);
```

For each stage, the body is the corresponding loop or call from the existing `apply_one_tick` — copy verbatim. The `DamageBuffer buf{}` and `has_damage` array are owned by `apply_one_tick` (preserves data flow exactly).

**Step 7.4 — Replace `apply_one_tick` body** with sequential calls in original order:
```cpp
void apply_one_tick(MatchState& state, const MatchConfig& config,
                    const std::array<common::Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    stage_validate_and_aim(state, actions, aim_consumed);
    stage_movement_and_bounds(state, config, actions);
    stage_cooldowns_and_weapon_tick(state);
    stage_abilities_combat_roll(state, actions, aim_consumed, config.map);
    // Steps 8–9 (spatial index, fog) deferred — Phase 7+.
    DamageBuffer buf{};
    std::array<bool, kAgentsPerMatch> has_damage{};
    stage_fire_resolution(state, actions, config.mechanics, buf, has_damage);
    stage_apply_damage(state, buf, has_damage);
    stage_process_deaths(state, buf, has_damage, config);
    stage_respawn(state, config);
    stage_objective(state, config.map);
    // Steps 16–18 (rewards / obs / replay) deferred — Phase 1b/1c.
    state.tick += 1;
}
```

**Step 7.5 — In `sim.cpp`:** delete `apply_one_tick`, `maybe_combat_roll`, `hero_speed`, the `kVanguardSpeed/kRangerSpeed/kMenderSpeed` constants. Update `Sim::step` and `Sim::step_decision` to call `internal::apply_one_tick(...)`.

**Step 7.6 — CMakeLists update + build + test.**
Expected: **all tests green, including `test_golden_replay` (byte-identical state hash) and `test_determinism`.** This is the strongest signal that the stage split preserved order and data flow.

---

## Task 8 — Final shape of `sim.cpp` + sweep

**Step 8.1 — Verify `sim.cpp` is now thin glue only.** It should contain:
- `validate_mechanics` (TU-private — `static` or anonymous namespace)
- `Sim::Sim` (calls `validate_mechanics` + `internal::reset_state`)
- `Sim::reset()` and `Sim::reset(seed)` (call `internal::reset_state`)
- `Sim::step` / `Sim::step_decision` (call `internal::apply_one_tick`)
- `Sim::episode_over`, `Sim::winner`, `Sim::team_a_kills`, `Sim::team_b_kills` (pure read accessors — leave in place)
- `Sim::state_hash` (one-line delegate to `internal::compute_state_hash`)

No `using common::...` aliases needed beyond what these tiny functions touch.

**Step 8.2 — Internal-linkage audit.** Grep for any non-`static` non-`internal::` free function leaking from the new `.cpp` files:
```
grep -nE '^[[:space:]]*(void|float|bool|int|std::|auto|template)' src/sim/src/internal/*.cpp
```
Anything outside `namespace internal` and not marked `static` is a leak — fix it.

**Step 8.3 — Final build (Debug + Release) + full ctest:**
```
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```
Expected: identical pass count to baseline from Task 1.3.

**Step 8.4 — Sanity-diff the public header:**
```
git diff main -- src/sim/include/xushi2/sim/sim.h
```
Expected: **empty diff** (public API unchanged is a hard requirement).

---

## Task 9 — Commit + push

**Step 9.1 — Stage only the refactor:**
```
git add src/sim/CMakeLists.txt src/sim/src/sim.cpp src/sim/src/internal/ docs/plans/2026-04-22-sim-cpp-modularization.md
```
(Leave `python/`, `.claude/settings.local.json`, and the `python/manual_*.csv` working-tree changes alone — they are unrelated.)

**Step 9.2 — Single commit.** Per saved memory, no per-task commits — one clean commit captures the whole refactor:
```
git commit -m "refactor(sim): split sim.cpp into domain-grouped internal modules"
```

**Step 9.3 — Push the branch:**
```
git push -u origin refactor/sim-internal-modules
```

**Step 9.4 — Report URL** of the pushed branch / PR-create link to the user.

---

## Risk / regression notes

- **Determinism is the load-bearing invariant.** `test_golden_replay` and `test_determinism` will detect any subtle move that changes evaluation order or hash input ordering. If either flips, revert that single task and inspect.
- **No behavior change is allowed in this branch.** Any tempting "while I'm in here" cleanup (renames, comment edits beyond stage invariants, dead-code removal) goes in a follow-up.
- The stage split in Task 7 is the only structural change beyond pure relocation; it must keep the exact same operations in the exact same order against the exact same data, including the deferred-step comments and the `state.tick += 1` placement.
