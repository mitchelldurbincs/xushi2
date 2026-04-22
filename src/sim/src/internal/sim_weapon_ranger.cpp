#include "sim_weapon_ranger.h"

#include <cstdint>
#include <limits>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

namespace xushi2::sim::internal {

// --- Ranger weapon state transitions (game_design.md §6). ---
//
// These are the ONLY functions that mutate a RangerWeaponState. Each one
// asserts preconditions and postconditions; invalid states are made hard to
// represent by forcing all mutations through these entry points.

void weapon_on_fire_success(RangerWeaponState& w,
                            const Phase1MechanicsConfig& m) {
    X2_REQUIRE(w.magazine > 0, common::ErrorCode::CorruptState);
    X2_REQUIRE(!w.reloading, common::ErrorCode::CorruptState);
    X2_REQUIRE(w.fire_cooldown_ticks == 0, common::ErrorCode::CorruptState);
    w.magazine -= 1;
    w.ticks_since_last_shot = 0;
    w.fire_cooldown_ticks = m.revolver_fire_cooldown_ticks;
    X2_ENSURE(w.magazine <= common::kRangerMaxMagazine, common::ErrorCode::CorruptState);
}

void weapon_on_combat_roll(RangerWeaponState& w) {
    // Instant reload: cancels any in-progress auto-reload.
    w.magazine = common::kRangerMaxMagazine;
    w.reloading = false;
    w.reload_ticks_left = 0;
    X2_ENSURE(w.magazine == common::kRangerMaxMagazine, common::ErrorCode::CorruptState);
    X2_ENSURE(!w.reloading, common::ErrorCode::CorruptState);
}

void weapon_tick_update(RangerWeaponState& w) {
    // Fire-rate gate counts down every tick regardless of reload state.
    if (w.fire_cooldown_ticks > 0) {
        w.fire_cooldown_ticks -= 1;
    }
    if (w.reloading) {
        X2_INVARIANT(w.reload_ticks_left > 0, common::ErrorCode::CorruptState);
        w.reload_ticks_left -= 1;
        if (w.reload_ticks_left == 0) {
            w.magazine = common::kRangerMaxMagazine;
            w.reloading = false;
        }
        return;
    }
    // Not reloading. Increment inactivity counter first, then check the
    // auto-reload trigger — so exactly kAutoReloadDelayTicks idle ticks after
    // the last shot causes reload to begin this tick (not the next).
    if (w.ticks_since_last_shot < std::numeric_limits<Tick>::max()) {
        w.ticks_since_last_shot += 1;
    }
    if (w.magazine < common::kRangerMaxMagazine &&
        w.ticks_since_last_shot >= common::kAutoReloadDelayTicks) {
        w.reloading = true;
        w.reload_ticks_left = common::kReloadDurationTicks;
    }
}

}  // namespace xushi2::sim::internal
