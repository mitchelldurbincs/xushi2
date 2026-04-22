#pragma once

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Narrow update functions — the ONLY way RangerWeaponState mutates.
// See docs/game_design.md §6 "Reload behavior."
void weapon_on_fire_success(RangerWeaponState& w, const Phase1MechanicsConfig& m);
void weapon_on_combat_roll(RangerWeaponState& w);
void weapon_tick_update(RangerWeaponState& w);

}  // namespace xushi2::sim::internal
