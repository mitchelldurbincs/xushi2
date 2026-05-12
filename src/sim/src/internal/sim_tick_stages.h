#pragma once

#include <xushi2/sim/sim.h>

#include "sim_tick_pipeline.h"

namespace xushi2::sim::internal {

float hero_speed(common::HeroKind kind);

void stage_validate_and_aim(const TickContext& ctx);
void stage_movement_and_bounds(const TickContext& ctx);
void stage_cooldowns_and_weapon_tick(MatchState& state);

}  // namespace xushi2::sim::internal
