#pragma once

#include "sim_tick_pipeline.h"

namespace xushi2::sim::internal {

void stage_abilities_mender_weapon_swap(const TickContext& ctx);
void stage_mender_staff_beam(const TickContext& ctx);
void stage_abilities_mender_tether(const TickContext& ctx);

}  // namespace xushi2::sim::internal
