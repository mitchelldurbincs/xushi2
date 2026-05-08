#pragma once

#include "sim_tick_pipeline.h"

namespace xushi2::sim::internal {

void stage_abilities_combat_roll(const TickContext& ctx);
void stage_abilities_ranger_mark_target(const TickContext& ctx);

}  // namespace xushi2::sim::internal
