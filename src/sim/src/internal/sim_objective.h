#pragma once

#include <array>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

void objective_tick_update(ObjectiveState& obj,
                           const std::array<HeroState, kAgentsPerMatch>& heroes,
                           common::Tick now,
                           const MapBounds& map);

}  // namespace xushi2::sim::internal
