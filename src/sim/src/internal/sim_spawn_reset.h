#pragma once

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

void spawn_ranger(HeroState& h, common::Team team, common::EntityId id,
                  common::Vec2 position, float aim_angle);

void reset_state(MatchState& state, const MatchConfig& config);

void respawn_tick_update(HeroState& h, common::Tick now, const MatchConfig& config);

}  // namespace xushi2::sim::internal
