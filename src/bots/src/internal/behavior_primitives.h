#pragma once

#include <xushi2/sim/sim.h>

#include <xushi2/common/action_canon.hpp>

namespace xushi2::bots::internal {

const sim::HeroState* find_opponent(const sim::MatchState& state,
                                    const sim::HeroState& self);

float aim_delta_toward(const sim::HeroState& self, float tx, float ty);

common::Action walk_to_objective(const sim::HeroState& self,
                                 const sim::MapBounds& map);

common::Action hold_and_shoot(const sim::MatchState& state,
                              const sim::HeroState& self);

}  // namespace xushi2::bots::internal
