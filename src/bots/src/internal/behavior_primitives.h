#pragma once

#include <xushi2/sim/sim.h>

#include <xushi2/common/action_canon.hpp>

namespace xushi2::bots::internal {

// Returns the closest live opponent by squared Euclidean distance, or
// nullptr if none exist. Ties are broken by lower slot index (stable
// across calls thanks to the strict-less comparison).
const sim::HeroState* find_opponent(const sim::MatchState& state,
                                    const sim::HeroState& self,
                                    const sim::MatchConfig& config);

// Computes the per-decision aim delta needed to face (tx, ty). The
// optional `noise_radians` is added to the raw angular error *before*
// wrap_angle + clamp, so the final delta is always inside the legal
// action range no matter how large the noise.
float aim_delta_toward(const sim::HeroState& self, float tx, float ty,
                       float noise_radians = 0.0F);

common::Action walk_to_objective(const sim::HeroState& self,
                                 const sim::MapBounds& map);

// `aim_noise_radians` is forwarded to aim_delta_toward and applied
// pre-clamp. Default 0 = perfect aim.
common::Action hold_and_shoot(const sim::MatchState& state,
                              const sim::HeroState& self,
                              const sim::MatchConfig& config,
                              float aim_noise_radians = 0.0F);

}  // namespace xushi2::bots::internal
