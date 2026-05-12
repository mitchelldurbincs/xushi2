#pragma once

#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

int nearest_enemy_in_range_with_los(const MatchState& state,
                                    std::uint32_t actor_slot,
                                    float range,
                                    const MatchConfig& config);

int nearest_enemy_in_cone_with_los(const MatchState& state,
                                  std::uint32_t actor_slot,
                                  float range,
                                  float half_angle_cos,
                                  const MatchConfig& config);

int nearest_ally_in_cone_with_los(const MatchState& state,
                                  std::uint32_t actor_slot,
                                  float range,
                                  float half_angle_cos,
                                  const MatchConfig& config);

struct RayHitCandidate {
    enum class Kind : std::uint8_t { None = 0, Enemy = 1, Barrier = 2 };
    Kind kind = Kind::None;
    int slot = -1;
    float t = 0.0F;
};

// Deterministic tie-break: when candidates share hit time (epsilon), lower slot wins.
RayHitCandidate first_ray_hit_enemy_or_barrier(const MatchState& state,
                                               std::uint32_t actor_slot,
                                               common::Vec2 direction_unit,
                                               float max_range,
                                               float enemy_hitbox_radius,
                                               const MatchConfig& config);

}  // namespace xushi2::sim::internal
