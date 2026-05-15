#pragma once

#include <array>
#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

struct DamageEvent {
    std::uint32_t victim_slot = 0;
    std::uint32_t damage_centi_hp = 0;
};

using DamageBuffer = std::array<DamageEvent, kAgentsPerMatch>;

float first_cover_hit_t(common::Vec2 origin,
                        common::Vec2 direction_unit,
                        float max_t,
                        const MatchConfig& config);

bool segment_blocked_by_cover(common::Vec2 a,
                              common::Vec2 b,
                              const MatchConfig& config);

void resolve_revolver_fire(MatchState& state,
                           const std::array<common::Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
                           const MatchConfig& config,
                           DamageBuffer& buf,
                           std::array<bool, kAgentsPerMatch>& has_damage);

void resolve_mender_sidearm_fire(MatchState& state,
                                 const std::array<common::Action, kAgentsPerMatch>& actions,
                                 const Phase1MechanicsConfig& m,
                                 const MatchConfig& config,
                                 DamageBuffer& buf,
                                 std::array<bool, kAgentsPerMatch>& has_damage);

void apply_damage_buffer(MatchState& state,
                         const DamageBuffer& buf,
                         const std::array<bool, kAgentsPerMatch>& has_damage);

void process_deaths(MatchState& state,
                    const DamageBuffer& buf,
                    const std::array<bool, kAgentsPerMatch>& has_damage,
                    const MatchConfig& config);

}  // namespace xushi2::sim::internal
