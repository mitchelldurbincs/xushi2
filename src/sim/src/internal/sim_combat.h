#pragma once

#include <array>
#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

struct DamageEvent {
    common::EntityId attacker_id = 0;
    std::uint32_t victim_slot = 0;
    std::uint32_t damage_centi_hp = 0;
};

using DamageBuffer = std::array<DamageEvent, kAgentsPerMatch>;

void resolve_revolver_fire(MatchState& state,
                           const std::array<common::Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
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
