#pragma once

#include <array>

#include <xushi2/sim/sim.h>

#include "sim_combat.h"

namespace xushi2::sim::internal {

struct TickContext {
    MatchState& state;
    const MatchConfig& config;
    const std::array<common::Action, kAgentsPerMatch>& actions;
    const std::array<bool, kAgentsPerMatch>& aim_consumed;
};

void apply_one_tick(MatchState& state,
                    const MatchConfig& config,
                    const std::array<common::Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed);

}  // namespace xushi2::sim::internal
