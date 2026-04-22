#pragma once

#include <array>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Apply one full simulation tick. Orchestrates stages 1–7 and 10–15 from
// game_design.md §11. Increments state.tick at the end. Stages are kept in
// strict declared order; their pre/post invariants are documented at the
// stage_* implementations in sim_tick_pipeline.cpp.
void apply_one_tick(MatchState& state,
                    const MatchConfig& config,
                    const std::array<common::Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed);

}  // namespace xushi2::sim::internal
