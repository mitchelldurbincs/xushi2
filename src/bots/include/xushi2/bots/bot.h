#pragma once

// Scripted bots. Used as:
//   - opponents during Phase 0 of the curriculum (pipeline smoke test)
//   - anchor baselines during evaluation (rl_design.md §11)
//   - reproducible opponents for golden-replay tests

#include <memory>
#include <string>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace xushi2::bots {

using common::Action;
using sim::MatchState;

// Every scripted bot implements this interface. Stateless *wrt the sim* —
// any bot memory lives inside the Bot instance itself, never inside the sim.
class IBot {
   public:
    virtual ~IBot() = default;

    // Called once per policy decision for a single agent slot.
    // `agent_index` is an offset into MatchState::heroes.
    virtual Action decide(const MatchState& state, int agent_index) = 0;

    virtual std::string name() const = 0;
};

// --- Concrete bots (stubs for Phase 0) ---

// Walks toward the objective along the shortest route. No aiming, no fire.
std::unique_ptr<IBot> make_walk_to_objective_bot();

// Holds position, fires at any visible enemy. Does not leave spawn.
std::unique_ptr<IBot> make_hold_and_shoot_bot();

// Combination of walk-to-objective + shoot-visible. Weakest meaningful bot.
std::unique_ptr<IBot> make_basic_bot();

}  // namespace xushi2::bots
