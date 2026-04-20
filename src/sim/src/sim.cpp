#include <xushi2/sim/sim.h>

// Phase-0 scaffold: enough structure for the C++ / CMake / pybind11 pipeline
// to build end-to-end. The full tick pipeline (game-design.md §11) is not
// yet implemented — see TODO markers below.

namespace xushi2::sim {

namespace {

void reset_state(MatchState& state, std::uint64_t seed) {
    state = MatchState{};
    state.rng.seed(seed);
    state.objective.owner = Team::Neutral;
    state.objective.cap_team = Team::Neutral;
    state.objective.cap_progress = 0.0F;
    state.objective.team_a_score = 0;
    state.objective.team_b_score = 0;
    state.objective.unlocked = false;
    // TODO: initialize hero roster and spawn positions.
}

}  // namespace

Sim::Sim(const MatchConfig& config) : config_(config) {
    reset_state(state_, config_.seed);
}

void Sim::reset() {
    reset_state(state_, config_.seed);
}

void Sim::reset(std::uint64_t seed) {
    reset_state(state_, seed);
}

void Sim::step(std::array<Action, kAgentsPerMatch> /*actions*/) {
    // Tick pipeline (game-design.md §11):
    //   1.  Read actions            (done by caller)
    //   2.  Clamp / validate actions
    //   3.  Update aim directions
    //   4.  Apply movement
    //   5.  Resolve wall collisions
    //   6.  Update cooldown timers
    //   7.  Activate / deactivate abilities
    //   8.  Rebuild spatial index
    //   9.  Compute per-agent visibility
    //  10.  Resolve weapon fire / abilities
    //  11.  Accumulate damage / healing
    //  12.  Apply accumulated effects simultaneously
    //  13.  Process deaths
    //  14.  Process respawn timers
    //  15.  Update objective control (state machine, game-design §3)
    //  16.  Compute rewards
    //  17.  Emit observations
    //  18.  Log state/action/reward frame
    // TODO: implement.
    state_.tick += 1;
}

bool Sim::episode_over() const noexcept {
    if (state_.objective.team_a_score >= 100 || state_.objective.team_b_score >= 100) {
        return true;
    }
    const Tick max_ticks = static_cast<Tick>(config_.round_length_seconds * kTickHz);
    return state_.tick >= max_ticks;
}

std::uint64_t Sim::state_hash() const noexcept {
    // Placeholder deterministic hash. The real implementation should feed
    // every piece of state (tick, heroes, objective, RNG position) into a
    // stable hash (e.g., FNV-1a or xxHash) in a fixed order.
    // TODO: implement real state hashing — used by golden-replay tests.
    std::uint64_t h = 1469598103934665603ULL;
    h ^= state_.tick;
    h *= 1099511628211ULL;
    h ^= static_cast<std::uint64_t>(state_.objective.team_a_score);
    h *= 1099511628211ULL;
    h ^= static_cast<std::uint64_t>(state_.objective.team_b_score);
    h *= 1099511628211ULL;
    return h;
}

}  // namespace xushi2::sim
