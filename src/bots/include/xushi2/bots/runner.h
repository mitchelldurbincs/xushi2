#pragma once

// Scripted-match runner. Given a MatchConfig and two bot names, runs one
// episode to completion (sim decides via episode_over) and returns the
// per-decision state_hash trajectory. Used by:
//   - Python Phase-0 determinism harness (python/train/train.py)
//   - Golden-replay tests (tests/replay/test_golden_replay.cpp)
//   - Bot-regression tests (tests/bots/test_runner.cpp)
//
// Bots run in C++ with full MatchState access (Tier 1 per
// docs/coding_philosophy.md §2). Python stays thin — one call per episode.

#include <cstdint>
#include <string_view>
#include <vector>

#include <xushi2/sim/sim.h>

namespace xushi2::bots {

struct ScriptedEpisodeResult {
    std::vector<std::uint64_t> decision_hashes;  // one per step_decision() call
    std::uint32_t final_tick = 0;
    std::uint32_t team_a_kills = 0;
    std::uint32_t team_b_kills = 0;
    sim::Team winner = sim::Team::Neutral;       // Neutral = draw / unresolved
};

// Valid bot names: "walk_to_objective", "hold_and_shoot", "basic", "noop".
// Unknown names abort via X2_REQUIRE.
ScriptedEpisodeResult run_scripted_episode(const sim::MatchConfig& config,
                                           std::string_view bot_a_name,
                                           std::string_view bot_b_name);

}  // namespace xushi2::bots
