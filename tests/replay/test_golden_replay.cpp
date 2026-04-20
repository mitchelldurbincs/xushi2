#include <gtest/gtest.h>

#include <xushi2/sim/sim.h>

// Golden-replay CI test. Compares a freshly-run deterministic rollout
// against a checkpointed list of (tick, state_hash) pairs stored under
// data/replays/. See docs/determinism_rules.md and docs/replay_format.md.
//
// Phase 0: placeholder. The real test will:
//   1. Load data/replays/golden_phase0.replay
//   2. Re-run the same seed + same scripted actions
//   3. Assert every state hash matches
//   4. Assert the final tick count matches

TEST(GoldenReplay, Placeholder) {
    xushi2::sim::MatchConfig config{};
    config.seed = 0xD1CEDA7AULL;
    xushi2::sim::Sim sim(config);

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> noop{};
    for (int t = 0; t < 100; ++t) {
        sim.step(noop);
    }
    // Record the current hash — once the sim is real, this becomes the
    // golden checkpoint.
    const std::uint64_t terminal_hash = sim.state_hash();
    EXPECT_NE(terminal_hash, 0ULL);
    SUCCEED() << "golden-replay harness wired, awaiting sim implementation";
}
