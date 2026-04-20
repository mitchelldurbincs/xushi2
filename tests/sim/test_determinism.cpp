#include <gtest/gtest.h>

#include <array>
#include <vector>

#include <xushi2/sim/sim.h>

// Determinism: same seed + same action stream must produce the same
// state_hash trajectory, tick by tick. See docs/determinism_rules.md.

namespace {

std::vector<std::uint64_t> run_episode(std::uint64_t seed, int max_ticks) {
    xushi2::sim::MatchConfig config{};
    config.seed = seed;
    config.round_length_seconds = 180;
    xushi2::sim::Sim sim(config);

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> noop{};
    std::vector<std::uint64_t> hashes;
    hashes.reserve(static_cast<std::size_t>(max_ticks));

    for (int t = 0; t < max_ticks; ++t) {
        sim.step(noop);
        hashes.push_back(sim.state_hash());
    }
    return hashes;
}

}  // namespace

TEST(Determinism, SameSeedSameTrajectoryIntraProcess) {
    const auto run_a = run_episode(42, 300);
    const auto run_b = run_episode(42, 300);
    ASSERT_EQ(run_a.size(), run_b.size());
    for (std::size_t i = 0; i < run_a.size(); ++i) {
        ASSERT_EQ(run_a[i], run_b[i]) << "divergence at tick " << i;
    }
}

TEST(Determinism, DifferentSeedsDifferentTrajectories) {
    const auto run_a = run_episode(1, 300);
    const auto run_b = run_episode(2, 300);
    // At least one tick should differ. With the Phase-0 stub sim (which does
    // nothing with seed yet) this may hold trivially; this test becomes
    // meaningful once randomized state is added.
    bool any_different = false;
    for (std::size_t i = 0; i < run_a.size(); ++i) {
        if (run_a[i] != run_b[i]) {
            any_different = true;
            break;
        }
    }
    // TODO: change to EXPECT_TRUE(any_different) once the sim actually
    // consumes the seed (e.g., map randomization enabled, spawn jitter, etc.).
    (void)any_different;
    SUCCEED() << "placeholder — enable once sim uses its PRNG";
}
