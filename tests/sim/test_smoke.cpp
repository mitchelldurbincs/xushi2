#include <gtest/gtest.h>

#include <test_config.hpp>
#include <xushi2/sim/sim.h>

// The most basic sim sanity test: construction, reset, one step, read state.
TEST(SimSmoke, ConstructResetStep) {
    auto config = xushi2::test_support::make_test_config();
    config.seed = 1234;
    xushi2::sim::Sim sim(config);

    EXPECT_EQ(sim.state().tick, 0U);
    EXPECT_EQ(sim.state().objective.team_a_score_ticks, 0U);
    EXPECT_EQ(sim.state().objective.team_b_score_ticks, 0U);
    EXPECT_FALSE(sim.episode_over());

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> noop{};
    sim.step(noop);

    EXPECT_EQ(sim.state().tick, 1U);
}
