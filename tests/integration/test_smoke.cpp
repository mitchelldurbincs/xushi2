#include <gtest/gtest.h>

#include <xushi2/bots/bot.h>
#include <xushi2/sim/sim.h>

// End-to-end integration smoke test: sim + bots. Drives the sim with two
// scripted bot teams for a short rollout, asserts no crashes and a sane
// terminal state.

TEST(Integration, ScriptedBotsSmoke) {
    xushi2::sim::MatchConfig config{};
    config.seed = 7;
    config.round_length_seconds = 10;  // short round for the test
    xushi2::sim::Sim sim(config);

    auto bot = xushi2::bots::make_basic_bot();
    ASSERT_NE(bot, nullptr);

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};
    const int max_ticks = config.round_length_seconds * xushi2::sim::kTickHz;

    for (int t = 0; t < max_ticks && !sim.episode_over(); ++t) {
        for (int i = 0; i < xushi2::sim::kAgentsPerMatch; ++i) {
            actions[static_cast<std::size_t>(i)] = bot->decide(sim.state(), i);
        }
        sim.step(actions);
    }

    EXPECT_GT(sim.state().tick, 0U);
}
