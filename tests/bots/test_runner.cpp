#include <gtest/gtest.h>

#include <test_config.hpp>
#include <xushi2/bots/runner.h>
#include <xushi2/sim/sim.h>

// Scripted-match runner smoke + determinism tests.

namespace {

xushi2::sim::MatchConfig phase0_config(std::uint64_t seed = 0xD1CEDA7AULL) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = seed;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = false;
    cfg.randomize_map = false;
    return cfg;
}

// round_length 30s × 30 Hz / action_repeat 3 = 300 decisions.
constexpr std::size_t kExpectedDecisions = 300;

}  // namespace

TEST(Runner, BasicVsBasicIsIntraProcessDeterministic) {
    const auto a = xushi2::bots::run_scripted_episode(phase0_config(), "basic", "basic");
    const auto b = xushi2::bots::run_scripted_episode(phase0_config(), "basic", "basic");
    ASSERT_EQ(a.decision_hashes.size(), kExpectedDecisions);
    ASSERT_EQ(a.decision_hashes, b.decision_hashes);
    ASSERT_EQ(a.final_tick, b.final_tick);
}

TEST(Runner, EachBotNameIsDeterministic) {
    for (const char* name : {"walk_to_objective", "hold_and_shoot", "basic", "noop"}) {
        const auto a = xushi2::bots::run_scripted_episode(phase0_config(), name, name);
        const auto b = xushi2::bots::run_scripted_episode(phase0_config(), name, name);
        ASSERT_EQ(a.decision_hashes, b.decision_hashes) << "bot=" << name;
    }
}

TEST(Runner, BasicDiffersFromNoop) {
    const auto basic =
        xushi2::bots::run_scripted_episode(phase0_config(), "basic", "basic");
    const auto noop =
        xushi2::bots::run_scripted_episode(phase0_config(), "noop", "noop");
    ASSERT_EQ(basic.decision_hashes.size(), noop.decision_hashes.size());
    // At least one decision must differ — bots must actually do something.
    EXPECT_NE(basic.decision_hashes, noop.decision_hashes);
}

TEST(Runner, BasicEvolvesState) {
    const auto r = xushi2::bots::run_scripted_episode(phase0_config(), "basic", "basic");
    ASSERT_FALSE(r.decision_hashes.empty());
    EXPECT_NE(r.decision_hashes.front(), r.decision_hashes.back());
}
