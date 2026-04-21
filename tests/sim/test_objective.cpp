#include <gtest/gtest.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a objective state machine (game_design.md §3). Tested by driving
// heroes into/out of the center circle using scripted bots. Bots park
// within 0.25 u of the arena center once they arrive, so extended-presence
// scenarios are stable.

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;
using xushi2::sim::Team;

MatchConfig objective_cfg() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 3;
    cfg.round_length_seconds = 240;
    return cfg;
}

// Drive the sim for n ticks using one bot per team slot. For each present
// hero, the team's bot decides; other slots get Action{}.
void step_with_bots(Sim& sim, xushi2::bots::IBot* bot_a,
                    xushi2::bots::IBot* bot_b, int n) {
    for (int i = 0; i < n; ++i) {
        std::array<Action, kAgentsPerMatch> actions{};
        const auto& state = sim.state();
        for (int slot = 0; slot < kAgentsPerMatch; ++slot) {
            const auto& h = state.heroes[static_cast<std::size_t>(slot)];
            if (!h.present) {
                continue;
            }
            xushi2::bots::IBot* bot = (h.team == Team::A) ? bot_a : bot_b;
            actions[static_cast<std::size_t>(slot)] = bot->decide(state, slot);
        }
        sim.step(actions);
    }
}

}  // namespace

TEST(Objective, LockedForFirst15Seconds) {
    Sim sim(objective_cfg());
    auto walk = xushi2::bots::make_walk_to_objective_bot();
    // Both teams walk to center; both end up on the point (but objective is
    // locked so no progress regardless).
    step_with_bots(sim, walk.get(), walk.get(),
                   static_cast<int>(xushi2::common::kObjectiveLockTicks) - 1);
    EXPECT_EQ(sim.state().objective.team_a_score_ticks, 0U);
    EXPECT_EQ(sim.state().objective.team_b_score_ticks, 0U);
    EXPECT_EQ(sim.state().objective.cap_progress_ticks, 0U);
    EXPECT_FALSE(sim.state().objective.unlocked);
}

TEST(Objective, UnlocksAtLockTicks) {
    Sim sim(objective_cfg());
    auto noop = xushi2::bots::make_noop_bot();
    step_with_bots(sim, noop.get(), noop.get(),
                   static_cast<int>(xushi2::common::kObjectiveLockTicks));
    EXPECT_TRUE(sim.state().objective.unlocked);
}

TEST(Objective, UncontestedNonOwnerCaptures) {
    Sim sim(objective_cfg());
    // Team A walks to center, parks there; Team B stays at spawn.
    auto walk = xushi2::bots::make_walk_to_objective_bot();
    auto noop = xushi2::bots::make_noop_bot();

    // Walk long enough for A to reach the point AND get past the lock.
    // Distance 20 u at 4.2 u/s = ~143 ticks to arrive.
    step_with_bots(sim, walk.get(), noop.get(),
                   static_cast<int>(xushi2::common::kObjectiveLockTicks));
    ASSERT_TRUE(sim.state().objective.unlocked);

    // From the unlock tick, A is on point (non-owner). Capture completes
    // after kCaptureTicks more ticks.
    step_with_bots(sim, walk.get(), noop.get(),
                   static_cast<int>(xushi2::common::kCaptureTicks));
    EXPECT_EQ(sim.state().objective.owner, Team::A);
    EXPECT_EQ(sim.state().objective.cap_progress_ticks, 0U);
    EXPECT_EQ(sim.state().objective.cap_team, Team::Neutral);
}

TEST(Objective, UncontestedOwnerScoresEveryTick) {
    Sim sim(objective_cfg());
    auto walk = xushi2::bots::make_walk_to_objective_bot();
    auto noop = xushi2::bots::make_noop_bot();

    // Get A to ownership: lock + capture.
    step_with_bots(sim, walk.get(), noop.get(),
                   static_cast<int>(xushi2::common::kObjectiveLockTicks +
                                    xushi2::common::kCaptureTicks));
    ASSERT_EQ(sim.state().objective.owner, Team::A);

    // 100 more ticks with A on point: +1 score per tick.
    const auto before = sim.state().objective.team_a_score_ticks;
    step_with_bots(sim, walk.get(), noop.get(), 100);
    EXPECT_EQ(sim.state().objective.team_a_score_ticks, before + 100U);
    EXPECT_EQ(sim.state().objective.team_b_score_ticks, 0U);
}

TEST(Objective, ContestedFreezesCapProgress) {
    Sim sim(objective_cfg());
    auto walk = xushi2::bots::make_walk_to_objective_bot();
    // Both teams walk to center and park — both are present on the point.
    step_with_bots(sim, walk.get(), walk.get(),
                   static_cast<int>(xushi2::common::kObjectiveLockTicks) + 200);

    // Capture progress stays 0 throughout contested presence, no score.
    EXPECT_EQ(sim.state().objective.cap_progress_ticks, 0U);
    EXPECT_EQ(sim.state().objective.team_a_score_ticks, 0U);
    EXPECT_EQ(sim.state().objective.team_b_score_ticks, 0U);
}

TEST(Objective, ScoresMonotoneNonDecreasing) {
    Sim sim(objective_cfg());
    auto walk = xushi2::bots::make_walk_to_objective_bot();
    auto noop = xushi2::bots::make_noop_bot();
    std::uint32_t last_a = 0;
    std::uint32_t last_b = 0;
    for (int i = 0; i < 1000; ++i) {
        step_with_bots(sim, walk.get(), noop.get(), 1);
        EXPECT_GE(sim.state().objective.team_a_score_ticks, last_a);
        EXPECT_GE(sim.state().objective.team_b_score_ticks, last_b);
        last_a = sim.state().objective.team_a_score_ticks;
        last_b = sim.state().objective.team_b_score_ticks;
    }
}
