#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include <test_config.hpp>
#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/math.hpp>
#include <xushi2/sim/sim.h>

// Phase-0 movement determinism. The slice: two Rangers on a 50×50 arena.
// With a non-zero move vector, state_hash must change tick-over-tick; same
// seed + same actions must produce identical hash trajectories.

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

std::array<Action, kAgentsPerMatch> make_actions_walk_north_for_slot_0() {
    std::array<Action, kAgentsPerMatch> a{};
    a[0].move_x = 0.0F;
    a[0].move_y = 1.0F;
    return a;
}

std::vector<std::uint64_t> run_with(const std::array<Action, kAgentsPerMatch>& actions,
                                    std::uint64_t seed, int max_ticks) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = seed;
    Sim sim(cfg);
    std::vector<std::uint64_t> hashes;
    hashes.reserve(static_cast<std::size_t>(max_ticks));
    for (int t = 0; t < max_ticks; ++t) {
        sim.step(actions);
        hashes.push_back(sim.state_hash());
    }
    return hashes;
}

}  // namespace

TEST(Movement, TwoRangersSpawnInsideArena) {
    auto cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto& s = sim.state();
    EXPECT_TRUE(s.heroes[0].present);
    EXPECT_TRUE(s.heroes[0].alive);
    EXPECT_TRUE(s.heroes[3].present);
    EXPECT_TRUE(s.heroes[3].alive);
    // Both Rangers must be inside arena bounds.
    EXPECT_GE(s.heroes[0].position.x, cfg.map.min_x);
    EXPECT_LE(s.heroes[0].position.x, cfg.map.max_x);
    EXPECT_GE(s.heroes[0].position.y, cfg.map.min_y);
    EXPECT_LE(s.heroes[0].position.y, cfg.map.max_y);
    // Slots 1, 2, 4, 5 are not occupied in the Phase-0 slice.
    EXPECT_FALSE(s.heroes[1].present);
    EXPECT_FALSE(s.heroes[5].present);
}

TEST(Movement, WalkingNorthChangesPositionAndHash) {
    auto cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto start_y = sim.state().heroes[0].position.y;
    const auto h0 = sim.state_hash();

    auto actions = make_actions_walk_north_for_slot_0();
    for (int t = 0; t < 30; ++t) {  // 1 second
        sim.step(actions);
    }
    const auto h1 = sim.state_hash();
    EXPECT_NE(h0, h1);
    EXPECT_GT(sim.state().heroes[0].position.y, start_y);
}

TEST(Movement, SameSeedSameActionsSameTrajectory) {
    const auto actions = make_actions_walk_north_for_slot_0();
    const auto a = run_with(actions, 42, 90);
    const auto b = run_with(actions, 42, 90);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a[i], b[i]) << "divergence at tick " << i;
    }
}

TEST(Movement, DifferentSeedsGiveDifferentHashes) {
    // state_hash mixes in the RNG state; different seeds must differ even
    // when the visible hero state is identical.
    const auto actions = make_actions_walk_north_for_slot_0();
    const auto a = run_with(actions, 1, 30);
    const auto b = run_with(actions, 2, 30);
    bool any_different = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different);
}

TEST(Movement, StepDecisionAdvancesActionRepeatTicks) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.action_repeat = 3;
    Sim sim(cfg);
    const auto start_tick = sim.state().tick;
    std::array<Action, kAgentsPerMatch> actions{};
    sim.step_decision(actions);
    EXPECT_EQ(sim.state().tick, start_tick + cfg.action_repeat);
}

TEST(Movement, AimDeltaOnlyAppliedOncePerDecision) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.action_repeat = 3;
    Sim sim_decision(cfg);
    Sim sim_perstep(cfg);

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].aim_delta = 0.1F;

    // step_decision applies aim_delta once, then advances 3 ticks total.
    sim_decision.step_decision(actions);

    // Per-tick stepping must NOT advance aim 3x — each step() call treats its
    // input as a fresh decision, but to emulate one policy decision across
    // 3 ticks, the second and third ticks should have aim_delta zeroed.
    sim_perstep.step(actions);
    std::array<Action, kAgentsPerMatch> actions_no_aim = actions;
    actions_no_aim[0].aim_delta = 0.0F;
    sim_perstep.step(actions_no_aim);
    sim_perstep.step(actions_no_aim);

    EXPECT_FLOAT_EQ(sim_decision.state().heroes[0].aim_angle,
                    sim_perstep.state().heroes[0].aim_angle);
}

TEST(Movement, ActionCanonicalizationIsIdempotent) {
    Action a{};
    a.move_x = 0.333333F;
    a.move_y = -0.777F;
    a.aim_delta = 0.1F;
    xushi2::common::canonicalize_action(a);
    Action b = a;
    xushi2::common::canonicalize_action(b);
    EXPECT_EQ(a.move_x, b.move_x);
    EXPECT_EQ(a.move_y, b.move_y);
    EXPECT_EQ(a.aim_delta, b.aim_delta);
}

TEST(Movement, ActionCanonicalizationClampsAimDelta) {
    Action a{};
    a.aim_delta = 10.0F;  // way beyond ±π/4
    xushi2::common::canonicalize_action(a);
    EXPECT_LE(a.aim_delta, xushi2::common::kAimDeltaMax + 1e-4F);
    EXPECT_GE(a.aim_delta, -xushi2::common::kAimDeltaMax - 1e-4F);
}
