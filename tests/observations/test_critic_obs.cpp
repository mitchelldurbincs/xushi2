#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <xushi2/sim/obs.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

namespace {

using xushi2::common::Action;
using xushi2::common::Team;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::kCriticObsPhase1Dim;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

constexpr float kEps = 1e-5F;

std::array<float, kCriticObsPhase1Dim> build_critic(const Sim& sim, Team team) {
    std::array<float, kCriticObsPhase1Dim> out{};
    xushi2::sim::build_critic_obs_phase1(
        sim, team, out.data(), static_cast<std::uint32_t>(out.size()));
    return out;
}

TEST(CriticObs, ShapeAndAllFinite) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

TEST(CriticObs, PrefixMatchesActorObsForSameTeam) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);

    std::array<float, kActorObsPhase1Dim> actor_a{};
    xushi2::sim::build_actor_obs_phase1(
        sim, 0, actor_a.data(),
        static_cast<std::uint32_t>(actor_a.size()));

    auto critic_a = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < kActorObsPhase1Dim; ++i) {
        EXPECT_FLOAT_EQ(critic_a[i], actor_a[i]) << "index " << i;
    }
}

TEST(CriticObs, WorldFrameOwnPositionMatchesHeroStateRaw) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_critic(sim, Team::A);
    const auto& own = sim.state().heroes[0];
    // world_own_position is the first two privileged entries.
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 0], own.position.x, kEps);
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 1], own.position.y, kEps);
}

TEST(CriticObs, TeamBWorldPositionIsTeamBRanger) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_critic(sim, Team::B);
    const auto& own_b = sim.state().heroes[3];
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 0], own_b.position.x, kEps);
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 1], own_b.position.y, kEps);
    // And world_enemy_position is Team A's Ranger.
    const auto& enemy_a = sim.state().heroes[0];
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 2], enemy_a.position.x, kEps);
    EXPECT_NEAR(obs[kActorObsPhase1Dim + 3], enemy_a.position.y, kEps);
}

TEST(CriticObs, RawTickCountersAtFreshStateAreZero) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_critic(sim, Team::A);
    // Order: ... (8 world-frame floats) cap_progress_ticks(1)
    // team_a_score_ticks(1) team_b_score_ticks(1) tick_raw(1) seed_hi(1)
    // seed_lo(1).
    const std::uint32_t base = kActorObsPhase1Dim + 8;
    EXPECT_NEAR(obs[base + 0], 0.0F, kEps);  // cap_progress_ticks
    EXPECT_NEAR(obs[base + 1], 0.0F, kEps);  // team_a_score_ticks
    EXPECT_NEAR(obs[base + 2], 0.0F, kEps);  // team_b_score_ticks
    EXPECT_NEAR(obs[base + 3], 0.0F, kEps);  // tick_raw
}

TEST(CriticObs, SeedBitsAreStableAcrossSteps) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xD1CEDA7A0BADF00DULL;
    Sim sim(cfg);
    auto a = build_critic(sim, Team::A);
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step_decision(idle);
    sim.step_decision(idle);
    auto b = build_critic(sim, Team::A);
    // Last two floats are seed bits; they must not change.
    EXPECT_FLOAT_EQ(a[kCriticObsPhase1Dim - 2], b[kCriticObsPhase1Dim - 2]);
    EXPECT_FLOAT_EQ(a[kCriticObsPhase1Dim - 1], b[kCriticObsPhase1Dim - 1]);
}

TEST(CriticObs, RawTickAdvancesAfterSteps) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    std::array<Action, kAgentsPerMatch> idle{};
    // One decision = action_repeat sim ticks.
    sim.step_decision(idle);
    auto obs = build_critic(sim, Team::A);
    // tick_raw is the 4th privileged counter after 8 world-frame floats +
    // 3 earlier counters (cap_progress, a_score, b_score).
    const float tick_raw = obs[kActorObsPhase1Dim + 8 + 3];
    EXPECT_GT(tick_raw, 0.0F);
}

}  // namespace
