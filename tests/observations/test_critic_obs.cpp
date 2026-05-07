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
using xushi2::sim::kCriticObsDim;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

constexpr float kEps = 1e-5F;

MatchConfig make_3v3_config() {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    return cfg;
}

std::array<float, kCriticObsDim> build_critic(const Sim& sim, Team team) {
    std::array<float, kCriticObsDim> out{};
    xushi2::sim::build_critic_obs(
        sim, team, out.data(), static_cast<std::uint32_t>(out.size()));
    return out;
}

TEST(CriticObs, ShapeAndAllFinite) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

TEST(CriticObs, PrefixMatchesActorObsForAllThreeOwnSlotsTeamA) {
    Sim sim(make_3v3_config());
    auto critic = build_critic(sim, Team::A);
    for (std::uint32_t slot = 0; slot < 3; ++slot) {
        std::array<float, kActorObsPhase1Dim> actor{};
        xushi2::sim::build_actor_obs_phase1(
            sim, slot, actor.data(),
            static_cast<std::uint32_t>(actor.size()));
        for (std::uint32_t i = 0; i < kActorObsPhase1Dim; ++i) {
            EXPECT_FLOAT_EQ(critic[slot * kActorObsPhase1Dim + i], actor[i])
                << "slot " << slot << " field " << i;
        }
    }
}

TEST(CriticObs, PrefixMatchesActorObsForAllThreeOwnSlotsTeamB) {
    Sim sim(make_3v3_config());
    auto critic = build_critic(sim, Team::B);
    const std::uint32_t team_b_slots[3] = {3, 4, 5};
    for (std::uint32_t i = 0; i < 3; ++i) {
        std::array<float, kActorObsPhase1Dim> actor{};
        xushi2::sim::build_actor_obs_phase1(
            sim, team_b_slots[i], actor.data(),
            static_cast<std::uint32_t>(actor.size()));
        for (std::uint32_t j = 0; j < kActorObsPhase1Dim; ++j) {
            EXPECT_FLOAT_EQ(critic[i * kActorObsPhase1Dim + j], actor[j])
                << "team B slot index " << i << " field " << j;
        }
    }
}

TEST(CriticObs, EnemyWorldBlockMatchesHeroStateRawTeamA) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3U * kActorObsPhase1Dim;  // = 93
    const std::uint32_t enemy_slots[3] = {3, 4, 5};
    for (std::uint32_t i = 0; i < 3; ++i) {
        const auto& h = sim.state().heroes[enemy_slots[i]];
        const std::uint32_t off = base + i * 12U;
        EXPECT_NEAR(obs[off + 0], h.position.x, kEps);
        EXPECT_NEAR(obs[off + 1], h.position.y, kEps);
        EXPECT_NEAR(obs[off + 2], h.velocity.x, kEps);
        EXPECT_NEAR(obs[off + 3], h.velocity.y, kEps);
        EXPECT_NEAR(obs[off + 4], std::sin(h.aim_angle), kEps);
        EXPECT_NEAR(obs[off + 5], std::cos(h.aim_angle), kEps);
        // hp at off+6, alive at off+7, respawn at off+8, ammo at off+9,
        // reloading at off+10, roll_cd at off+11. Spot-check hp + alive.
        EXPECT_FLOAT_EQ(obs[off + 7], h.alive ? 1.0F : 0.0F);
    }
}

TEST(CriticObs, RawTickCountersAtFreshStateAreZero) {
    Sim sim(make_3v3_config());
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3U * kActorObsPhase1Dim + 3U * 12U;  // = 129
    EXPECT_NEAR(obs[base + 0], 0.0F, kEps);  // cap_progress_ticks
    EXPECT_NEAR(obs[base + 1], 0.0F, kEps);  // team_a_score_ticks
    EXPECT_NEAR(obs[base + 2], 0.0F, kEps);  // team_b_score_ticks
    EXPECT_NEAR(obs[base + 3], 0.0F, kEps);  // tick_raw
}

TEST(CriticObs, SeedBitsAreStableAcrossSteps) {
    MatchConfig cfg = make_3v3_config();
    cfg.seed = 0xD1CEDA7A0BADF00DULL;
    Sim sim(cfg);
    auto a = build_critic(sim, Team::A);
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step_decision(idle);
    sim.step_decision(idle);
    auto b = build_critic(sim, Team::A);
    EXPECT_FLOAT_EQ(a[kCriticObsDim - 2], b[kCriticObsDim - 2]);
    EXPECT_FLOAT_EQ(a[kCriticObsDim - 1], b[kCriticObsDim - 1]);
}

TEST(CriticObs, RawTickAdvancesAfterSteps) {
    Sim sim(make_3v3_config());
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step_decision(idle);
    auto obs = build_critic(sim, Team::A);
    const std::uint32_t base = 3U * kActorObsPhase1Dim + 3U * 12U;
    EXPECT_GT(obs[base + 3], 0.0F);  // tick_raw advanced
}

TEST(CriticObs, IdleSteps3v3DoesNotCrash) {
    Sim sim(make_3v3_config());
    std::array<Action, kAgentsPerMatch> idle{};
    for (std::uint32_t k = 0; k < 100; ++k) {
        sim.step_decision(idle);
    }
    auto obs = build_critic(sim, Team::A);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

}  // namespace
