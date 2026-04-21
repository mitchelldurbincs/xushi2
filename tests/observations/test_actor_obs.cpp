#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <xushi2/common/limits.hpp>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

namespace {

using xushi2::common::Action;
using xushi2::common::Team;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

constexpr float kEps = 1e-5F;

// Indices into the Phase-1 actor obs tensor — MUST stay in sync with
// python/xushi2/obs_manifest.py and src/sim/src/actor_obs.cpp.
namespace idx {
constexpr std::uint32_t OWN_HP                 = 0;
constexpr std::uint32_t OWN_VELX               = 1;
constexpr std::uint32_t OWN_VELY               = 2;
constexpr std::uint32_t OWN_AIM_SIN            = 3;
constexpr std::uint32_t OWN_AIM_COS            = 4;
constexpr std::uint32_t OWN_POSX               = 5;
constexpr std::uint32_t OWN_POSY               = 6;
constexpr std::uint32_t OWN_AMMO               = 7;
constexpr std::uint32_t OWN_RELOADING          = 8;
constexpr std::uint32_t OWN_COMBAT_ROLL_CD     = 9;
constexpr std::uint32_t ENEMY_ALIVE            = 10;
constexpr std::uint32_t ENEMY_RESPAWN_TIMER    = 11;
constexpr std::uint32_t ENEMY_REL_POSX         = 12;
constexpr std::uint32_t ENEMY_REL_POSY         = 13;
constexpr std::uint32_t ENEMY_HP               = 14;
constexpr std::uint32_t ENEMY_VELX             = 15;
constexpr std::uint32_t ENEMY_VELY             = 16;
constexpr std::uint32_t OWNER_NEUTRAL          = 17;
constexpr std::uint32_t OWNER_US               = 18;
constexpr std::uint32_t OWNER_THEM             = 19;
constexpr std::uint32_t CAP_NEUTRAL            = 20;
constexpr std::uint32_t CAP_US                 = 21;
constexpr std::uint32_t CAP_THEM               = 22;
constexpr std::uint32_t CAP_PROGRESS           = 23;
constexpr std::uint32_t CONTESTED              = 24;
constexpr std::uint32_t OBJECTIVE_UNLOCKED     = 25;
constexpr std::uint32_t OWN_SCORE              = 26;
constexpr std::uint32_t ENEMY_SCORE            = 27;
constexpr std::uint32_t SELF_ON_POINT          = 28;
constexpr std::uint32_t ENEMY_ON_POINT         = 29;
constexpr std::uint32_t ROUND_TIMER            = 30;
}  // namespace idx
static_assert(idx::ROUND_TIMER + 1 == kActorObsPhase1Dim,
              "idx table must cover the whole actor obs tensor");

// Build an obs tensor for the given viewer slot into a local buffer.
std::array<float, kActorObsPhase1Dim> build_obs(const Sim& sim,
                                                std::uint32_t viewer_slot) {
    std::array<float, kActorObsPhase1Dim> out{};
    xushi2::sim::build_actor_obs_phase1(
        sim, viewer_slot, out.data(),
        static_cast<std::uint32_t>(out.size()));
    return out;
}

TEST(ActorObs, ShapeAndAllFinite) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    for (std::uint32_t i = 0; i < obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(obs[i])) << "index " << i;
    }
}

TEST(ActorObs, FreshStateOwnHpIsOne) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::OWN_HP], 1.0F, kEps);
}

TEST(ActorObs, FreshStateAmmoIsOneAndNotReloading) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::OWN_AMMO], 1.0F, kEps);
    EXPECT_NEAR(obs[idx::OWN_RELOADING], 0.0F, kEps);
}

TEST(ActorObs, FreshStateEnemyAliveIsOneAndRespawnTimerIsZero) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::ENEMY_ALIVE], 1.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_RESPAWN_TIMER], 0.0F, kEps);
}

TEST(ActorObs, FreshStateOwnerIsNeutral) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::OWNER_NEUTRAL], 1.0F, kEps);
    EXPECT_NEAR(obs[idx::OWNER_US], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::OWNER_THEM], 0.0F, kEps);
}

TEST(ActorObs, FreshStateObjectiveUnlockedIsZero) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::OBJECTIVE_UNLOCKED], 0.0F, kEps);
}

TEST(ActorObs, FreshStateRoundTimerIsZero) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.round_length_seconds = 60;
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::ROUND_TIMER], 0.0F, kEps);
}

TEST(ActorObs, FreshStateSelfAndEnemyNotOnPoint) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::SELF_ON_POINT], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_ON_POINT], 0.0F, kEps);
}

TEST(ActorObs, OneHotsSumToOneExactly) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::OWNER_NEUTRAL] + obs[idx::OWNER_US] +
                    obs[idx::OWNER_THEM],
                1.0F, kEps);
    EXPECT_NEAR(obs[idx::CAP_NEUTRAL] + obs[idx::CAP_US] + obs[idx::CAP_THEM],
                1.0F, kEps);
}

TEST(ActorObs, TeamAAndTeamBViewPositionsSumToZero) {
    // 180°-rotation symmetry of the fixed Phase-1 spawns means that the
    // team-frame own_position for A plus the team-frame own_position for
    // B must sum to zero component-wise: after mirroring both views, each
    // Ranger sees its spawn at the same relative coordinate.
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    auto obs_a = build_obs(sim, 0);
    auto obs_b = build_obs(sim, 3);
    EXPECT_NEAR(obs_a[idx::OWN_POSX] + obs_b[idx::OWN_POSX], 0.0F, 1e-4F);
    // After the 180° mirror, team-A's Y is at -0.8 and team-B's Y is ALSO
    // at -0.8 (both see themselves "at the bottom"); they do not cancel.
    // Instead, each team sees its own spawn mirrored to the same team-frame
    // coordinate.
    EXPECT_NEAR(obs_a[idx::OWN_POSY], obs_b[idx::OWN_POSY], 1e-4F);
}

TEST(ActorObs, KilledEnemyZeroesEnemyFields) {
    // Use a tiny arena so the default Phase-1 spawns start already within
    // revolver range and facing each other (mirrors tests/sim/test_combat).
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.fog_of_war_enabled = false;
    cfg.map.max_x = 5.0F;
    cfg.map.max_y = 5.0F;
    Sim sim(cfg);

    std::array<Action, kAgentsPerMatch> acts{};
    acts[0].primary_fire = true;  // Team A Ranger pumps rounds into Team B.

    // Fire shot 1 (tick 0), wait 15 ticks for fire-rate gate, fire shot 2.
    // Each shot = 75 HP off a 150 HP pool. Two shots kill.
    sim.step(acts);
    for (int j = 0; j < 14; ++j) sim.step(acts);
    sim.step(acts);  // killing shot
    ASSERT_FALSE(sim.state().heroes[3].alive);

    auto obs = build_obs(sim, 0);
    EXPECT_NEAR(obs[idx::ENEMY_ALIVE], 0.0F, kEps);
    EXPECT_GT(obs[idx::ENEMY_RESPAWN_TIMER], 0.0F);
    EXPECT_LE(obs[idx::ENEMY_RESPAWN_TIMER], 1.0F);
    EXPECT_NEAR(obs[idx::ENEMY_REL_POSX], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_REL_POSY], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_HP], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_VELX], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_VELY], 0.0F, kEps);
    EXPECT_NEAR(obs[idx::ENEMY_ON_POINT], 0.0F, kEps);
}

TEST(ActorObs, RoundTimerGrowsWithTick) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.round_length_seconds = 10;
    Sim sim(cfg);
    std::array<Action, kAgentsPerMatch> acts{};
    for (int t = 0; t < 30; ++t) {  // 1 second of sim -> 10 decisions @ 10 Hz
        sim.step_decision(acts);
    }
    auto obs = build_obs(sim, 0);
    EXPECT_GT(obs[idx::ROUND_TIMER], 0.0F);
    EXPECT_LT(obs[idx::ROUND_TIMER], 1.0F);
}

}  // namespace
