#include <gtest/gtest.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a combat: hitscan Revolver, damage apply, kill on HP depletion,
// fire-rate gate, simultaneous-kill tie-break.
//
// Setup: a tiny arena so the two default Rangers start within Revolver range
// and are facing each other (reset_state points team A up, team B down).

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

MatchConfig close_arena_config() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 1;
    cfg.round_length_seconds = 30;
    cfg.map.max_x = 5.0F;
    cfg.map.max_y = 5.0F;
    return cfg;
}

}  // namespace

TEST(Combat, HitscanHitsEnemyAndReducesHp) {
    Sim sim(close_arena_config());
    const auto start_hp = sim.state().heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;  // only team A fires
    sim.step(actions);

    // Team A's mag decremented; Team B took damage.
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);
    EXPECT_LT(sim.state().heroes[3].health_centi_hp, start_hp);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp,
              start_hp - 7500);  // 75.0 HP from config
}

TEST(Combat, FireRateGateSkipsConsecutiveShots) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    // Tick 0: fires, mag=5, fire_cooldown=15.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);

    // Tick 1: fire gate closed, mag unchanged.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);

    // Step through until cooldown expires (tick 15 total). Keep firing input.
    for (int i = 0; i < 14; ++i) {
        sim.step(actions);
    }
    // By tick 15 the gate opens and the shot lands.
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 4);
}

TEST(Combat, EmptyMagazineFireIsNoop) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;

    // Fire until mag is empty. Each shot uses 15-tick cooldown.
    for (int k = 0; k < 6; ++k) {
        sim.step(actions);
        for (int j = 0; j < 14; ++j) {
            sim.step(actions);
        }
    }
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 0);
    const auto enemy_hp_after_empty = sim.state().heroes[3].health_centi_hp;

    // Step one more tick with fire held; mag stays 0, enemy HP unchanged.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 0);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, enemy_hp_after_empty);
}

TEST(Combat, KillsCreditedAndDeathsTracked) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;

    // Two 75-HP hits kill a 150-HP Ranger. Space them by fire-rate cooldown.
    sim.step(actions);
    for (int j = 0; j < 14; ++j) {
        sim.step(actions);
    }
    sim.step(actions);  // the killing shot

    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 0);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].deaths, 1U);
    EXPECT_EQ(sim.team_a_kills(), 1U);
    EXPECT_EQ(sim.team_b_kills(), 0U);
}

TEST(Combat, SimultaneousKillBothDie) {
    // Drive both Rangers down to exactly 75 HP (one shot kill), then fire
    // on the same tick; both should die.
    Sim sim(close_arena_config());

    // Step 1: A fires, B down to 75 HP. A.cd=15, B.cd=0.
    std::array<Action, kAgentsPerMatch> only_a{};
    only_a[0].primary_fire = true;
    sim.step(only_a);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 7500);

    // Step 2: B fires this tick, A down to 75. B.cd=15, A.cd=14.
    std::array<Action, kAgentsPerMatch> only_b{};
    only_b[3].primary_fire = true;
    sim.step(only_b);
    EXPECT_EQ(sim.state().heroes[0].health_centi_hp, 7500);

    // Wait for both cooldowns to clear. A.cd=14, B.cd=15. Need 15 idle ticks.
    for (int j = 0; j < 15; ++j) {
        std::array<Action, kAgentsPerMatch> idle{};
        sim.step(idle);
    }
    EXPECT_EQ(sim.state().heroes[0].weapon.fire_cooldown_ticks, 0U);
    EXPECT_EQ(sim.state().heroes[3].weapon.fire_cooldown_ticks, 0U);

    // Both fire simultaneously. Damage accumulates, applies, both die.
    std::array<Action, kAgentsPerMatch> both{};
    both[0].primary_fire = true;
    both[3].primary_fire = true;
    sim.step(both);

    EXPECT_FALSE(sim.state().heroes[0].alive);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].kills, 1U);
}
