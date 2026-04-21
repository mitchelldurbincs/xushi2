#include <gtest/gtest.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a Ranger reload state machine (game_design.md §6).

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

// Full arena — no firing partner, so no damage / no kills. Just the reload
// state machine on slot 0.
MatchConfig reload_cfg() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 2;
    cfg.round_length_seconds = 60;
    return cfg;
}

void fire_once_and_wait(Sim& sim, int slot, int ticks_to_hold) {
    std::array<Action, kAgentsPerMatch> actions{};
    actions[static_cast<std::size_t>(slot)].primary_fire = true;
    sim.step(actions);
    for (int i = 0; i < ticks_to_hold; ++i) {
        std::array<Action, kAgentsPerMatch> no_fire{};
        sim.step(no_fire);
    }
}

}  // namespace

TEST(Reload, AutoReloadTriggersAfterInactivity) {
    Sim sim(reload_cfg());
    // Fire one shot, then stop firing. mag is 5.
    fire_once_and_wait(sim, 0, /*ticks_to_hold=*/0);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);
    EXPECT_FALSE(sim.state().heroes[0].weapon.reloading);

    // Wait kAutoReloadDelayTicks more ticks — auto-reload should trigger.
    for (std::uint32_t t = 0; t < xushi2::common::kAutoReloadDelayTicks; ++t) {
        std::array<Action, kAgentsPerMatch> no_fire{};
        sim.step(no_fire);
    }
    EXPECT_TRUE(sim.state().heroes[0].weapon.reloading);

    // Wait kReloadDurationTicks more — mag refills.
    for (std::uint32_t t = 0; t < xushi2::common::kReloadDurationTicks; ++t) {
        std::array<Action, kAgentsPerMatch> no_fire{};
        sim.step(no_fire);
    }
    EXPECT_FALSE(sim.state().heroes[0].weapon.reloading);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine,
              xushi2::common::kRangerMaxMagazine);
}

TEST(Reload, CombatRollCancelsInProgressReload) {
    Sim sim(reload_cfg());
    // Fire one shot.
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);

    // Idle until auto-reload fires.
    for (std::uint32_t t = 0; t < xushi2::common::kAutoReloadDelayTicks; ++t) {
        std::array<Action, kAgentsPerMatch> idle{};
        sim.step(idle);
    }
    ASSERT_TRUE(sim.state().heroes[0].weapon.reloading);

    // Combat Roll: impulse on step() means ability_1=1 fires (aim_consumed
    // is always false for raw step()). Should cancel reload and fill mag.
    std::array<Action, kAgentsPerMatch> roll{};
    roll[0].ability_1 = true;
    sim.step(roll);

    EXPECT_FALSE(sim.state().heroes[0].weapon.reloading);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine,
              xushi2::common::kRangerMaxMagazine);
}

TEST(Reload, CombatRollSetsCooldownFullValue) {
    Sim sim(reload_cfg());
    std::array<Action, kAgentsPerMatch> roll{};
    roll[0].ability_1 = true;
    sim.step(roll);
    EXPECT_EQ(sim.state().heroes[0].cd_ability_1,
              xushi2::common::kRangerCombatRollCooldownTicks);

    // Next tick, step 6 decrements it to N-1.
    std::array<Action, kAgentsPerMatch> idle{};
    sim.step(idle);
    EXPECT_EQ(sim.state().heroes[0].cd_ability_1,
              xushi2::common::kRangerCombatRollCooldownTicks - 1U);
}

TEST(Reload, FiringWhileReloadingIsNoop) {
    Sim sim(reload_cfg());
    std::array<Action, kAgentsPerMatch> fire{};
    fire[0].primary_fire = true;
    sim.step(fire);
    // Wait until auto-reload triggers.
    for (std::uint32_t t = 0; t < xushi2::common::kAutoReloadDelayTicks; ++t) {
        std::array<Action, kAgentsPerMatch> idle{};
        sim.step(idle);
    }
    ASSERT_TRUE(sim.state().heroes[0].weapon.reloading);
    const auto mag_before = sim.state().heroes[0].weapon.magazine;

    // Fire while reloading — mag must not decrement.
    sim.step(fire);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, mag_before);
    EXPECT_TRUE(sim.state().heroes[0].weapon.reloading);
}
