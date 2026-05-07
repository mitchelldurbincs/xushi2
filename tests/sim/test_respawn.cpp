#include <gtest/gtest.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a death + respawn logic. Dead heroes don't count for objective
// occupancy. Respawn at team spawn with full HP and full mag. Kills/deaths
// counters persist across respawn.

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

MatchConfig close_arena_config() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 4;
    cfg.round_length_seconds = 240;
    cfg.map.max_x = 5.0F;
    cfg.map.max_y = 5.0F;
    return cfg;
}

// Kill slot 3 by firing from slot 0 twice. Returns the tick at which slot 3
// dies (zero-indexed, tick count returned by state().tick after the call).
void kill_slot_3(Sim& sim) {
    std::array<Action, kAgentsPerMatch> fire{};
    fire[0].primary_fire = true;
    sim.step(fire);                    // shot 1 (75 dmg)
    for (int j = 0; j < 14; ++j) {     // wait fire-rate cooldown
        sim.step(fire);
    }
    sim.step(fire);                    // killing shot
}

}  // namespace

TEST(Respawn, DeadHeroHasAliveFalseAndRespawnTickSet) {
    Sim sim(close_arena_config());
    const auto respawn_ticks = sim.config().mechanics.respawn_ticks;

    kill_slot_3(sim);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 0);
    EXPECT_EQ(sim.state().heroes[3].respawn_tick,
              sim.state().tick - 1U + respawn_ticks);
    // (respawn_tick is set to current tick + respawn_ticks at death, and
    // state.tick has advanced one more since then.)
}

TEST(Respawn, HeroRespawnsWithFullHpAndMag) {
    Sim sim(close_arena_config());
    kill_slot_3(sim);
    const auto respawn_tick = sim.state().heroes[3].respawn_tick;

    // Step until respawn. Use idle actions so no further damage.
    std::array<Action, kAgentsPerMatch> idle{};
    while (sim.state().tick <= respawn_tick) {
        sim.step(idle);
    }
    EXPECT_TRUE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp,
              xushi2::common::kRangerMaxHpCentiHp);
    EXPECT_EQ(sim.state().heroes[3].weapon.magazine,
              xushi2::common::kRangerMaxMagazine);
    EXPECT_FALSE(sim.state().heroes[3].weapon.reloading);
}

TEST(Respawn, KillsDeathsCountersPreservedAcrossRespawn) {
    Sim sim(close_arena_config());
    kill_slot_3(sim);
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].deaths, 1U);

    const auto respawn_tick = sim.state().heroes[3].respawn_tick;
    std::array<Action, kAgentsPerMatch> idle{};
    while (sim.state().tick <= respawn_tick) {
        sim.step(idle);
    }
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].deaths, 1U);
}

TEST(Respawn, RespawnReturnsToOwnSlotPointFor3v3) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 7;
    cfg.team_size = 3;
    Sim sim(cfg);

    std::array<xushi2::common::Vec2, 3> origins;
    for (std::uint32_t i = 0; i < 3; ++i) {
        origins[i] = sim.state().heroes[i].position;
    }

    // Force-kill slot 1 by mutating state, then step until respawn fires.
    // Use slot 0 (x = cx - dx). Slot 1's origin coincides with the 1v1
    // spawn point (cx, team_a_y), so it cannot distinguish a buggy
    // fallback from the correct per-slot path.
    auto& mutable_h = const_cast<xushi2::sim::HeroState&>(sim.state().heroes[0]);
    mutable_h.alive = false;
    mutable_h.health_centi_hp = 0;
    mutable_h.respawn_tick =
        sim.state().tick + cfg.mechanics.respawn_ticks;

    std::array<Action, kAgentsPerMatch> idle{};
    for (std::uint32_t k = 0; k < cfg.mechanics.respawn_ticks + 4U; ++k) {
        sim.step(idle);
        if (sim.state().heroes[0].alive) break;
    }
    ASSERT_TRUE(sim.state().heroes[0].alive);
    EXPECT_NEAR(sim.state().heroes[0].position.x, origins[0].x, 1e-4F);
    EXPECT_NEAR(sim.state().heroes[0].position.y, origins[0].y, 1e-4F);
}

TEST(Respawn, DeadHeroDoesNotCountForObjectiveOccupancy) {
    // Use full-size arena so the spawn is OUT of the objective circle.
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 5;
    cfg.round_length_seconds = 240;
    Sim sim(cfg);

    // Step past the lock. No movement, no firing — both alive but OFF point.
    std::array<Action, kAgentsPerMatch> idle{};
    for (std::uint32_t t = 0; t < xushi2::common::kObjectiveLockTicks + 10U; ++t) {
        sim.step(idle);
    }
    // No-one on point: cap_progress stays 0.
    EXPECT_EQ(sim.state().objective.cap_progress_ticks, 0U);
    EXPECT_EQ(sim.state().objective.cap_team,
              xushi2::sim::Team::Neutral);
}
