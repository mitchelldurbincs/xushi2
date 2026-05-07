#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <test_config.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a config validation: the Sim ctor must abort when any required
// mechanics value is missing/invalid. No silent defaults.
//
// Uses gtest's death tests — a process fork that checks the child aborts.

namespace {

using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

}  // namespace

TEST(ConfigValidation, DefaultConstructedMatchConfigRejected) {
    // A bare MatchConfig{} has every mechanics field at its sentinel.
    MatchConfig bad_cfg{};
    EXPECT_DEATH({ Sim sim(bad_cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, MissingRevolverDamageRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_damage_centi_hp =
        std::numeric_limits<std::uint32_t>::max();
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, MissingFireCooldownRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_fire_cooldown_ticks =
        std::numeric_limits<std::uint32_t>::max();
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, MissingHitboxRadiusRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_hitbox_radius = std::numeric_limits<float>::quiet_NaN();
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, MissingRespawnTicksRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.respawn_ticks = std::numeric_limits<std::uint32_t>::max();
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, ZeroDamageRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_damage_centi_hp = 0U;
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, ZeroFireCooldownRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_fire_cooldown_ticks = 0U;
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, NegativeHitboxRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.revolver_hitbox_radius = -1.0F;
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, ZeroRespawnTicksRejected) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.mechanics.respawn_ticks = 0U;
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(ConfigValidation, ValidConfigConstructs) {
    auto cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);  // should not abort
    EXPECT_EQ(sim.state().tick, 0U);
}

TEST(MatchConfig, DefaultTeamSizeIsOne) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    EXPECT_EQ(cfg.team_size, 1U);
}

TEST(MatchConfig, TeamSizeThreeIsAccepted) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    Sim sim(cfg);  // should not abort
    EXPECT_EQ(sim.state().tick, 0U);
}

TEST(MatchConfig, TeamSizeTwoIsRejected) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 2;
    EXPECT_DEATH({ Sim sim(cfg); (void)sim; }, "REQUIRE");
}

TEST(Spawn3v3, ProducesSixPresentHeroesAtDistinctPositions) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    cfg.team_size = 3;
    Sim sim(cfg);
    const auto& heroes = sim.state().heroes;

    for (std::uint32_t i = 0; i < 6; ++i) {
        EXPECT_TRUE(heroes[i].present) << "slot " << i;
        EXPECT_TRUE(heroes[i].alive) << "slot " << i;
    }
    for (std::uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(heroes[i].team, xushi2::common::Team::A) << "slot " << i;
    }
    for (std::uint32_t i = 3; i < 6; ++i) {
        EXPECT_EQ(heroes[i].team, xushi2::common::Team::B) << "slot " << i;
    }
    for (std::uint32_t i = 0; i < 6; ++i) {
        for (std::uint32_t j = i + 1; j < 6; ++j) {
            const float dx = heroes[i].position.x - heroes[j].position.x;
            const float dy = heroes[i].position.y - heroes[j].position.y;
            EXPECT_GT(dx*dx + dy*dy, 1e-3F)
                << "slots " << i << " and " << j;
        }
    }
}

TEST(Spawn3v3, TeamSizeOnePathIsUnchanged) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto& heroes = sim.state().heroes;
    EXPECT_TRUE(heroes[0].present);
    EXPECT_TRUE(heroes[3].present);
    for (std::uint32_t i : {1u, 2u, 4u, 5u}) {
        EXPECT_FALSE(heroes[i].present) << "slot " << i;
    }
}
