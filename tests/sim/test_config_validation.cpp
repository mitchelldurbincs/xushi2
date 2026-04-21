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
