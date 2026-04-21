#include <gtest/gtest.h>

#include <cmath>

#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

namespace {

using xushi2::common::Team;
using xushi2::common::Vec2;
using xushi2::sim::MapBounds;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;
namespace ou = xushi2::sim::obs_utils;

constexpr float kEps = 1e-5F;

TEST(ObsUtils, MirrorPositionNoopForTeamA) {
    MapBounds map{};  // default 0..50 × 0..50
    Vec2 p{10.0F, 3.0F};
    Vec2 r = ou::mirror_position_for_team(p, Team::A, map);
    EXPECT_NEAR(r.x, 10.0F, kEps);
    EXPECT_NEAR(r.y, 3.0F, kEps);
}

TEST(ObsUtils, MirrorPositionFlipsForTeamB) {
    MapBounds map{};
    Vec2 p{10.0F, 3.0F};
    Vec2 r = ou::mirror_position_for_team(p, Team::B, map);
    // Arena center is (25, 25); 180° around center sends (10, 3) -> (40, 47).
    EXPECT_NEAR(r.x, 40.0F, kEps);
    EXPECT_NEAR(r.y, 47.0F, kEps);
}

TEST(ObsUtils, MirrorPositionIsIdempotent) {
    MapBounds map{};
    Vec2 p{7.5F, 11.25F};
    Vec2 r = ou::mirror_position_for_team(
        ou::mirror_position_for_team(p, Team::B, map), Team::B, map);
    EXPECT_NEAR(r.x, p.x, kEps);
    EXPECT_NEAR(r.y, p.y, kEps);
}

TEST(ObsUtils, MirrorVelocityFlipsBothAxesForTeamB) {
    Vec2 v{1.0F, -2.0F};
    Vec2 r = ou::mirror_velocity_for_team(v, Team::B);
    EXPECT_NEAR(r.x, -1.0F, kEps);
    EXPECT_NEAR(r.y, 2.0F, kEps);
}

TEST(ObsUtils, MirrorAngleFlipsForTeamB) {
    // angle 0 maps to +π then wraps to -π (or +π — both are valid
    // representations of the same direction). The mirrored unit vector must
    // equal the input reversed.
    const float a = 0.25F;  // ~14.3°
    const float m = ou::mirror_angle_for_team(a, Team::B);
    float u0[2];
    float um[2];
    ou::angle_to_unit(a, u0);
    ou::angle_to_unit(m, um);
    EXPECT_NEAR(u0[0], -um[0], kEps);
    EXPECT_NEAR(u0[1], -um[1], kEps);
}

TEST(ObsUtils, NormalizePositionEndpointsMapToPlusMinusOne) {
    MapBounds map{};  // 0..50
    const Vec2 lo = ou::normalize_position_to_map(Vec2{0.0F, 0.0F}, map);
    const Vec2 hi = ou::normalize_position_to_map(Vec2{50.0F, 50.0F}, map);
    const Vec2 mid = ou::normalize_position_to_map(Vec2{25.0F, 25.0F}, map);
    EXPECT_NEAR(lo.x, -1.0F, kEps);
    EXPECT_NEAR(lo.y, -1.0F, kEps);
    EXPECT_NEAR(hi.x,  1.0F, kEps);
    EXPECT_NEAR(hi.y,  1.0F, kEps);
    EXPECT_NEAR(mid.x, 0.0F, kEps);
    EXPECT_NEAR(mid.y, 0.0F, kEps);
}

TEST(ObsUtils, NormalizeBoundedBasic) {
    EXPECT_NEAR(ou::normalize_bounded(0.0F, 0.0F, 10.0F), -1.0F, kEps);
    EXPECT_NEAR(ou::normalize_bounded(10.0F, 0.0F, 10.0F), 1.0F, kEps);
    EXPECT_NEAR(ou::normalize_bounded(5.0F, 0.0F, 10.0F), 0.0F, kEps);
    EXPECT_EQ(ou::normalize_bounded(3.0F, 5.0F, 5.0F), 0.0F);  // degenerate span
}

TEST(ObsUtils, WrapAnglePiKeepsInRange) {
    EXPECT_NEAR(ou::wrap_angle_pi(0.0F), 0.0F, kEps);
    EXPECT_NEAR(ou::wrap_angle_pi(3.14159F), 3.14159F, 1e-3F);
    EXPECT_NEAR(ou::wrap_angle_pi(-3.14159F), -3.14159F, 1e-3F);
    // 3π wraps to π (or -π); unit-vector conversion is the same either way.
    const float wrapped = ou::wrap_angle_pi(3.0F * 3.14159265F);
    EXPECT_LE(std::abs(wrapped), 3.14159265F + 1e-3F);
}

TEST(ObsUtils, AngleToUnitIsUnitLength) {
    float u[2];
    ou::angle_to_unit(1.234F, u);
    const float mag = std::sqrt(u[0] * u[0] + u[1] * u[1]);
    EXPECT_NEAR(mag, 1.0F, kEps);
}

TEST(ObsUtils, VisibleEnemyForTeamASeesSlotThree) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    // Slot 0 is Team A Ranger, slot 3 is Team B Ranger (see sim.cpp reset).
    const auto& state = sim.state();
    auto e = ou::visible_enemy_1v1(state, 0);
    EXPECT_TRUE(e.present);
    EXPECT_TRUE(e.alive);
    EXPECT_EQ(e.id, state.heroes[3].id);
    EXPECT_NEAR(e.world_position.x, state.heroes[3].position.x, kEps);
    EXPECT_NEAR(e.world_position.y, state.heroes[3].position.y, kEps);
}

TEST(ObsUtils, VisibleEnemyForTeamBSeesSlotZero) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto& state = sim.state();
    auto e = ou::visible_enemy_1v1(state, 3);
    EXPECT_TRUE(e.present);
    EXPECT_EQ(e.id, state.heroes[0].id);
}

TEST(ObsUtils, VisibleEnemyAbsentSlotReturnsFalse) {
    MatchConfig cfg = xushi2::test_support::make_test_config();
    Sim sim(cfg);
    const auto& state = sim.state();
    // Slot 1 is unoccupied at Phase 1.
    auto e = ou::visible_enemy_1v1(state, 1);
    // viewer's team is Neutral -> no opposite; present == false.
    EXPECT_FALSE(e.present);
}

TEST(ObsUtils, PositionOnObjectiveDetectsCenter) {
    MapBounds map{};  // center (25, 25)
    EXPECT_TRUE(ou::position_on_objective(Vec2{25.0F, 25.0F}, map));
    EXPECT_TRUE(ou::position_on_objective(Vec2{26.5F, 25.0F}, map));
    // Radius is 3.0u; far corner is definitely off-point.
    EXPECT_FALSE(ou::position_on_objective(Vec2{0.0F, 0.0F}, map));
}

TEST(ObsUtils, RangerMaxSpeedIsPositive) {
    EXPECT_GT(ou::ranger_max_speed(), 0.0F);
}

}  // namespace
