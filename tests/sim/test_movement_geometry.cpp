#include <gtest/gtest.h>

#include "../../src/sim/src/internal/sim_movement_geometry.h"

namespace {
using namespace xushi2::sim;
using namespace xushi2::sim::internal;

TEST(MovementGeometry, ParallelMovementDoesNotCrossWall) {
    WallSegment wall{}; wall.a={0.0F,0.0F}; wall.b={10.0F,0.0F}; wall.half_width=0.5F;
    EXPECT_FALSE(movement_crosses_wall({0.0F,1.0F},{10.0F,1.0F}, wall));
}

TEST(MovementGeometry, ZeroLengthSegmentDoesNotCrossWall) {
    WallSegment wall{}; wall.a={0.0F,0.0F}; wall.b={10.0F,0.0F}; wall.half_width=0.5F;
    EXPECT_FALSE(movement_crosses_wall({1.0F,1.0F},{1.0F,1.0F}, wall));
}

TEST(MovementGeometry, InsideCoverPushesOutUsingFallback) {
    MatchConfig cfg{}; cfg.map.min_x=-100; cfg.map.min_y=-100; cfg.map.max_x=100; cfg.map.max_y=100;
    cfg.num_cover_circles=1; cfg.cover_circles[0].center={0.0F,0.0F}; cfg.cover_circles[0].radius=2.0F;
    xushi2::common::Vec2 p = resolve_cover_overlap({0.0F,0.0F}, {0.0F,1.0F}, cfg);
    EXPECT_NEAR(p.x, 0.0F, 1e-4F);
    EXPECT_NEAR(p.y, 2.0F, 1e-4F);
}

} // namespace
