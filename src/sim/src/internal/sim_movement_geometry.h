#pragma once

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

float cross(common::Vec2 a, common::Vec2 b);
bool movement_crosses_wall(common::Vec2 from, common::Vec2 to, const WallSegment& wall);
common::Vec2 prevent_wall_crossing(common::Vec2 from, common::Vec2 to, const MatchConfig& config);
common::Vec2 resolve_cover_overlap(common::Vec2 p, common::Vec2 fallback_dir,
                                   const MatchConfig& config);

}  // namespace xushi2::sim::internal
