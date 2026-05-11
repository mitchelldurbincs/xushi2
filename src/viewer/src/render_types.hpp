#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

struct ArenaTransform {
    float world_min_x;
    float world_min_y;
    float world_w;
    float world_h;
    float pixels_per_unit;
    float screen_origin_x;
    float screen_origin_y;
};

struct ShotTracer {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

struct TetherTrail {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

struct LosDebugCounts {
    std::size_t visible = 0;
    std::size_t blocked = 0;
};
