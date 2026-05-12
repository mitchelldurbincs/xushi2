#include "sim_movement_geometry.h"

#include <algorithm>
#include <cmath>

#include <xushi2/common/math.hpp>

namespace xushi2::sim::internal {

float cross(common::Vec2 a, common::Vec2 b) { return a.x * b.y - a.y * b.x; }

bool movement_crosses_wall(common::Vec2 from, common::Vec2 to,
                           const WallSegment& wall) {
    if (wall.half_width <= 0.0F || !std::isfinite(wall.half_width)) {
        return false;
    }
    const common::Vec2 move{to.x - from.x, to.y - from.y};
    const common::Vec2 wall_vec{wall.b.x - wall.a.x, wall.b.y - wall.a.y};
    if ((move.x * move.x + move.y * move.y) <= 1e-8F ||
        (wall_vec.x * wall_vec.x + wall_vec.y * wall_vec.y) <= 1e-8F) {
        return false;
    }
    const float denom = cross(move, wall_vec);
    if (std::fabs(denom) <= 1e-6F) {
        return false;
    }
    const common::Vec2 wall_from{wall.a.x - from.x, wall.a.y - from.y};
    const float move_t = cross(wall_from, wall_vec) / denom;
    const float wall_t = cross(wall_from, move) / denom;
    return move_t >= 0.0F && move_t <= 1.0F && wall_t >= 0.0F && wall_t <= 1.0F;
}

common::Vec2 prevent_wall_crossing(common::Vec2 from, common::Vec2 to,
                                   const MatchConfig& config) {
    const std::uint32_t num_walls =
        std::min<std::uint32_t>(config.num_wall_segments, common::kMaxWalls);
    for (std::uint32_t i = 0; i < num_walls; ++i) {
        if (movement_crosses_wall(from, to, config.wall_segments[i])) {
            return from;
        }
    }
    return to;
}

common::Vec2 resolve_cover_overlap(common::Vec2 p, common::Vec2 fallback_dir,
                                   const MatchConfig& config) {
    const std::uint32_t n =
        std::min<std::uint32_t>(config.num_cover_circles, common::kMaxWalls);
    for (std::uint32_t i = 0; i < n; ++i) {
        const CoverCircle& cover = config.cover_circles[i];
        if (cover.radius <= 0.0F || !std::isfinite(cover.radius)) {
            continue;
        }
        common::Vec2 delta{p.x - cover.center.x, p.y - cover.center.y};
        float dist_sq = delta.x * delta.x + delta.y * delta.y;
        if (dist_sq >= cover.radius * cover.radius) {
            continue;
        }
        if (dist_sq <= 1e-6F) {
            delta = fallback_dir;
            dist_sq = delta.x * delta.x + delta.y * delta.y;
            if (dist_sq <= 1e-6F) {
                delta = common::Vec2{1.0F, 0.0F};
                dist_sq = 1.0F;
            }
        }
        const float inv_dist = 1.0F / std::sqrt(dist_sq);
        p = common::Vec2{cover.center.x + delta.x * inv_dist * cover.radius,
                         cover.center.y + delta.y * inv_dist * cover.radius};
    }
    const std::uint32_t num_walls =
        std::min<std::uint32_t>(config.num_wall_segments, common::kMaxWalls);
    for (std::uint32_t i = 0; i < num_walls; ++i) {
        const WallSegment& wall = config.wall_segments[i];
        if (wall.half_width <= 0.0F || !std::isfinite(wall.half_width)) {
            continue;
        }
        const common::Vec2 ab{wall.b.x - wall.a.x, wall.b.y - wall.a.y};
        const float len_sq = ab.x * ab.x + ab.y * ab.y;
        if (len_sq <= 1e-6F) {
            continue;
        }
        const float t = std::clamp(((p.x - wall.a.x) * ab.x + (p.y - wall.a.y) * ab.y) / len_sq,
                                   0.0F, 1.0F);
        const common::Vec2 nearest{wall.a.x + ab.x * t, wall.a.y + ab.y * t};
        common::Vec2 delta{p.x - nearest.x, p.y - nearest.y};
        float dist_sq = delta.x * delta.x + delta.y * delta.y;
        if (dist_sq >= wall.half_width * wall.half_width) {
            continue;
        }
        if (dist_sq <= 1e-6F) {
            delta = common::Vec2{-ab.y, ab.x};
            dist_sq = delta.x * delta.x + delta.y * delta.y;
            if (dist_sq <= 1e-6F) {
                delta = fallback_dir;
                dist_sq = delta.x * delta.x + delta.y * delta.y;
            }
            if (dist_sq <= 1e-6F) {
                delta = common::Vec2{1.0F, 0.0F};
                dist_sq = 1.0F;
            }
        }
        const float inv_dist = 1.0F / std::sqrt(dist_sq);
        p = common::Vec2{nearest.x + delta.x * inv_dist * wall.half_width,
                         nearest.y + delta.y * inv_dist * wall.half_width};
    }
    p.x = common::clampf(p.x, config.map.min_x, config.map.max_x);
    p.y = common::clampf(p.y, config.map.min_y, config.map.max_y);
    return p;
}

common::Vec2 resolve_displaced_position(common::Vec2 current_pos, common::Vec2 intended_next_pos,
                                      common::Vec2 cover_fallback_dir,
                                      const MatchConfig& config) {
    intended_next_pos.x =
        common::clampf(intended_next_pos.x, config.map.min_x, config.map.max_x);
    intended_next_pos.y =
        common::clampf(intended_next_pos.y, config.map.min_y, config.map.max_y);
    intended_next_pos = prevent_wall_crossing(current_pos, intended_next_pos, config);
    return resolve_cover_overlap(intended_next_pos, cover_fallback_dir, config);
}

}  // namespace xushi2::sim::internal
