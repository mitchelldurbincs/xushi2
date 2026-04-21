#include <xushi2/sim/obs_utils.h>

#include <cmath>

#include <xushi2/common/limits.hpp>

namespace xushi2::sim::obs_utils {

namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kTwoPi = 2.0F * kPi;

// Ranger max speed mirrors the private constant in sim.cpp. If the sim
// introduces a second moving hero with a different speed, bump this and
// expose a per-hero accessor. Phase 1 is 1v1 Ranger only.
constexpr float kRangerMaxSpeed = 4.2F;

common::Vec2 arena_center(const MapBounds& map) noexcept {
    return common::Vec2{0.5F * (map.min_x + map.max_x),
                        0.5F * (map.min_y + map.max_y)};
}

}  // namespace

common::Vec2 mirror_position_for_team(common::Vec2 world_pos,
                                      common::Team viewer_team,
                                      const MapBounds& map) noexcept {
    if (viewer_team != common::Team::B) {
        return world_pos;
    }
    const common::Vec2 c = arena_center(map);
    return common::Vec2{2.0F * c.x - world_pos.x,
                        2.0F * c.y - world_pos.y};
}

common::Vec2 mirror_velocity_for_team(common::Vec2 world_vel,
                                      common::Team viewer_team) noexcept {
    if (viewer_team != common::Team::B) {
        return world_vel;
    }
    return common::Vec2{-world_vel.x, -world_vel.y};
}

float mirror_angle_for_team(float world_angle_radians,
                            common::Team viewer_team) noexcept {
    if (viewer_team != common::Team::B) {
        return wrap_angle_pi(world_angle_radians);
    }
    return wrap_angle_pi(world_angle_radians + kPi);
}

common::Vec2 normalize_position_to_map(common::Vec2 pos,
                                       const MapBounds& map) noexcept {
    const float half_w = 0.5F * (map.max_x - map.min_x);
    const float half_h = 0.5F * (map.max_y - map.min_y);
    const common::Vec2 c = arena_center(map);
    const float nx = (half_w > 0.0F) ? (pos.x - c.x) / half_w : 0.0F;
    const float ny = (half_h > 0.0F) ? (pos.y - c.y) / half_h : 0.0F;
    return common::Vec2{nx, ny};
}

float normalize_bounded(float value, float lo, float hi) noexcept {
    const float span = hi - lo;
    if (span <= 0.0F) {
        return 0.0F;
    }
    return 2.0F * (value - lo) / span - 1.0F;
}

float wrap_angle_pi(float angle_radians) noexcept {
    // Pull into [-π, π] without looping on possibly-large inputs.
    float a = std::fmod(angle_radians + kPi, kTwoPi);
    if (a < 0.0F) {
        a += kTwoPi;
    }
    return a - kPi;
}

void angle_to_unit(float angle_radians, float* out_sincos2) noexcept {
    out_sincos2[0] = std::sin(angle_radians);
    out_sincos2[1] = std::cos(angle_radians);
}

VisibleEnemySlot visible_enemy_1v1(const MatchState& s,
                                   std::uint32_t viewer_slot) noexcept {
    VisibleEnemySlot out{};
    if (viewer_slot >= s.heroes.size()) {
        return out;
    }
    const auto& viewer = s.heroes[viewer_slot];
    if (!viewer.present || viewer.team == common::Team::Neutral) {
        return out;
    }
    const common::Team viewer_team = viewer.team;
    // Walk the fixed-size array once; no allocation. The first occupied
    // opposite-team hero is the enemy at Phase 1 (1v1 Ranger).
    for (std::uint32_t i = 0; i < s.heroes.size(); ++i) {
        const auto& h = s.heroes[i];
        if (!h.present) {
            continue;
        }
        if (h.team == viewer_team || h.team == common::Team::Neutral) {
            continue;
        }
        out.present = true;
        out.id = h.id;
        out.alive = h.alive;
        out.respawn_tick = h.respawn_tick;
        out.world_position = h.position;
        out.velocity = h.velocity;
        out.health_centi_hp = h.health_centi_hp;
        out.max_health_centi_hp = h.max_health_centi_hp;
        return out;
    }
    return out;
}

bool position_on_objective(common::Vec2 world_pos,
                           const MapBounds& map) noexcept {
    const common::Vec2 c = arena_center(map);
    const float dx = world_pos.x - c.x;
    const float dy = world_pos.y - c.y;
    return (dx * dx + dy * dy)
           <= (common::kObjectiveRadius * common::kObjectiveRadius);
}

float ranger_max_speed() noexcept {
    return kRangerMaxSpeed;
}

}  // namespace xushi2::sim::obs_utils
