#include "behavior_primitives.h"

#include <cmath>

#include <xushi2/common/math.hpp>
#include <xushi2/sim/sim.h>

namespace xushi2::bots::internal {

namespace {

constexpr float kArriveRadius = 0.25F;

common::Vec2 objective_center(const sim::MapBounds& map) {
    return common::Vec2{0.5F * (map.min_x + map.max_x),
                        0.5F * (map.min_y + map.max_y)};
}

bool observable_enemy(const sim::MatchState&,
                      const sim::HeroState& self,
                      const sim::HeroState& candidate,
                      const sim::MatchConfig& config) {
    return candidate.present && candidate.alive && candidate.team != self.team &&
           candidate.team != common::Team::Neutral &&
           sim::line_of_sight(self.position, candidate.position, config);
}

}  // namespace

const sim::HeroState* find_opponent(const sim::MatchState& state,
                                    const sim::HeroState& self,
                                    const sim::MatchConfig& config) {
    const sim::HeroState* best = nullptr;
    float best_dist2 = 0.0F;
    for (const auto& h : state.heroes) {
        if (!observable_enemy(state, self, h, config)) {
            continue;
        }
        const float dx = h.position.x - self.position.x;
        const float dy = h.position.y - self.position.y;
        const float d2 = dx * dx + dy * dy;
        if (best == nullptr || d2 < best_dist2) {
            best = &h;
            best_dist2 = d2;
        }
    }
    return best;
}

float aim_delta_toward(const sim::HeroState& self, float tx, float ty,
                       float noise_radians) {
    const float dx = tx - self.position.x;
    const float dy = ty - self.position.y;
    const float desired = std::atan2(dy, dx);
    const float raw = common::wrap_angle(desired - self.aim_angle + noise_radians);
    return common::clampf(raw, -common::kAimDeltaMax, common::kAimDeltaMax);
}

common::Action walk_to_objective(const sim::HeroState& self,
                                 const sim::MapBounds& map) {
    common::Action a{};
    const common::Vec2 center = objective_center(map);
    const float dx = center.x - self.position.x;
    const float dy = center.y - self.position.y;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 > kArriveRadius * kArriveRadius) {
        const float inv = 1.0F / std::sqrt(dist2);
        a.move_x = dx * inv;
        a.move_y = dy * inv;
    }
    return a;
}

common::Action hold_and_shoot(const sim::MatchState& state,
                              const sim::HeroState& self,
                              const sim::MatchConfig& config,
                              float aim_noise_radians) {
    common::Action a{};
    const sim::HeroState* opp = find_opponent(state, self, config);
    if (opp == nullptr) {
        return a;
    }
    a.aim_delta = aim_delta_toward(self, opp->position.x, opp->position.y,
                                   aim_noise_radians);
    a.primary_fire = true;
    return a;
}

}  // namespace xushi2::bots::internal
