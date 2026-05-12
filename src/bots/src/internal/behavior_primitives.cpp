#include "behavior_primitives.h"

#include <cmath>

#include <xushi2/common/math.hpp>

namespace xushi2::bots::internal {

namespace {

constexpr float kObjectiveX = 25.0F;
constexpr float kObjectiveY = 25.0F;
constexpr float kArriveRadius = 0.25F;

}  // namespace

const sim::HeroState* find_opponent(const sim::MatchState& state,
                                    const sim::HeroState& self) {
    for (const auto& h : state.heroes) {
        if (!h.present || !h.alive) {
            continue;
        }
        if (h.team == self.team) {
            continue;
        }
        return &h;
    }
    return nullptr;
}

float aim_delta_toward(const sim::HeroState& self, float tx, float ty) {
    const float dx = tx - self.position.x;
    const float dy = ty - self.position.y;
    const float desired = std::atan2(dy, dx);
    const float raw = common::wrap_angle(desired - self.aim_angle);
    return common::clampf(raw, -common::kAimDeltaMax, common::kAimDeltaMax);
}

common::Action walk_to_objective(const sim::HeroState& self) {
    common::Action a{};
    const float dx = kObjectiveX - self.position.x;
    const float dy = kObjectiveY - self.position.y;
    const float dist2 = dx * dx + dy * dy;
    if (dist2 > kArriveRadius * kArriveRadius) {
        const float inv = 1.0F / std::sqrt(dist2);
        a.move_x = dx * inv;
        a.move_y = dy * inv;
    }
    return a;
}

common::Action hold_and_shoot(const sim::MatchState& state,
                              const sim::HeroState& self) {
    common::Action a{};
    const sim::HeroState* opp = find_opponent(state, self);
    if (opp == nullptr) {
        return a;
    }
    a.aim_delta = aim_delta_toward(self, opp->position.x, opp->position.y);
    a.primary_fire = true;
    return a;
}

}  // namespace xushi2::bots::internal
