#include "sim_tick_stages.h"

#include <cmath>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/math.hpp>

#include "sim_movement_geometry.h"
#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

static constexpr float kVanguardSpeed = 3.6F;
static constexpr float kRangerSpeed = 4.2F;
static constexpr float kMenderSpeed = 4.0F;

float hero_speed(common::HeroKind kind) {
    switch (kind) {
        case common::HeroKind::Vanguard:
            return kVanguardSpeed;
        case common::HeroKind::Ranger:
            return kRangerSpeed;
        case common::HeroKind::Mender:
            return kMenderSpeed;
    }
    X2_UNREACHABLE();
}

void stage_validate_and_aim(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = ctx.state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        const common::Action& a = ctx.actions[i];
        X2_INVARIANT(std::isfinite(a.move_x) && std::isfinite(a.move_y), common::ErrorCode::NonFiniteFloat);
        X2_INVARIANT(std::isfinite(a.aim_delta), common::ErrorCode::NonFiniteFloat);
        if (!ctx.aim_consumed[i]) {
            h.aim_angle = common::wrap_angle(h.aim_angle + common::clampf(a.aim_delta, -common::kAimDeltaMax, common::kAimDeltaMax));
        }
    }
}

void stage_movement_and_bounds(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = ctx.state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }

        common::Vec2 move_vec = common::normalize_move_input(common::Vec2{ctx.actions[i].move_x, ctx.actions[i].move_y});
        float speed = hero_speed(h.kind);
        if (h.kind == common::HeroKind::Vanguard && h.vanguard_barrier_active) {
            speed *= 0.7F;
        }

        h.velocity = common::scale(move_vec, speed);
        common::Vec2 next = common::add(h.position, common::scale(h.velocity, kDt));
        next.x = common::clampf(next.x, ctx.config.map.min_x, ctx.config.map.max_x);
        next.y = common::clampf(next.y, ctx.config.map.min_y, ctx.config.map.max_y);
        next = prevent_wall_crossing(h.position, next, ctx.config);
        h.position = resolve_cover_overlap(next, move_vec, ctx.config);
    }
}

void stage_cooldowns_and_weapon_tick(MatchState& state) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present) {
            continue;
        }

        if (h.cd_ability_1 > 0) {
            --h.cd_ability_1;
        }
        if (h.cd_ability_2 > 0) {
            --h.cd_ability_2;
        }
        if (h.ranger_marked_ticks > 0) {
            --h.ranger_marked_ticks;
            if (h.ranger_marked_ticks == 0) {
                h.ranger_marked_by = common::Team::Neutral;
            }
        }

        if (h.alive && h.kind == common::HeroKind::Ranger) {
            weapon_tick_update(h.weapon);
        } else if (h.alive && (h.kind == common::HeroKind::Vanguard || h.kind == common::HeroKind::Mender) &&
                   h.weapon.fire_cooldown_ticks > 0) {
            --h.weapon.fire_cooldown_ticks;
        }
    }
}

}  // namespace xushi2::sim::internal
