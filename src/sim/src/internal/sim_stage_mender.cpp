#include "sim_stage_mender.h"

#include <cmath>

#include <xushi2/common/math.hpp>

#include "sim_combat.h"
#include "sim_movement_geometry.h"
#include "sim_target_query.h"

namespace xushi2::sim::internal {

static int slot_for_entity_id(const MatchState& state, common::EntityId id) {
    if (id == 0)
        return -1;
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        const HeroState& h = state.heroes[i];
        if (h.present && h.id == id)
            return (int)i;
    }
    return -1;
}

void stage_abilities_mender_weapon_swap(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& m = ctx.state.heroes[i];
        if (!m.present || !m.alive || m.kind != common::HeroKind::Mender)
            continue;
        const auto& a = ctx.actions[i];
        if (ctx.aim_consumed[i] || !a.ability_1 || m.cd_ability_1 != 0)
            continue;
        m.mender_weapon = (m.mender_weapon == common::MenderWeapon::Staff)
                              ? common::MenderWeapon::Sidearm
                              : common::MenderWeapon::Staff;
        m.mender_beam_locked_on = 0;
        m.cd_ability_1 = common::kMenderWeaponSwapCooldownTicks;
    }
}

void stage_mender_staff_beam(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& m = ctx.state.heroes[i];
        if (!m.present || !m.alive || m.kind != common::HeroKind::Mender ||
            m.mender_weapon != common::MenderWeapon::Staff || !ctx.actions[i].primary_fire) {
            m.mender_beam_locked_on = 0;
            continue;
        }
        int ts = slot_for_entity_id(ctx.state, m.mender_beam_locked_on);
        if (ts >= 0) {
            const HeroState& t = ctx.state.heroes[(std::uint32_t)ts];
            float dx = t.position.x - m.position.x, dy = t.position.y - m.position.y,
                  max = common::kMenderBeamRange * common::kMenderBeamRange;
            if (!t.alive || t.team != m.team || (dx * dx + dy * dy) > max ||
                segment_blocked_by_cover(m.position, t.position, ctx.config))
                ts = -1;
        }
        if (ts < 0) {
            ts = nearest_ally_in_cone_with_los(ctx.state,
                                               i,
                                               common::kMenderBeamRange,
                                               common::kMenderBeamHalfAngleCos,
                                               ctx.config);
            m.mender_beam_locked_on = (ts >= 0) ? ctx.state.heroes[(std::uint32_t)ts].id : 0;
        }
        if (ts < 0)
            continue;
        HeroState& t = ctx.state.heroes[(std::uint32_t)ts];
        t.health_centi_hp = std::min<std::int32_t>(
            t.max_health_centi_hp, t.health_centi_hp + common::kMenderBeamHealCentiHpPerTick);
    }
}

void stage_abilities_mender_tether(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& m = ctx.state.heroes[i];
        if (!m.present || !m.alive || m.kind != common::HeroKind::Mender)
            continue;
        const auto& a = ctx.actions[i];
        if (ctx.aim_consumed[i] || !a.ability_2 || m.cd_ability_2 != 0)
            continue;
        int ts = nearest_ally_in_cone_with_los(
            ctx.state, i, common::kMenderTetherRange, common::kMenderBeamHalfAngleCos, ctx.config);
        if (ts < 0)
            continue;
        const HeroState& t = ctx.state.heroes[(std::uint32_t)ts];
        common::Vec2 from{m.position.x - t.position.x, m.position.y - t.position.y};
        float ds = from.x * from.x + from.y * from.y;
        if (ds <= 1e-6F)
            continue;
        float inv = 1.0F / std::sqrt(ds);
        common::Vec2 next{t.position.x + from.x * inv * common::kMenderTetherStopDistance,
                          t.position.y + from.y * inv * common::kMenderTetherStopDistance};
        // Intentionally delegate movement collision/bounds handling to shared geometry helper.
        m.position = resolve_displaced_position(m.position, next, from, ctx.config);
        m.cd_ability_2 = common::kMenderTetherCooldownTicks;
    }
}
}  // namespace xushi2::sim::internal
