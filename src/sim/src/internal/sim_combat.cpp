#include "sim_combat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

// --- Tick-pipeline step 10: hitscan fire resolution. ---
//
// Writes DamageEvents into the per-tick buffer; applies no HP changes.

// Ray–circle intersection. Returns t >= 0 at which the ray enters the
// circle, or a negative value if the ray misses / is past the circle. The
// ray is (origin + t*d), d assumed unit length.
static float ray_circle_hit_t(common::Vec2 origin, common::Vec2 d, common::Vec2 center, float radius) {
    const common::Vec2 oc{origin.x - center.x, origin.y - center.y};
    const float b = oc.x * d.x + oc.y * d.y;
    const float c_term = oc.x * oc.x + oc.y * oc.y - radius * radius;
    const float disc = b * b - c_term;
    if (disc < 0.0F) {
        return -1.0F;
    }
    const float sqrt_disc = std::sqrt(disc);
    const float t_near = -b - sqrt_disc;
    if (t_near >= 0.0F) {
        return t_near;
    }
    const float t_far = -b + sqrt_disc;
    if (t_far >= 0.0F) {
        return 0.0F;  // origin inside circle
    }
    return -1.0F;
}

void resolve_revolver_fire(MatchState& state,
                           const std::array<common::Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
                           DamageBuffer& buf,
                           std::array<bool, kAgentsPerMatch>& has_damage) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& shooter = state.heroes[i];
        if (!shooter.present || !shooter.alive) {
            continue;
        }
        if (shooter.kind != common::HeroKind::Ranger) {
            continue;  // Phase 1: only Rangers fire
        }
        const common::Action& a = actions[i];
        if (!a.primary_fire) {
            continue;
        }
        if (shooter.weapon.reloading || shooter.weapon.magazine == 0 ||
            shooter.weapon.fire_cooldown_ticks > 0) {
            continue;
        }
        // Hitscan ray.
        const common::Vec2 d{std::cos(shooter.aim_angle), std::sin(shooter.aim_angle)};
        float best_t = std::numeric_limits<float>::infinity();
        int best_slot = -1;
        for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
            const HeroState& target = state.heroes[j];
            if (!target.present || !target.alive) {
                continue;
            }
            if (target.team == shooter.team) {
                continue;  // no friendly fire
            }
            const float t =
                ray_circle_hit_t(shooter.position, d, target.position, m.revolver_hitbox_radius);
            if (t < 0.0F || t > common::kRangerRevolverRange) {
                continue;
            }
            // Deterministic tie-break by slot index (stable).
            if (t < best_t) {
                best_t = t;
                best_slot = static_cast<int>(j);
            }
        }
        // Magazine decrement + fire-gate arm happen regardless of hit/miss.
        weapon_on_fire_success(shooter.weapon, m);
        if (best_slot >= 0) {
            buf[i].attacker_id = shooter.id;
            buf[i].victim_slot = static_cast<std::uint32_t>(best_slot);
            buf[i].damage_centi_hp = m.revolver_damage_centi_hp;
            has_damage[i] = true;
        }
    }
}

// --- Tick-pipeline steps 11–12: apply accumulated damage simultaneously. ---

void apply_damage_buffer(MatchState& state, const DamageBuffer& buf,
                         const std::array<bool, kAgentsPerMatch>& has_damage) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        if (!has_damage[i]) {
            continue;
        }
        const DamageEvent& ev = buf[i];
        HeroState& victim = state.heroes[ev.victim_slot];
        X2_INVARIANT(victim.present, common::ErrorCode::CorruptState);
        if (!victim.alive) {
            continue;  // already dead this tick from earlier-slot attacker
        }
        const std::int32_t damage = static_cast<std::int32_t>(ev.damage_centi_hp);
        victim.health_centi_hp = std::max<std::int32_t>(0, victim.health_centi_hp - damage);
    }
}

// --- Tick-pipeline step 13: process deaths. ---

void process_deaths(MatchState& state, const DamageBuffer& buf,
                    const std::array<bool, kAgentsPerMatch>& has_damage,
                    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        if (h.health_centi_hp > 0) {
            continue;
        }
        h.alive = false;
        h.respawn_tick = state.tick + config.mechanics.respawn_ticks;
        h.deaths += 1;
    }
    // Credit kills: any attacker whose victim just died this tick gets +1.
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        if (!has_damage[i]) {
            continue;
        }
        const HeroState& victim = state.heroes[buf[i].victim_slot];
        if (victim.alive) {
            continue;  // didn't die from this damage
        }
        // Attacker is slot i.
        state.heroes[i].kills += 1;
    }
}

}  // namespace xushi2::sim::internal
