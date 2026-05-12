#include "sim_combat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

#include "sim_target_query.h"
#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

// --- Tick-pipeline step 10: hitscan fire resolution. ---
//
// Writes DamageEvents into the per-tick buffer; applies no HP changes.

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
        return 0.0F;
    }
    return -1.0F;
}

static float cross(common::Vec2 a, common::Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

static float ray_segment_hit_t(common::Vec2 origin,
                               common::Vec2 d,
                               common::Vec2 a,
                               common::Vec2 b) {
    const common::Vec2 s{b.x - a.x, b.y - a.y};
    const float denom = cross(d, s);
    if (std::abs(denom) <= 1e-6F) {
        return -1.0F;
    }
    const common::Vec2 ao{a.x - origin.x, a.y - origin.y};
    const float t = cross(ao, s) / denom;
    const float u = cross(ao, d) / denom;
    if (t < 0.0F || u < 0.0F || u > 1.0F) {
        return -1.0F;
    }
    return t;
}

static float point_segment_distance_sq(common::Vec2 p, common::Vec2 a, common::Vec2 b) {
    const common::Vec2 ab{b.x - a.x, b.y - a.y};
    const float len_sq = ab.x * ab.x + ab.y * ab.y;
    if (len_sq <= 1e-6F) {
        const float dx = p.x - a.x;
        const float dy = p.y - a.y;
        return dx * dx + dy * dy;
    }
    const float t = std::clamp(
        ((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len_sq,
        0.0F,
        1.0F);
    const common::Vec2 q{a.x + ab.x * t, a.y + ab.y * t};
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    return dx * dx + dy * dy;
}

static float segment_distance_sq(common::Vec2 a0,
                                 common::Vec2 a1,
                                 common::Vec2 b0,
                                 common::Vec2 b1) {
    const common::Vec2 da{a1.x - a0.x, a1.y - a0.y};
    const common::Vec2 db{b1.x - b0.x, b1.y - b0.y};
    if (std::abs(cross(da, db)) > 1e-6F) {
        const common::Vec2 b0a0{b0.x - a0.x, b0.y - a0.y};
        const float t = cross(b0a0, db) / cross(da, db);
        const float u = cross(b0a0, da) / cross(da, db);
        if (t >= 0.0F && t <= 1.0F && u >= 0.0F && u <= 1.0F) {
            return 0.0F;
        }
    }
    return std::min({
        point_segment_distance_sq(a0, b0, b1),
        point_segment_distance_sq(a1, b0, b1),
        point_segment_distance_sq(b0, a0, a1),
        point_segment_distance_sq(b1, a0, a1),
    });
}

float first_cover_hit_t(common::Vec2 origin,
                        common::Vec2 direction_unit,
                        float max_t,
                        const MatchConfig& config) {
    float best_t = std::numeric_limits<float>::infinity();
    const std::uint32_t n =
        std::min<std::uint32_t>(config.num_cover_circles, common::kMaxWalls);
    for (std::uint32_t i = 0; i < n; ++i) {
        const CoverCircle& cover = config.cover_circles[i];
        if (cover.radius <= 0.0F || !std::isfinite(cover.radius)) {
            continue;
        }
        const float t =
            ray_circle_hit_t(origin, direction_unit, cover.center, cover.radius);
        if (t < 0.0F || t > max_t) {
            continue;
        }
        if (t < best_t) {
            best_t = t;
        }
    }
    const std::uint32_t num_walls =
        std::min<std::uint32_t>(config.num_wall_segments, common::kMaxWalls);
    for (std::uint32_t i = 0; i < num_walls; ++i) {
        const WallSegment& wall = config.wall_segments[i];
        if (wall.half_width <= 0.0F || !std::isfinite(wall.half_width)) {
            continue;
        }
        const float t_center = ray_segment_hit_t(origin, direction_unit, wall.a, wall.b);
        if (t_center >= 0.0F && t_center <= max_t && t_center < best_t) {
            best_t = t_center;
        }
        const float t_a = ray_circle_hit_t(origin, direction_unit, wall.a, wall.half_width);
        if (t_a >= 0.0F && t_a <= max_t && t_a < best_t) {
            best_t = t_a;
        }
        const float t_b = ray_circle_hit_t(origin, direction_unit, wall.b, wall.half_width);
        if (t_b >= 0.0F && t_b <= max_t && t_b < best_t) {
            best_t = t_b;
        }
    }
    return best_t;
}

bool segment_blocked_by_cover(common::Vec2 a,
                              common::Vec2 b,
                              const MatchConfig& config) {
    const common::Vec2 delta{b.x - a.x, b.y - a.y};
    const float len_sq = delta.x * delta.x + delta.y * delta.y;
    if (len_sq <= 1e-6F) {
        return false;
    }
    const float len = std::sqrt(len_sq);
    const common::Vec2 d{delta.x / len, delta.y / len};
    if (first_cover_hit_t(a, d, len, config) <= len) {
        return true;
    }
    const std::uint32_t num_walls =
        std::min<std::uint32_t>(config.num_wall_segments, common::kMaxWalls);
    for (std::uint32_t i = 0; i < num_walls; ++i) {
        const WallSegment& wall = config.wall_segments[i];
        if (wall.half_width <= 0.0F || !std::isfinite(wall.half_width)) {
            continue;
        }
        if (segment_distance_sq(a, b, wall.a, wall.b) <=
            wall.half_width * wall.half_width) {
            return true;
        }
    }
    return false;
}

void resolve_revolver_fire(MatchState& state,
                           const std::array<common::Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
                           const MatchConfig& config,
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
        const RayHitCandidate hit = first_ray_hit_enemy_or_barrier(
            state, i, d, common::kRangerRevolverRange, m.revolver_hitbox_radius, config);
        // Magazine decrement + fire-gate arm happen regardless of hit/miss.
        weapon_on_fire_success(shooter.weapon, m);
        if (hit.kind == RayHitCandidate::Kind::Barrier && hit.slot >= 0) {
            HeroState& barrier_owner = state.heroes[static_cast<std::uint32_t>(hit.slot)];
            barrier_owner.vanguard_barrier_hp_centi = std::max<std::int32_t>(
                0,
                barrier_owner.vanguard_barrier_hp_centi -
                    static_cast<std::int32_t>(m.revolver_damage_centi_hp));
            if (barrier_owner.vanguard_barrier_hp_centi == 0) {
                barrier_owner.vanguard_barrier_active = false;
                barrier_owner.cd_ability_1 =
                    common::kVanguardBarrierBrokenCooldownTicks;
            }
            continue;
        }
        if (hit.kind == RayHitCandidate::Kind::Enemy && hit.slot >= 0) {
            buf[i].attacker_id = shooter.id;
            buf[i].victim_slot = static_cast<std::uint32_t>(hit.slot);
            buf[i].damage_centi_hp = m.revolver_damage_centi_hp;
            has_damage[i] = true;
        }
    }
}

void resolve_mender_sidearm_fire(MatchState& state,
                                 const std::array<common::Action, kAgentsPerMatch>& actions,
                                 const Phase1MechanicsConfig& m,
                                 const MatchConfig& config,
                                 DamageBuffer& buf,
                                 std::array<bool, kAgentsPerMatch>& has_damage) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& shooter = state.heroes[i];
        if (!shooter.present || !shooter.alive ||
            shooter.kind != common::HeroKind::Mender ||
            shooter.mender_weapon != common::MenderWeapon::Sidearm) {
            continue;
        }
        const common::Action& a = actions[i];
        if (!a.primary_fire || shooter.weapon.fire_cooldown_ticks > 0) {
            continue;
        }

        const common::Vec2 d{std::cos(shooter.aim_angle), std::sin(shooter.aim_angle)};
        const RayHitCandidate hit = first_ray_hit_enemy_or_barrier(
            state, i, d, common::kMenderSidearmRange, m.revolver_hitbox_radius, config);

        shooter.weapon.fire_cooldown_ticks = common::kMenderSidearmCooldownTicks;
        if (hit.kind == RayHitCandidate::Kind::Barrier && hit.slot >= 0) {
            HeroState& barrier_owner = state.heroes[static_cast<std::uint32_t>(hit.slot)];
            barrier_owner.vanguard_barrier_hp_centi = std::max<std::int32_t>(
                0,
                barrier_owner.vanguard_barrier_hp_centi -
                    common::kMenderSidearmDamageCentiHp);
            if (barrier_owner.vanguard_barrier_hp_centi == 0) {
                barrier_owner.vanguard_barrier_active = false;
                barrier_owner.cd_ability_1 =
                    common::kVanguardBarrierBrokenCooldownTicks;
            }
            continue;
        }
        if (hit.kind == RayHitCandidate::Kind::Enemy && hit.slot >= 0) {
            buf[i].attacker_id = shooter.id;
            buf[i].victim_slot = static_cast<std::uint32_t>(hit.slot);
            buf[i].damage_centi_hp =
                static_cast<std::uint32_t>(common::kMenderSidearmDamageCentiHp);
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
        const std::int32_t applied =
            std::min<std::int32_t>(damage, victim.health_centi_hp);
        victim.health_centi_hp = std::max<std::int32_t>(0, victim.health_centi_hp - damage);
        // Credit the attacker (slot i) for the damage that actually reduced
        // HP — clamps to remaining HP so overflow past 0 isn't double-counted.
        if (applied > 0) {
            state.heroes[i].damage_dealt_centi_hp +=
                static_cast<std::uint64_t>(applied);
        }
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
