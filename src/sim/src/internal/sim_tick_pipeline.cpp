#include "sim_tick_pipeline.h"

#include <algorithm>
#include <cmath>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/math.hpp>

#include "sim_combat.h"
#include "sim_objective.h"
#include "sim_spawn_reset.h"
#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

// --- Per-hero movement speeds (game-design.md §6). ---
static constexpr float kVanguardSpeed = 3.6F;
static constexpr float kRangerSpeed   = 4.2F;
static constexpr float kMenderSpeed   = 4.0F;

static float hero_speed(common::HeroKind kind) {
    switch (kind) {
        case common::HeroKind::Vanguard: return kVanguardSpeed;
        case common::HeroKind::Ranger:   return kRangerSpeed;
        case common::HeroKind::Mender:   return kMenderSpeed;
    }
    X2_UNREACHABLE();
}

static float cross(common::Vec2 a, common::Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

static bool movement_crosses_wall(common::Vec2 from, common::Vec2 to,
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

static common::Vec2 prevent_wall_crossing(common::Vec2 from, common::Vec2 to,
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

static common::Vec2 resolve_cover_overlap(common::Vec2 p,
                                          common::Vec2 fallback_dir,
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
        p = common::Vec2{
            cover.center.x + delta.x * inv_dist * cover.radius,
            cover.center.y + delta.y * inv_dist * cover.radius,
        };
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
        const float t = std::clamp(
            ((p.x - wall.a.x) * ab.x + (p.y - wall.a.y) * ab.y) / len_sq,
            0.0F,
            1.0F);
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
        p = common::Vec2{
            nearest.x + delta.x * inv_dist * wall.half_width,
            nearest.y + delta.y * inv_dist * wall.half_width,
        };
    }
    p.x = common::clampf(p.x, config.map.min_x, config.map.max_x);
    p.y = common::clampf(p.y, config.map.min_y, config.map.max_y);
    return p;
}

static void stage_abilities_vanguard_barrier(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || h.kind != common::HeroKind::Vanguard) {
            continue;
        }
        if (!h.alive || !actions[i].ability_1 || h.cd_ability_1 != 0) {
            h.vanguard_barrier_active = false;
            continue;
        }
        if (h.vanguard_barrier_hp_centi <= 0) {
            h.vanguard_barrier_hp_centi = common::kVanguardBarrierHpCenti;
        }
        h.vanguard_barrier_active = true;
    }
}

// Tick-pipeline step 7: Combat Roll (impulse, first tick of decision).
static void maybe_combat_roll(HeroState& h, const common::Action& a, bool aim_consumed,
                              const MatchConfig& config) {
    // Impulse semantics: fires only on the first tick of a decision window.
    if (aim_consumed) {
        return;
    }
    if (!a.ability_1 || !h.alive || h.cd_ability_1 != 0) {
        return;
    }
    // Dash direction: movement input if any, else aim direction.
    common::Vec2 dir{};
    const float move_mag_sq = a.move_x * a.move_x + a.move_y * a.move_y;
    if (move_mag_sq > 1e-6F) {
        const float inv = 1.0F / std::sqrt(move_mag_sq);
        dir.x = a.move_x * inv;
        dir.y = a.move_y * inv;
    } else {
        dir.x = std::cos(h.aim_angle);
        dir.y = std::sin(h.aim_angle);
    }
    common::Vec2 next{h.position.x + dir.x * common::kRangerCombatRollDistance,
                      h.position.y + dir.y * common::kRangerCombatRollDistance};
    // Clamp to arena bounds (Phase 1 has no interior walls).
    next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
    next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
    next = prevent_wall_crossing(h.position, next, config);
    h.position = resolve_cover_overlap(next, dir, config);
    weapon_on_combat_roll(h.weapon);
    h.cd_ability_1 = common::kRangerCombatRollCooldownTicks;
}

// Pre: actions have been canonicalized by the caller. aim_consumed[i] is
//      true iff this tick is a non-first sub-tick of step_decision().
// Post: living heroes' aim_angle updated by the canonicalized aim_delta
//       (wrapped). Positions/velocities untouched.
static void stage_validate_and_aim(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        const common::Action& a = actions[i];
        X2_INVARIANT(std::isfinite(a.move_x) && std::isfinite(a.move_y),
                     common::ErrorCode::NonFiniteFloat);
        X2_INVARIANT(std::isfinite(a.aim_delta), common::ErrorCode::NonFiniteFloat);

        if (!aim_consumed[i]) {
            const float delta =
                common::clampf(a.aim_delta, -common::kAimDeltaMax, common::kAimDeltaMax);
            h.aim_angle = common::wrap_angle(h.aim_angle + delta);
        }
    }
}

// Pre: aim updated. Post: positions advanced by velocity * kDt and clamped
//      to map bounds; velocity reflects the canonicalized move input at
//      the hero's speed.
static void stage_movement_and_bounds(
    MatchState& state, const MatchConfig& config,
    const std::array<common::Action, kAgentsPerMatch>& actions) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        const common::Action& a = actions[i];
        common::Vec2 move_vec = common::normalize_move_input(common::Vec2{a.move_x, a.move_y});
        float speed = hero_speed(h.kind);
        if (h.kind == common::HeroKind::Vanguard && h.vanguard_barrier_active) {
            speed *= 0.7F;
        }
        h.velocity = common::scale(move_vec, speed);
        common::Vec2 next = common::add(h.position, common::scale(h.velocity, kDt));
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        next = prevent_wall_crossing(h.position, next, config);
        h.position = resolve_cover_overlap(next, move_vec, config);
    }
}

// Pre: positions stable. Post: per-hero ability cooldowns decremented;
//      living Ranger weapon state advanced (auto-reload bookkeeping).
//      MUST run before abilities so Combat Roll's cd check sees the
//      correct value, and before fire resolution so fire_cooldown_ticks
//      is current this tick.
static void stage_cooldowns_and_weapon_tick(MatchState& state) {
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
        } else if (h.alive &&
                   (h.kind == common::HeroKind::Vanguard ||
                    h.kind == common::HeroKind::Mender) &&
                   h.weapon.fire_cooldown_ticks > 0) {
            --h.weapon.fire_cooldown_ticks;
        }
    }
}

// Pre: cooldowns decremented for THIS tick. Post: any qualifying Ranger
//      that requested ability_1 this decision-window is dashed and its
//      magazine refilled (instant reload). Only fires on the first
//      sub-tick (aim_consumed false) — impulse semantics.
static void stage_abilities_combat_roll(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed,
    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || h.kind != common::HeroKind::Ranger) {
            continue;
        }
        maybe_combat_roll(h, actions[i], aim_consumed[i], config);
    }
}

static void stage_abilities_vanguard_guard_step(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed,
    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive || h.kind != common::HeroKind::Vanguard) {
            continue;
        }
        const common::Action& a = actions[i];
        if (aim_consumed[i] || !a.ability_2 || h.cd_ability_2 != 0) {
            continue;
        }
        common::Vec2 next{
            h.position.x + std::cos(h.aim_angle) * common::kVanguardGuardStepDistance,
            h.position.y + std::sin(h.aim_angle) * common::kVanguardGuardStepDistance,
        };
        const common::Vec2 dir{std::cos(h.aim_angle), std::sin(h.aim_angle)};
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        next = prevent_wall_crossing(h.position, next, config);
        h.position = resolve_cover_overlap(next, dir, config);
        h.cd_ability_2 = common::kVanguardGuardStepCooldownTicks;
    }
}

static void stage_abilities_ranger_mark_target(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed,
    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& ranger = state.heroes[i];
        if (!ranger.present || !ranger.alive ||
            ranger.kind != common::HeroKind::Ranger) {
            continue;
        }
        const common::Action& a = actions[i];
        if (aim_consumed[i] || !a.ability_2 || a.target_slot != 1 ||
            ranger.cd_ability_2 != 0) {
            continue;
        }
        float best_dist_sq = common::kRangerRevolverRange * common::kRangerRevolverRange;
        int best_slot = -1;
        for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
            const HeroState& target = state.heroes[j];
            if (!target.present || !target.alive || target.team == ranger.team) {
                continue;
            }
            const common::Vec2 to_target{
                target.position.x - ranger.position.x,
                target.position.y - ranger.position.y,
            };
            const float dist_sq = to_target.x * to_target.x + to_target.y * to_target.y;
            if (dist_sq > best_dist_sq ||
                segment_blocked_by_cover(ranger.position, target.position, config)) {
                continue;
            }
            best_dist_sq = dist_sq;
            best_slot = static_cast<int>(j);
        }
        ranger.cd_ability_2 = common::kRangerMarkTargetCooldownTicks;
        if (best_slot >= 0) {
            HeroState& target = state.heroes[static_cast<std::uint32_t>(best_slot)];
            target.ranger_marked_ticks = common::kRangerMarkTargetDurationTicks;
            target.ranger_marked_by = ranger.team;
        }
    }
}

static void stage_vanguard_warhammer(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed,
    DamageBuffer& buf,
    std::array<bool, kAgentsPerMatch>& has_damage,
    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& vanguard = state.heroes[i];
        if (!vanguard.present || !vanguard.alive ||
            vanguard.kind != common::HeroKind::Vanguard) {
            continue;
        }
        const common::Action& a = actions[i];
        if (aim_consumed[i] || !a.primary_fire || vanguard.vanguard_barrier_active ||
            vanguard.weapon.fire_cooldown_ticks > 0) {
            continue;
        }
        const common::Vec2 facing{std::cos(vanguard.aim_angle), std::sin(vanguard.aim_angle)};
        float best_dist_sq = common::kVanguardWarhammerRange * common::kVanguardWarhammerRange;
        int best_slot = -1;
        for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
            const HeroState& target = state.heroes[j];
            if (!target.present || !target.alive || target.team == vanguard.team) {
                continue;
            }
            const common::Vec2 to_target{
                target.position.x - vanguard.position.x,
                target.position.y - vanguard.position.y,
            };
            const float dist_sq = to_target.x * to_target.x + to_target.y * to_target.y;
            if (dist_sq <= 1e-6F || dist_sq > best_dist_sq) {
                continue;
            }
            const float inv_dist = 1.0F / std::sqrt(dist_sq);
            if (segment_blocked_by_cover(vanguard.position, target.position, config)) {
                continue;
            }
            const float dot = (to_target.x * facing.x + to_target.y * facing.y) * inv_dist;
            if (dot < common::kVanguardWarhammerHalfAngleCos) {
                continue;
            }
            best_dist_sq = dist_sq;
            best_slot = static_cast<int>(j);
        }
        vanguard.weapon.fire_cooldown_ticks = common::kVanguardWarhammerCooldownTicks;
        if (best_slot >= 0) {
            buf[i].attacker_id = vanguard.id;
            buf[i].victim_slot = static_cast<std::uint32_t>(best_slot);
            buf[i].damage_centi_hp =
                static_cast<std::uint32_t>(common::kVanguardWarhammerDamageCentiHp);
            has_damage[i] = true;
        }
    }
}

static void stage_abilities_mender_weapon_swap(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& mender = state.heroes[i];
        if (!mender.present || !mender.alive ||
            mender.kind != common::HeroKind::Mender) {
            continue;
        }
        const common::Action& a = actions[i];
        if (aim_consumed[i] || !a.ability_1 || mender.cd_ability_1 != 0) {
            continue;
        }
        mender.mender_weapon =
            (mender.mender_weapon == common::MenderWeapon::Staff)
                ? common::MenderWeapon::Sidearm
                : common::MenderWeapon::Staff;
        mender.mender_beam_locked_on = 0;
        mender.cd_ability_1 = common::kMenderWeaponSwapCooldownTicks;
    }
}

static int slot_for_entity_id(const MatchState& state, common::EntityId id) {
    if (id == 0) {
        return -1;
    }
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        const HeroState& h = state.heroes[i];
        if (h.present && h.id == id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static int nearest_ally_in_mender_beam_cone(const MatchState& state,
                                            std::uint32_t mender_slot,
                                            float range,
                                            float half_angle_cos,
                                            const MatchConfig& config) {
    const HeroState& mender = state.heroes[mender_slot];
    const common::Vec2 facing{std::cos(mender.aim_angle), std::sin(mender.aim_angle)};
    float best_dist_sq = range * range;
    int best_slot = -1;
    for (std::uint32_t j = 0; j < kAgentsPerMatch; ++j) {
        if (j == mender_slot) {
            continue;
        }
        const HeroState& ally = state.heroes[j];
        if (!ally.present || !ally.alive || ally.team != mender.team) {
            continue;
        }
        const common::Vec2 to_ally{
            ally.position.x - mender.position.x,
            ally.position.y - mender.position.y,
        };
        const float dist_sq = to_ally.x * to_ally.x + to_ally.y * to_ally.y;
        if (dist_sq <= 1e-6F || dist_sq > best_dist_sq) {
            continue;
        }
        const float inv_dist = 1.0F / std::sqrt(dist_sq);
        if (segment_blocked_by_cover(mender.position, ally.position, config)) {
            continue;
        }
        const float dot = (to_ally.x * facing.x + to_ally.y * facing.y) * inv_dist;
        if (dot < half_angle_cos) {
            continue;
        }
        best_dist_sq = dist_sq;
        best_slot = static_cast<int>(j);
    }
    return best_slot;
}

static void stage_mender_staff_beam(
    MatchState& state,
    const MatchConfig& config,
    const std::array<common::Action, kAgentsPerMatch>& actions) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& mender = state.heroes[i];
        if (!mender.present || !mender.alive ||
            mender.kind != common::HeroKind::Mender ||
            mender.mender_weapon != common::MenderWeapon::Staff ||
            !actions[i].primary_fire) {
            mender.mender_beam_locked_on = 0;
            continue;
        }

        int target_slot = slot_for_entity_id(state, mender.mender_beam_locked_on);
        if (target_slot >= 0) {
            const HeroState& target = state.heroes[static_cast<std::uint32_t>(target_slot)];
            const float dx = target.position.x - mender.position.x;
            const float dy = target.position.y - mender.position.y;
            const float max_dist_sq = common::kMenderBeamRange * common::kMenderBeamRange;
            if (!target.alive || target.team != mender.team ||
                (dx * dx + dy * dy) > max_dist_sq ||
                segment_blocked_by_cover(mender.position, target.position, config)) {
                target_slot = -1;
            }
        }
        if (target_slot < 0) {
            target_slot = nearest_ally_in_mender_beam_cone(
                state, i, common::kMenderBeamRange,
                common::kMenderBeamHalfAngleCos, config);
            mender.mender_beam_locked_on =
                (target_slot >= 0)
                    ? state.heroes[static_cast<std::uint32_t>(target_slot)].id
                    : 0;
        }
        if (target_slot < 0) {
            continue;
        }

        HeroState& target = state.heroes[static_cast<std::uint32_t>(target_slot)];
        target.health_centi_hp = std::min<std::int32_t>(
            target.max_health_centi_hp,
            target.health_centi_hp + common::kMenderBeamHealCentiHpPerTick);
    }
}

static void stage_abilities_mender_tether(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const std::array<bool, kAgentsPerMatch>& aim_consumed,
    const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& mender = state.heroes[i];
        if (!mender.present || !mender.alive ||
            mender.kind != common::HeroKind::Mender) {
            continue;
        }
        const common::Action& a = actions[i];
        if (aim_consumed[i] || !a.ability_2 || mender.cd_ability_2 != 0) {
            continue;
        }
        const int target_slot = nearest_ally_in_mender_beam_cone(
            state, i, common::kMenderTetherRange,
            common::kMenderBeamHalfAngleCos, config);
        if (target_slot < 0) {
            continue;
        }
        const HeroState& target = state.heroes[static_cast<std::uint32_t>(target_slot)];
        const common::Vec2 from_target{
            mender.position.x - target.position.x,
            mender.position.y - target.position.y,
        };
        const float dist_sq = from_target.x * from_target.x + from_target.y * from_target.y;
        if (dist_sq <= 1e-6F) {
            continue;
        }
        const float inv_dist = 1.0F / std::sqrt(dist_sq);
        common::Vec2 next{
            target.position.x + from_target.x * inv_dist * common::kMenderTetherStopDistance,
            target.position.y + from_target.y * inv_dist * common::kMenderTetherStopDistance,
        };
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        next = prevent_wall_crossing(mender.position, next, config);
        mender.position = resolve_cover_overlap(next, from_target, config);
        mender.cd_ability_2 = common::kMenderTetherCooldownTicks;
    }
}

// Pre: positions + cooldowns current. Post: DamageBuffer populated;
//      attackers' magazines/fire-cooldowns updated via
//      weapon_on_fire_success. NO HP changes here. Per-attacker
//      target tie-break is by victim slot index for determinism.
static void stage_fire_resolution(
    MatchState& state,
    const std::array<common::Action, kAgentsPerMatch>& actions,
    const Phase1MechanicsConfig& m,
    const MatchConfig& config,
    DamageBuffer& buf,
    std::array<bool, kAgentsPerMatch>& has_damage) {
    resolve_revolver_fire(state, actions, m, config, buf, has_damage);
}

// Pre: DamageBuffer populated. Post: victim HP reduced. All damage from
//      this tick is applied SIMULTANEOUSLY (no kill-credit ordering bias)
//      — a victim already dead this tick is left at 0 HP; subsequent
//      damage to a dead victim is dropped.
static void stage_apply_damage(MatchState& state,
                               const DamageBuffer& buf,
                               const std::array<bool, kAgentsPerMatch>& has_damage) {
    apply_damage_buffer(state, buf, has_damage);
}

// Pre: HP applied. Post: heroes at 0 HP marked dead with respawn_tick set;
//      death counters incremented; kill credit awarded to attackers whose
//      victim died this tick. MUST run after damage application so
//      simultaneous lethal trades both score.
static void stage_process_deaths(MatchState& state,
                                 const DamageBuffer& buf,
                                 const std::array<bool, kAgentsPerMatch>& has_damage,
                                 const MatchConfig& config) {
    process_deaths(state, buf, has_damage, config);
}

// Pre: deaths processed. Post: any hero whose respawn_tick has elapsed is
//      respawned at its team's spawn point with full HP/magazine, kills
//      and deaths preserved. Order: respawn after death-processing so a
//      hero that died and revived in the same tick is impossible.
static void stage_respawn(MatchState& state, const MatchConfig& config) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        respawn_tick_update(state.heroes[i], i, state.tick, config);
    }
}

// Pre: hero positions/alive flags reflect this tick's outcomes. Post:
//      ObjectiveState advanced by one tick of the 5-case state machine.
//      Score counters monotonically non-decreasing (asserted inside).
static void stage_objective(MatchState& state, const MapBounds& map) {
    objective_tick_update(state.objective, state.heroes, state.tick, map);
}

void apply_one_tick(MatchState& state, const MatchConfig& config,
                    const std::array<common::Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    stage_validate_and_aim(state, actions, aim_consumed);
    stage_abilities_vanguard_barrier(state, actions);
    stage_movement_and_bounds(state, config, actions);
    stage_cooldowns_and_weapon_tick(state);
    stage_abilities_combat_roll(state, actions, aim_consumed, config);
    stage_abilities_vanguard_guard_step(state, actions, aim_consumed, config);
    stage_abilities_ranger_mark_target(state, actions, aim_consumed, config);
    stage_abilities_mender_weapon_swap(state, actions, aim_consumed);
    stage_mender_staff_beam(state, config, actions);
    stage_abilities_mender_tether(state, actions, aim_consumed, config);
    // Steps 8–9 (spatial index, fog) deferred — Phase 7+.
    DamageBuffer buf{};
    std::array<bool, kAgentsPerMatch> has_damage{};
    stage_fire_resolution(state, actions, config.mechanics, config, buf, has_damage);
    resolve_mender_sidearm_fire(state, actions, config.mechanics, config, buf, has_damage);
    stage_vanguard_warhammer(state, actions, aim_consumed, buf, has_damage, config);
    stage_apply_damage(state, buf, has_damage);
    stage_process_deaths(state, buf, has_damage, config);
    stage_respawn(state, config);
    stage_objective(state, config.map);
    // Steps 16–18 (rewards / obs / replay) deferred — Phase 1b/1c.
    state.tick += 1;
}

}  // namespace xushi2::sim::internal
