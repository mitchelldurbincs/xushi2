#include "sim_tick_pipeline.h"

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

// Tick-pipeline step 7: Combat Roll (impulse, first tick of decision).
static void maybe_combat_roll(HeroState& h, const common::Action& a, bool aim_consumed,
                              const MapBounds& map) {
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
    next.x = common::clampf(next.x, map.min_x, map.max_x);
    next.y = common::clampf(next.y, map.min_y, map.max_y);
    h.position = next;
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
        const float speed = hero_speed(h.kind);
        h.velocity = common::scale(move_vec, speed);
        common::Vec2 next = common::add(h.position, common::scale(h.velocity, kDt));
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        h.position = next;
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
        if (h.alive && h.kind == common::HeroKind::Ranger) {
            weapon_tick_update(h.weapon);
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
    const MapBounds& map) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || h.kind != common::HeroKind::Ranger) {
            continue;
        }
        maybe_combat_roll(h, actions[i], aim_consumed[i], map);
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
    DamageBuffer& buf,
    std::array<bool, kAgentsPerMatch>& has_damage) {
    resolve_revolver_fire(state, actions, m, buf, has_damage);
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
        respawn_tick_update(state.heroes[i], state.tick, config);
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
    stage_movement_and_bounds(state, config, actions);
    stage_cooldowns_and_weapon_tick(state);
    stage_abilities_combat_roll(state, actions, aim_consumed, config.map);
    // Steps 8–9 (spatial index, fog) deferred — Phase 7+.
    DamageBuffer buf{};
    std::array<bool, kAgentsPerMatch> has_damage{};
    stage_fire_resolution(state, actions, config.mechanics, buf, has_damage);
    stage_apply_damage(state, buf, has_damage);
    stage_process_deaths(state, buf, has_damage, config);
    stage_respawn(state, config);
    stage_objective(state, config.map);
    // Steps 16–18 (rewards / obs / replay) deferred — Phase 1b/1c.
    state.tick += 1;
}

}  // namespace xushi2::sim::internal
