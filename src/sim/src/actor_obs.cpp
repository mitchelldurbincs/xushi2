// Phase-1 actor observation builder.
//
// Field order and widths MUST match python/xushi2/obs_manifest.py.
// Changing either without the other silently breaks training.
//
// The separation contract (docs/observation_spec.md invariant 1):
// this TU must not call any function that iterates hidden enemy state.
// Enemy data enters only through `obs_utils::visible_enemy_1v1`, which is
// the single seam that will gain LoS filtering at Phase 7.

#include <xushi2/sim/obs.h>

#include <cmath>
#include <cstdint>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/sim.h>

namespace xushi2::sim {

namespace {

// Writer that advances a cursor and asserts finiteness at each push.
struct Writer {
    float* out;
    std::uint32_t cursor;

    void push1(float v) noexcept {
        X2_ENSURE(std::isfinite(v), common::ErrorCode::NonFiniteFloat);
        out[cursor++] = v;
    }
    void push2(float a, float b) noexcept { push1(a); push1(b); }
    void push3(float a, float b, float c) noexcept {
        push1(a); push1(b); push1(c);
    }
};

// One-hot encoding {Neutral, Us, Them} for a team-valued field.
void onehot_team(Writer& w, common::Team value, common::Team viewer) noexcept {
    float us = 0.0F;
    float them = 0.0F;
    float neutral = 0.0F;
    if (value == common::Team::Neutral) {
        neutral = 1.0F;
    } else if (value == viewer) {
        us = 1.0F;
    } else {
        them = 1.0F;
    }
    w.push3(neutral, us, them);
}

float clamp01(float v) noexcept {
    if (v < 0.0F) return 0.0F;
    if (v > 1.0F) return 1.0F;
    return v;
}

}  // namespace

void build_actor_obs_phase1(const Sim& sim,
                            std::uint32_t agent_slot,
                            float* out_buffer,
                            std::uint32_t out_capacity) noexcept {
    X2_REQUIRE(out_buffer != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(out_capacity >= kActorObsPhase1Dim,
               common::ErrorCode::CapacityExceeded);

    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();
    const MapBounds& map = cfg.map;

    X2_REQUIRE(agent_slot < s.heroes.size(),
               common::ErrorCode::InvalidHeroId);
    const HeroState& self = s.heroes[agent_slot];
    X2_REQUIRE(self.present, common::ErrorCode::InvalidHeroId);
    X2_REQUIRE(self.team != common::Team::Neutral,
               common::ErrorCode::InvalidHeroId);

    const common::Team viewer = self.team;
    Writer w{out_buffer, 0};

    // --- own_hp ---
    const float own_hp_norm =
        (self.max_health_centi_hp > 0)
            ? clamp01(static_cast<float>(self.health_centi_hp) /
                      static_cast<float>(self.max_health_centi_hp))
            : 0.0F;
    w.push1(own_hp_norm);

    // --- own_velocity (team-frame, normalized by ranger max speed) ---
    const common::Vec2 own_vel_team =
        obs_utils::mirror_velocity_for_team(self.velocity, viewer);
    const float vmax = obs_utils::ranger_max_speed();
    w.push2(own_vel_team.x / vmax, own_vel_team.y / vmax);

    // --- own_aim_unit (team-frame, sin/cos) ---
    const float own_aim_team =
        obs_utils::mirror_angle_for_team(self.aim_angle, viewer);
    float aim_unit[2];
    obs_utils::angle_to_unit(own_aim_team, aim_unit);
    w.push2(aim_unit[0], aim_unit[1]);

    // --- own_position (team-frame, normalized to map extent) ---
    const common::Vec2 own_pos_team =
        obs_utils::mirror_position_for_team(self.position, viewer, map);
    const common::Vec2 own_pos_norm =
        obs_utils::normalize_position_to_map(own_pos_team, map);
    w.push2(own_pos_norm.x, own_pos_norm.y);

    // --- own_ammo ---
    w.push1(static_cast<float>(self.weapon.magazine) /
            static_cast<float>(common::kRangerMaxMagazine));

    // --- own_reloading ---
    w.push1(self.weapon.reloading ? 1.0F : 0.0F);

    // --- own_combat_roll_cd ---
    w.push1(clamp01(static_cast<float>(self.cd_ability_1) /
                    static_cast<float>(common::kRangerCombatRollCooldownTicks)));

    // --- enemy_* section: enter ONLY through visible_enemy_1v1. ---
    const auto enemy = obs_utils::visible_enemy_1v1(s, agent_slot);
    const bool enemy_alive = enemy.present && enemy.alive;

    // --- enemy_alive ---
    w.push1(enemy_alive ? 1.0F : 0.0F);

    // --- enemy_respawn_timer (0 when alive; else ticks_remaining / max) ---
    float respawn_norm = 0.0F;
    if (enemy.present && !enemy.alive && cfg.mechanics.respawn_ticks > 0U) {
        std::int64_t remaining =
            static_cast<std::int64_t>(enemy.respawn_tick) -
            static_cast<std::int64_t>(s.tick);
        if (remaining < 0) remaining = 0;
        respawn_norm = clamp01(
            static_cast<float>(remaining) /
            static_cast<float>(cfg.mechanics.respawn_ticks));
    }
    w.push1(respawn_norm);

    // --- enemy_relative_position (team-frame; zero if dead) ---
    if (enemy_alive) {
        const common::Vec2 enemy_pos_team =
            obs_utils::mirror_position_for_team(enemy.world_position, viewer, map);
        const common::Vec2 enemy_pos_norm =
            obs_utils::normalize_position_to_map(enemy_pos_team, map);
        w.push2(enemy_pos_norm.x - own_pos_norm.x,
                enemy_pos_norm.y - own_pos_norm.y);
    } else {
        w.push2(0.0F, 0.0F);
    }

    // --- enemy_hp ---
    float enemy_hp_norm = 0.0F;
    if (enemy_alive && enemy.max_health_centi_hp > 0) {
        enemy_hp_norm = clamp01(
            static_cast<float>(enemy.health_centi_hp) /
            static_cast<float>(enemy.max_health_centi_hp));
    }
    w.push1(enemy_hp_norm);

    // --- enemy_velocity (team-frame; zero if dead) ---
    if (enemy_alive) {
        const common::Vec2 enemy_vel_team =
            obs_utils::mirror_velocity_for_team(enemy.velocity, viewer);
        w.push2(enemy_vel_team.x / vmax, enemy_vel_team.y / vmax);
    } else {
        w.push2(0.0F, 0.0F);
    }

    // --- objective_owner_onehot ---
    onehot_team(w, s.objective.owner, viewer);

    // --- cap_team_onehot ---
    onehot_team(w, s.objective.cap_team, viewer);

    // --- cap_progress ---
    w.push1(clamp01(static_cast<float>(s.objective.cap_progress_ticks) /
                    static_cast<float>(common::kCaptureTicks)));

    // --- contested (both teams present and alive on point) ---
    bool a_on = false;
    bool b_on = false;
    for (std::uint32_t i = 0; i < s.heroes.size(); ++i) {
        const auto& h = s.heroes[i];
        if (!h.present || !h.alive) continue;
        if (!obs_utils::position_on_objective(h.position, map)) continue;
        if (h.team == common::Team::A) a_on = true;
        if (h.team == common::Team::B) b_on = true;
    }
    const bool contested = a_on && b_on;
    w.push1(contested ? 1.0F : 0.0F);

    // --- objective_unlocked ---
    w.push1(s.objective.unlocked ? 1.0F : 0.0F);

    // --- own_score, enemy_score ---
    const std::uint32_t own_score_ticks = (viewer == common::Team::A)
                                              ? s.objective.team_a_score_ticks
                                              : s.objective.team_b_score_ticks;
    const std::uint32_t enemy_score_ticks = (viewer == common::Team::A)
                                                ? s.objective.team_b_score_ticks
                                                : s.objective.team_a_score_ticks;
    w.push1(clamp01(static_cast<float>(own_score_ticks) /
                    static_cast<float>(common::kWinTicks)));
    w.push1(clamp01(static_cast<float>(enemy_score_ticks) /
                    static_cast<float>(common::kWinTicks)));

    // --- self_on_point, enemy_on_point ---
    const bool self_on = self.alive &&
                         obs_utils::position_on_objective(self.position, map);
    w.push1(self_on ? 1.0F : 0.0F);
    const bool enemy_on = enemy_alive &&
                          obs_utils::position_on_objective(
                              enemy.world_position, map);
    w.push1(enemy_on ? 1.0F : 0.0F);

    // --- round_timer (elapsed / total) ---
    const std::uint64_t round_len_ticks =
        static_cast<std::uint64_t>(cfg.round_length_seconds) *
        static_cast<std::uint64_t>(common::kTickHz);
    const float round_timer = (round_len_ticks > 0)
                                  ? static_cast<float>(s.tick) /
                                        static_cast<float>(round_len_ticks)
                                  : 0.0F;
    w.push1(clamp01(round_timer));

    X2_ENSURE(w.cursor == kActorObsPhase1Dim,
              common::ErrorCode::CapacityExceeded);
}

}  // namespace xushi2::sim
