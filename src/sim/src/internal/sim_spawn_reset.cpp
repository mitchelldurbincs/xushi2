#include "sim_spawn_reset.h"

#include <cstdint>

#include <xushi2/common/limits.hpp>
#include <xushi2/common/math.hpp>

namespace xushi2::sim::internal {

// --- Hero initializers (full hero, full magazine where applicable). ---

void spawn_ranger(HeroState& h, common::Team team, common::EntityId id,
                  common::Vec2 position, float aim_angle) {
    h.id = id;
    h.team = team;
    h.kind = common::HeroKind::Ranger;
    h.role = common::Role::Damage;
    h.position = position;
    h.velocity = common::Vec2{};
    h.aim_angle = aim_angle;
    h.health_centi_hp = common::kRangerMaxHpCentiHp;
    h.max_health_centi_hp = common::kRangerMaxHpCentiHp;
    h.alive = true;
    h.respawn_tick = 0;
    h.cd_ability_1 = 0;
    h.cd_ability_2 = 0;
    h.weapon = RangerWeaponState{};
    h.weapon.magazine = common::kRangerMaxMagazine;
    h.vanguard_barrier_active = false;
    h.vanguard_barrier_hp_centi = 0;
    h.ranger_marked_ticks = 0;
    h.ranger_marked_by = common::Team::Neutral;
    h.mender_weapon = common::MenderWeapon::Staff;
    h.mender_beam_locked_on = 0;
    h.present = true;
    // kills/deaths persist across respawns — set by reset_state at match start.
}

static void spawn_hero(HeroState& h, common::HeroKind kind, common::Team team,
                       common::EntityId id, common::Vec2 position, float aim_angle) {
    spawn_ranger(h, team, id, position, aim_angle);
    h.kind = kind;
    switch (kind) {
        case common::HeroKind::Vanguard:
            h.role = common::Role::Tank;
            h.health_centi_hp = common::kVanguardMaxHpCentiHp;
            h.max_health_centi_hp = common::kVanguardMaxHpCentiHp;
            h.weapon = RangerWeaponState{};
            break;
        case common::HeroKind::Ranger:
            h.role = common::Role::Damage;
            break;
        case common::HeroKind::Mender:
            h.role = common::Role::Support;
            h.health_centi_hp = common::kMenderMaxHpCentiHp;
            h.max_health_centi_hp = common::kMenderMaxHpCentiHp;
            h.weapon = RangerWeaponState{};
            h.mender_weapon = common::MenderWeapon::Staff;
            break;
    }
}

// --- Reset: called from ctor and Sim::reset(). ---

void reset_state(MatchState& state, const MatchConfig& config) {
    state = MatchState{};
    state.rng.seed(config.seed);

    state.objective.owner = common::Team::Neutral;
    state.objective.cap_team = common::Team::Neutral;
    state.objective.cap_progress_ticks = 0;
    state.objective.team_a_score_ticks = 0;
    state.objective.team_b_score_ticks = 0;
    state.objective.unlocked = false;

    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;
    const float team_a_y = config.map.min_y + 0.1F * span_y;
    const float team_b_y = config.map.min_y + 0.9F * span_y;

    if (config.team_size == 1) {
        // Phase-1a: two Rangers, one per team. Slots 1, 2, 4, 5 unoccupied.
        spawn_hero(state.heroes[0], config.hero_kinds[0], common::Team::A, 1,
                   common::Vec2{cx, team_a_y},
                   0.5F * common::kPi);
        spawn_hero(state.heroes[3], config.hero_kinds[3], common::Team::B, 2,
                   common::Vec2{cx, team_b_y},
                   -0.5F * common::kPi);
    } else {
        // Phase 4: 3v3, slots 0–2 (A) and 3–5 (B), x-offset by ±dx.
        const float dx = 0.15F * (config.map.max_x - config.map.min_x);
        const float xs[3] = {cx - dx, cx, cx + dx};
        for (std::uint32_t i = 0; i < 3; ++i) {
            spawn_hero(state.heroes[i], config.hero_kinds[i], common::Team::A,
                       static_cast<common::EntityId>(i + 1),
                       common::Vec2{xs[i], team_a_y},
                       0.5F * common::kPi);
            spawn_hero(state.heroes[3 + i], config.hero_kinds[3 + i], common::Team::B,
                       static_cast<common::EntityId>(4 + i),
                       common::Vec2{xs[i], team_b_y},
                       -0.5F * common::kPi);
        }
    }
}

// --- Tick-pipeline step 14: respawn heroes whose timer elapsed. ---

void respawn_tick_update(HeroState& h, std::uint32_t slot,
                         common::Tick now, const MatchConfig& config) {
    if (h.alive || !h.present) {
        return;
    }
    if (now < h.respawn_tick) {
        return;
    }
    // Respawn at original team spawn point; preserve kills/deaths counters.
    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;
    const float team_a_y = config.map.min_y + 0.1F * span_y;
    const float team_b_y = config.map.min_y + 0.9F * span_y;

    common::Vec2 spawn_pos{};
    float aim_angle = 0.0F;
    if (config.team_size == 1) {
        if (h.team == common::Team::A) {
            spawn_pos = common::Vec2{cx, team_a_y};
            aim_angle = 0.5F * common::kPi;
        } else {
            spawn_pos = common::Vec2{cx, team_b_y};
            aim_angle = -0.5F * common::kPi;
        }
    } else {
        const float dx = 0.15F * (config.map.max_x - config.map.min_x);
        const std::uint32_t local =
            (h.team == common::Team::A) ? slot : (slot - 3U);
        const float xs[3] = {cx - dx, cx, cx + dx};
        spawn_pos = common::Vec2{
            xs[local],
            (h.team == common::Team::A) ? team_a_y : team_b_y};
        aim_angle = (h.team == common::Team::A)
                        ? 0.5F * common::kPi
                        : -0.5F * common::kPi;
    }

    const std::uint32_t preserved_kills = h.kills;
    const std::uint32_t preserved_deaths = h.deaths;
    const common::HeroKind preserved_kind = h.kind;
    spawn_hero(h, preserved_kind, h.team, h.id, spawn_pos, aim_angle);
    h.kills = preserved_kills;
    h.deaths = preserved_deaths;
}

}  // namespace xushi2::sim::internal
