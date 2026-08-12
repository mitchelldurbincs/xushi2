// Native entity-token/grid observation builder (ObservationEngine).
//
// Field-for-field port of python/xushi2/multi_enemy_obs.py
// `actor_obs_to_multi_enemy_entity_grid_obs` plus the Phase-11 visibility
// rule and last-seen memory from
// python/envs/phase11_current_selfplay_mappo.py. Parity with the Python
// path is pinned by python/tests/test_entity_obs_native_parity.py; three
// deliberate warts are preserved for checkpoint continuity:
//   - visible_radius is measured in normalized (anisotropic) map units;
//   - enemy aim units are world-frame even for Team-B viewers (positions
//     and velocities are mirrored, aim is not);
//   - the self token's alive bit is 1.0 even when the viewer is dead.
//
// Separation contract (docs/observation_spec.md invariant 1): this is the
// one translation unit permitted to iterate enemy HeroState for
// actor-destined data, and only inside `fog_gate`, which emits the
// FoggedEnemyView that all token/grid writing reads from. The critic
// tensor is never an input here. Visibility routes through
// obs_utils::observable_enemy — the same native rule as every other
// actor-obs path.

#include <xushi2/sim/entity_obs.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/obs_utils.h>

namespace xushi2::sim {

namespace {

// Token field offsets — mirror the slices in multi_enemy_obs.py.
constexpr std::uint32_t kFieldKind = 0;       // 3 wide
constexpr std::uint32_t kFieldTeam = 3;       // 3 wide
constexpr std::uint32_t kFieldHp = 6;
constexpr std::uint32_t kFieldAlive = 7;
constexpr std::uint32_t kFieldPosition = 8;   // 2 wide
constexpr std::uint32_t kFieldVelocity = 10;  // 2 wide
constexpr std::uint32_t kFieldAim = 12;       // 2 wide
constexpr std::uint32_t kFieldAmmo = 14;
constexpr std::uint32_t kFieldReloading = 15;
constexpr std::uint32_t kFieldAbilityCd = 16;
constexpr std::uint32_t kFieldAux = 17;

constexpr std::uint32_t kSelfToken = 0;
constexpr std::uint32_t kFirstEnemyToken = 1;
constexpr std::uint32_t kObjectiveToken = 4;

constexpr std::uint32_t kTokenWidth = kEntityTokenCount * kEntityTokenDim;

float clamp01(float v) noexcept {
    if (v < 0.0F) return 0.0F;
    if (v > 1.0F) return 1.0F;
    return v;
}

common::Vec2 arena_center(const MapBounds& map) noexcept {
    return common::Vec2{0.5F * (map.min_x + map.max_x),
                        0.5F * (map.min_y + map.max_y)};
}

common::Vec2 team_norm_position(common::Vec2 world_pos,
                                common::Team viewer_team,
                                const MapBounds& map) noexcept {
    return obs_utils::normalize_position_to_map(
        obs_utils::mirror_position_for_team(world_pos, viewer_team, map), map);
}

// The Python side paints with built-in round() on a double, which rounds
// half to even. std::lrint under the default FE_TONEAREST mode matches;
// std::round (half away from zero) would not.
void paint(float* grid, std::uint32_t channel, float x, float y,
           float value) noexcept {
    constexpr double kScale = static_cast<double>(kEntityGridSize - 1);
    constexpr long kMaxIndex = static_cast<long>(kEntityGridSize - 1);
    const long ix = std::clamp(
        std::lrint((static_cast<double>(x) + 1.0) * 0.5 * kScale), 0L, kMaxIndex);
    const long iy = std::clamp(
        std::lrint((1.0 - ((static_cast<double>(y) + 1.0) * 0.5)) * kScale), 0L,
        kMaxIndex);
    float& cell =
        grid[(((channel * kEntityGridSize) + static_cast<std::uint32_t>(iy)) *
              kEntityGridSize) +
             static_cast<std::uint32_t>(ix)];
    cell = std::max(cell, value);
}

// Ascending opposite-team slots for one viewer — the same ordering the
// critic's enemy blocks and the Phase-11 rule use.
std::array<std::uint32_t, kTeamSize> enemy_slots_for(
        const MatchState& s, common::Team viewer_team) noexcept {
    std::array<std::uint32_t, kTeamSize> out{};
    std::uint32_t found = 0;
    for (std::uint32_t i = 0;
         i < s.heroes.size() && found < static_cast<std::uint32_t>(kTeamSize);
         ++i) {
        const auto& h = s.heroes[i];
        if (h.present && h.team != viewer_team &&
            h.team != common::Team::Neutral) {
            out[found++] = i;
        }
    }
    X2_REQUIRE(found == static_cast<std::uint32_t>(kTeamSize),
               common::ErrorCode::InvalidHeroId);
    return out;
}

std::array<std::uint32_t, kTeamSize> ally_slots_for(
        const MatchState& s, common::Team viewer_team) noexcept {
    std::array<std::uint32_t, kTeamSize> out{};
    std::uint32_t found = 0;
    for (std::uint32_t i = 0;
         i < s.heroes.size() && found < static_cast<std::uint32_t>(kTeamSize);
         ++i) {
        const auto& h = s.heroes[i];
        if (h.present && h.team == viewer_team) {
            out[found++] = i;
        }
    }
    X2_REQUIRE(found == static_cast<std::uint32_t>(kTeamSize),
               common::ErrorCode::InvalidHeroId);
    return out;
}

// Normalized-frame distance test — the Phase-11 radius rule. Frame choice
// (which team's mirror) does not affect the distance; the anisotropy of
// per-axis normalization does, and is preserved.
bool within_radius(const HeroState& observer, const HeroState& enemy,
                   const MapBounds& map, float radius) noexcept {
    const common::Vec2 a = team_norm_position(observer.position,
                                              observer.team, map);
    const common::Vec2 b = team_norm_position(enemy.position,
                                              observer.team, map);
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    return std::sqrt((dx * dx) + (dy * dy)) <= radius;
}

// THE fog gate: the only place enemy HeroState is read for actor-destined
// data. Everything the token/grid writers see about an enemy comes out of
// the FoggedEnemyView returned here.
FoggedEnemyView fog_gate(const Sim& sim, std::uint32_t enemy_slot,
                         common::Team viewer_team, common::Vec2 own_pos,
                         bool visible, bool stale_available,
                         common::Vec2 stale_pos_norm) noexcept {
    FoggedEnemyView view{};
    if (visible) {
        const MapBounds& map = sim.config().map;
        const HeroState& enemy = sim.state().heroes[enemy_slot];
        view.visible = true;
        const common::Vec2 pos =
            team_norm_position(enemy.position, viewer_team, map);
        view.rel_position = common::Vec2{pos.x - own_pos.x, pos.y - own_pos.y};
        view.hp = (enemy.max_health_centi_hp > 0)
                      ? static_cast<float>(enemy.health_centi_hp) /
                            static_cast<float>(enemy.max_health_centi_hp)
                      : 0.0F;
        const common::Vec2 vel =
            obs_utils::mirror_velocity_for_team(enemy.velocity, viewer_team);
        const float vmax = obs_utils::ranger_max_speed();
        view.velocity_norm = common::Vec2{vel.x / vmax, vel.y / vmax};
        // Parity wart: world-frame aim, never mirrored.
        view.aim_sin = std::sin(enemy.aim_angle);
        view.aim_cos = std::cos(enemy.aim_angle);
        view.ammo = static_cast<float>(enemy.weapon.magazine) /
                    static_cast<float>(common::kRangerMaxMagazine);
        view.reloading = enemy.weapon.reloading ? 1.0F : 0.0F;
        view.ability_cd = clamp01(
            static_cast<float>(enemy.cd_ability_1) /
            static_cast<float>(common::kRangerCombatRollCooldownTicks));
        const common::Vec2 center = arena_center(map);
        const float dx = enemy.position.x - center.x;
        const float dy = enemy.position.y - center.y;
        view.on_objective =
            (std::sqrt((dx * dx) + (dy * dy)) <= common::kObjectiveRadius)
                ? 1.0F
                : 0.0F;
    } else if (stale_available) {
        view.stale = true;
        view.rel_position = common::Vec2{stale_pos_norm.x - own_pos.x,
                                         stale_pos_norm.y - own_pos.y};
    }
    return view;
}

// Write one enemy token + mask bit + grid mark from its FoggedEnemyView.
// Reads no sim state: the gate above is the only enemy-data source.
void write_enemy_token(const FoggedEnemyView& view, bool zero_hidden_markers,
                       float* tok, float* mask_bit, float* grid) noexcept {
    const bool hidden = !(view.visible || view.stale);
    if (!hidden || !zero_hidden_markers) {
        tok[kFieldKind + 1] = 1.0F;
        tok[kFieldTeam + 1] = 1.0F;
    }
    if (view.visible) {
        tok[kFieldHp] = view.hp;
        tok[kFieldAlive] = 1.0F;
        tok[kFieldPosition + 0] = view.rel_position.x;
        tok[kFieldPosition + 1] = view.rel_position.y;
        tok[kFieldVelocity + 0] = view.velocity_norm.x;
        tok[kFieldVelocity + 1] = view.velocity_norm.y;
        tok[kFieldAim + 0] = view.aim_sin;
        tok[kFieldAim + 1] = view.aim_cos;
        tok[kFieldAmmo] = view.ammo;
        tok[kFieldReloading] = view.reloading;
        tok[kFieldAbilityCd] = view.ability_cd;
        tok[kFieldAux] = view.on_objective;
        *mask_bit = 1.0F;
        paint(grid, 2, view.rel_position.x, view.rel_position.y, 1.0F);
    } else if (view.stale) {
        tok[kFieldPosition + 0] = view.rel_position.x;
        tok[kFieldPosition + 1] = view.rel_position.y;
        tok[kFieldAux] = 0.5F;
        *mask_bit = 1.0F;
        paint(grid, 2, view.rel_position.x, view.rel_position.y, 0.5F);
    }
}

// Team-shared visibility of one enemy: the radius term and the LoS term
// union independently across the team — one teammate in radius plus a
// different teammate with LoS makes the enemy visible. This matches the
// Phase-11 Python rule exactly.
bool team_shared_visible(const Sim& sim, const ObsConfig& cfg,
                         const std::array<std::uint32_t, kTeamSize>& allies,
                         std::uint32_t enemy_slot) noexcept {
    const MatchState& s = sim.state();
    const bool use_radius = has_visible_radius(cfg);
    bool any_radius = !use_radius;
    bool any_los = false;
    for (const std::uint32_t ally_slot : allies) {
        const HeroState& ally = s.heroes[ally_slot];
        if (use_radius && !any_radius &&
            within_radius(ally, s.heroes[enemy_slot], sim.config().map,
                          cfg.visible_radius)) {
            any_radius = true;
        }
        if (!any_los && obs_utils::observable_enemy(sim, ally_slot, enemy_slot)) {
            any_los = true;
        }
    }
    return any_radius && any_los;
}

}  // namespace

ObservationEngine::ObservationEngine(const ObsConfig& cfg) noexcept
    : cfg_(cfg) {
    X2_REQUIRE(!has_visible_radius(cfg_) || cfg_.visible_radius > 0.0F,
               common::ErrorCode::CorruptState);
}

void ObservationEngine::reset() noexcept {
    for (auto& per_viewer : last_seen_) {
        per_viewer.fill(LastSeen{});
    }
}

std::array<bool, kTeamSize> ObservationEngine::visible_enemies(
        const Sim& sim, std::uint32_t viewer_slot) const noexcept {
    std::array<bool, kTeamSize> out{};
    const MatchState& s = sim.state();
    X2_REQUIRE(viewer_slot < s.heroes.size(), common::ErrorCode::InvalidHeroId);
    const HeroState& viewer = s.heroes[viewer_slot];
    X2_REQUIRE(viewer.present && viewer.team != common::Team::Neutral,
               common::ErrorCode::InvalidHeroId);

    const auto enemies = enemy_slots_for(s, viewer.team);
    const bool use_radius = has_visible_radius(cfg_);

    if (cfg_.fog_mode == FogMode::TeamShared) {
        const auto allies = ally_slots_for(s, viewer.team);
        for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(kTeamSize);
             ++e) {
            if (!s.heroes[enemies[e]].alive) continue;
            out[e] = team_shared_visible(sim, cfg_, allies, enemies[e]);
        }
        return out;
    }

    for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(kTeamSize); ++e) {
        const HeroState& enemy = s.heroes[enemies[e]];
        if (!enemy.alive) continue;
        if (cfg_.fog_mode == FogMode::None) {
            out[e] = true;
            continue;
        }
        const bool radius_ok =
            !use_radius ||
            within_radius(viewer, enemy, sim.config().map, cfg_.visible_radius);
        out[e] = radius_ok &&
                 obs_utils::observable_enemy(sim, viewer_slot, enemies[e]);
    }
    return out;
}

void ObservationEngine::update_last_seen(
        const Sim& sim, std::uint32_t viewer_slot,
        const std::array<bool, kTeamSize>& visible) noexcept {
    const MatchState& s = sim.state();
    const HeroState& viewer = s.heroes[viewer_slot];
    const auto enemies = enemy_slots_for(s, viewer.team);
    for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(kTeamSize); ++e) {
        if (!visible[e]) continue;
        const HeroState& enemy = s.heroes[enemies[e]];
        last_seen_[viewer_slot][e] = LastSeen{
            team_norm_position(enemy.position, viewer.team, sim.config().map),
            true,
        };
    }
}

void ObservationEngine::build_entity_obs(const Sim& sim,
                                         std::uint32_t viewer_slot,
                                         float* out,
                                         std::uint32_t capacity) noexcept {
    X2_REQUIRE(out != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(capacity >= kEntityGridObsDim,
               common::ErrorCode::CapacityExceeded);

    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();
    const MapBounds& map = cfg.map;

    X2_REQUIRE(viewer_slot < s.heroes.size(), common::ErrorCode::InvalidHeroId);
    const HeroState& self = s.heroes[viewer_slot];
    X2_REQUIRE(self.present && self.team != common::Team::Neutral,
               common::ErrorCode::InvalidHeroId);
    const common::Team viewer_team = self.team;

    const auto visible = visible_enemies(sim, viewer_slot);
    if (cfg_.last_seen_enabled) {
        update_last_seen(sim, viewer_slot, visible);
    }

    std::memset(out, 0, sizeof(float) * kEntityGridObsDim);
    float* tokens = out;
    float* mask = out + kTokenWidth;
    float* grid = out + kTokenWidth + kEntityTokenCount;

    const common::Vec2 own_pos =
        team_norm_position(self.position, viewer_team, map);

    // --- self token -------------------------------------------------------
    {
        float* tok = tokens + (kSelfToken * kEntityTokenDim);
        tok[kFieldKind + 0] = 1.0F;
        tok[kFieldTeam + 0] = 1.0F;
        tok[kFieldHp] =
            (self.max_health_centi_hp > 0)
                ? clamp01(static_cast<float>(self.health_centi_hp) /
                          static_cast<float>(self.max_health_centi_hp))
                : 0.0F;
        // Parity wart: 1.0 even when the viewer is dead.
        tok[kFieldAlive] = 1.0F;
        tok[kFieldPosition + 0] = own_pos.x;
        tok[kFieldPosition + 1] = own_pos.y;
        const common::Vec2 vel =
            obs_utils::mirror_velocity_for_team(self.velocity, viewer_team);
        const float vmax = obs_utils::ranger_max_speed();
        tok[kFieldVelocity + 0] = vel.x / vmax;
        tok[kFieldVelocity + 1] = vel.y / vmax;
        float aim_unit[2];
        obs_utils::angle_to_unit(
            obs_utils::mirror_angle_for_team(self.aim_angle, viewer_team),
            aim_unit);
        tok[kFieldAim + 0] = aim_unit[0];
        tok[kFieldAim + 1] = aim_unit[1];
        tok[kFieldAmmo] = static_cast<float>(self.weapon.magazine) /
                          static_cast<float>(common::kRangerMaxMagazine);
        tok[kFieldReloading] = self.weapon.reloading ? 1.0F : 0.0F;
        tok[kFieldAbilityCd] =
            clamp01(static_cast<float>(self.cd_ability_1) /
                    static_cast<float>(common::kRangerCombatRollCooldownTicks));
        const bool self_on =
            self.alive && obs_utils::position_on_objective(self.position, map);
        tok[kFieldAux] = self_on ? 1.0F : 0.0F;
        mask[kSelfToken] = 1.0F;
    }

    // --- objective token --------------------------------------------------
    {
        float* tok = tokens + (kObjectiveToken * kEntityTokenDim);
        tok[kFieldKind + 2] = 1.0F;
        // Owner one-hot in {Neutral, Us, Them} order, same as the actor
        // builder's objective_owner_onehot.
        const common::Team owner = s.objective.owner;
        if (owner == common::Team::Neutral) {
            tok[kFieldTeam + 0] = 1.0F;
        } else if (owner == viewer_team) {
            tok[kFieldTeam + 1] = 1.0F;
        } else {
            tok[kFieldTeam + 2] = 1.0F;
        }
        tok[kFieldAlive] = 1.0F;
        tok[kFieldPosition + 0] = -own_pos.x;
        tok[kFieldPosition + 1] = -own_pos.y;
        tok[kFieldAux] =
            clamp01(static_cast<float>(s.objective.cap_progress_ticks) /
                    static_cast<float>(cfg.objective_capture_ticks));
        mask[kObjectiveToken] = 1.0F;
    }

    paint(grid, 0, -own_pos.x, -own_pos.y, 1.0F);
    paint(grid, 1, 0.0F, 0.0F, 1.0F);

    // --- enemy tokens -----------------------------------------------------
    const auto enemies = enemy_slots_for(s, viewer_team);
    for (std::uint32_t e = 0; e < static_cast<std::uint32_t>(kTeamSize); ++e) {
        const LastSeen& memory = last_seen_[viewer_slot][e];
        const FoggedEnemyView view =
            fog_gate(sim, enemies[e], viewer_team, own_pos, visible[e],
                     cfg_.last_seen_enabled && memory.valid, memory.pos_norm);
        write_enemy_token(view, cfg_.zero_hidden_token_markers,
                          tokens + ((kFirstEnemyToken + e) * kEntityTokenDim),
                          mask + kFirstEnemyToken + e, grid);
    }
}

void ObservationEngine::build_entity_obs_all(const Sim& sim,
                                             float* out,
                                             std::uint32_t capacity) noexcept {
    X2_REQUIRE(out != nullptr, common::ErrorCode::CorruptState);
    const std::uint32_t total =
        static_cast<std::uint32_t>(kAgentsPerMatch) * kEntityGridObsDim;
    X2_REQUIRE(capacity >= total, common::ErrorCode::CapacityExceeded);
    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(kAgentsPerMatch); ++slot) {
        build_entity_obs(sim, slot, out + (slot * kEntityGridObsDim),
                         kEntityGridObsDim);
    }
}

std::uint64_t ObservationEngine::obs_state_hash() const noexcept {
    // FNV-1a over the last-seen memory. Per-platform stable (float bit
    // patterns), same discipline as the golden replay fixtures.
    std::uint64_t h = 14695981039346656037ULL;
    const auto mix = [&h](std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (i * 8)) & 0xFFULL;
            h *= 1099511628211ULL;
        }
    };
    for (const auto& per_viewer : last_seen_) {
        for (const auto& entry : per_viewer) {
            std::uint32_t bits_x = 0;
            std::uint32_t bits_y = 0;
            std::memcpy(&bits_x, &entry.pos_norm.x, sizeof(bits_x));
            std::memcpy(&bits_y, &entry.pos_norm.y, sizeof(bits_y));
            mix((static_cast<std::uint64_t>(bits_x) << 32) |
                static_cast<std::uint64_t>(bits_y));
            mix(entry.valid ? 1ULL : 0ULL);
        }
    }
    return h;
}

}  // namespace xushi2::sim
