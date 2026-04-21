#include <xushi2/sim/sim.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/math.hpp>

// Phase-1a playable slice. 1v1 Ranger on a 50×50 arena. Movement + aim +
// combat (hitscan Revolver + magazine/reload state machine), Combat Roll,
// death + respawn, full 5-case objective state machine. No fog, no walls.
//
// See docs/coding_philosophy.md §13 for the Tier 0 review checklist applied
// here.

namespace xushi2::sim {

namespace {

using common::ErrorCode;
using common::HeroKind;
using common::Team;
using common::Vec2;

// --- Per-hero movement speeds (game-design.md §6). ---
inline constexpr float kVanguardSpeed = 3.6F;
inline constexpr float kRangerSpeed   = 4.2F;
inline constexpr float kMenderSpeed   = 4.0F;

float hero_speed(HeroKind kind) {
    switch (kind) {
        case HeroKind::Vanguard: return kVanguardSpeed;
        case HeroKind::Ranger:   return kRangerSpeed;
        case HeroKind::Mender:   return kMenderSpeed;
    }
    X2_UNREACHABLE();
}

// --- Arena / objective geometry helpers. ---

Vec2 arena_center(const MapBounds& map) {
    return Vec2{0.5F * (map.min_x + map.max_x), 0.5F * (map.min_y + map.max_y)};
}

bool inside_objective(const Vec2& pos, const MapBounds& map) {
    const Vec2 c = arena_center(map);
    const float dx = pos.x - c.x;
    const float dy = pos.y - c.y;
    return (dx * dx + dy * dy) <= (common::kObjectiveRadius * common::kObjectiveRadius);
}

// --- Ranger weapon state transitions (game_design.md §6). ---
//
// These are the ONLY functions that mutate a RangerWeaponState. Each one
// asserts preconditions and postconditions; invalid states are made hard to
// represent by forcing all mutations through these entry points.

void weapon_on_fire_success(RangerWeaponState& w,
                            const Phase1MechanicsConfig& m) {
    X2_REQUIRE(w.magazine > 0, ErrorCode::CorruptState);
    X2_REQUIRE(!w.reloading, ErrorCode::CorruptState);
    X2_REQUIRE(w.fire_cooldown_ticks == 0, ErrorCode::CorruptState);
    w.magazine -= 1;
    w.ticks_since_last_shot = 0;
    w.fire_cooldown_ticks = m.revolver_fire_cooldown_ticks;
    X2_ENSURE(w.magazine <= common::kRangerMaxMagazine, ErrorCode::CorruptState);
}

void weapon_on_combat_roll(RangerWeaponState& w) {
    // Instant reload: cancels any in-progress auto-reload.
    w.magazine = common::kRangerMaxMagazine;
    w.reloading = false;
    w.reload_ticks_left = 0;
    X2_ENSURE(w.magazine == common::kRangerMaxMagazine, ErrorCode::CorruptState);
    X2_ENSURE(!w.reloading, ErrorCode::CorruptState);
}

void weapon_tick_update(RangerWeaponState& w) {
    // Fire-rate gate counts down every tick regardless of reload state.
    if (w.fire_cooldown_ticks > 0) {
        w.fire_cooldown_ticks -= 1;
    }
    if (w.reloading) {
        X2_INVARIANT(w.reload_ticks_left > 0, ErrorCode::CorruptState);
        w.reload_ticks_left -= 1;
        if (w.reload_ticks_left == 0) {
            w.magazine = common::kRangerMaxMagazine;
            w.reloading = false;
        }
        return;
    }
    // Not reloading. Increment inactivity counter first, then check the
    // auto-reload trigger — so exactly kAutoReloadDelayTicks idle ticks after
    // the last shot causes reload to begin this tick (not the next).
    if (w.ticks_since_last_shot < std::numeric_limits<Tick>::max()) {
        w.ticks_since_last_shot += 1;
    }
    if (w.magazine < common::kRangerMaxMagazine &&
        w.ticks_since_last_shot >= common::kAutoReloadDelayTicks) {
        w.reloading = true;
        w.reload_ticks_left = common::kReloadDurationTicks;
    }
}

// --- Ranger-specific initializers (full hero, full magazine). ---

void spawn_ranger(HeroState& h, Team team, EntityId id, Vec2 position, float aim_angle) {
    h.id = id;
    h.team = team;
    h.kind = HeroKind::Ranger;
    h.role = Role::Damage;
    h.position = position;
    h.velocity = Vec2{};
    h.aim_angle = aim_angle;
    h.health_centi_hp = common::kRangerMaxHpCentiHp;
    h.max_health_centi_hp = common::kRangerMaxHpCentiHp;
    h.alive = true;
    h.respawn_tick = 0;
    h.cd_ability_1 = 0;
    h.cd_ability_2 = 0;
    h.weapon = RangerWeaponState{};
    h.weapon.magazine = common::kRangerMaxMagazine;
    h.present = true;
    // kills/deaths persist across respawns — set by reset_state at match start.
}

// --- Reset: called from ctor and Sim::reset(). ---

void reset_state(MatchState& state, const MatchConfig& config) {
    state = MatchState{};
    state.rng.seed(config.seed);

    state.objective.owner = Team::Neutral;
    state.objective.cap_team = Team::Neutral;
    state.objective.cap_progress_ticks = 0;
    state.objective.team_a_score_ticks = 0;
    state.objective.team_b_score_ticks = 0;
    state.objective.unlocked = false;

    // Phase-1a: two Rangers, one per team, fixed spawn points. Other slots
    // unoccupied until Phase 4+.
    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;
    spawn_ranger(state.heroes[0], Team::A, 1,
                 Vec2{cx, config.map.min_y + 0.1F * span_y},
                 0.5F * common::kPi);   // face +y (toward center)
    spawn_ranger(state.heroes[3], Team::B, 2,
                 Vec2{cx, config.map.min_y + 0.9F * span_y},
                 -0.5F * common::kPi);  // face -y (toward center)
}

// --- Tick-pipeline step 7: Combat Roll (impulse, first tick of decision). ---

void maybe_combat_roll(HeroState& h, const Action& a, bool aim_consumed,
                       const MapBounds& map) {
    // Impulse semantics: fires only on the first tick of a decision window.
    if (aim_consumed) {
        return;
    }
    if (!a.ability_1 || !h.alive || h.cd_ability_1 != 0) {
        return;
    }
    // Dash direction: movement input if any, else aim direction.
    Vec2 dir{};
    const float move_mag_sq = a.move_x * a.move_x + a.move_y * a.move_y;
    if (move_mag_sq > 1e-6F) {
        const float inv = 1.0F / std::sqrt(move_mag_sq);
        dir.x = a.move_x * inv;
        dir.y = a.move_y * inv;
    } else {
        dir.x = std::cos(h.aim_angle);
        dir.y = std::sin(h.aim_angle);
    }
    Vec2 next{h.position.x + dir.x * common::kRangerCombatRollDistance,
              h.position.y + dir.y * common::kRangerCombatRollDistance};
    // Clamp to arena bounds (Phase 1 has no interior walls).
    next.x = common::clampf(next.x, map.min_x, map.max_x);
    next.y = common::clampf(next.y, map.min_y, map.max_y);
    h.position = next;
    weapon_on_combat_roll(h.weapon);
    h.cd_ability_1 = common::kRangerCombatRollCooldownTicks;
}

// --- Tick-pipeline step 10: hitscan fire resolution. ---
//
// Writes DamageEvents into the per-tick buffer; applies no HP changes.

struct DamageEvent {
    EntityId attacker_id = 0;
    std::uint32_t victim_slot = 0;
    std::uint32_t damage_centi_hp = 0;
};

using DamageBuffer = std::array<DamageEvent, kAgentsPerMatch>;

// Ray–circle intersection. Returns t >= 0 at which the ray enters the
// circle, or a negative value if the ray misses / is past the circle. The
// ray is (origin + t*d), d assumed unit length.
float ray_circle_hit_t(Vec2 origin, Vec2 d, Vec2 center, float radius) {
    const Vec2 oc{origin.x - center.x, origin.y - center.y};
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
                           const std::array<Action, kAgentsPerMatch>& actions,
                           const Phase1MechanicsConfig& m,
                           DamageBuffer& buf,
                           std::array<bool, kAgentsPerMatch>& has_damage) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& shooter = state.heroes[i];
        if (!shooter.present || !shooter.alive) {
            continue;
        }
        if (shooter.kind != HeroKind::Ranger) {
            continue;  // Phase 1: only Rangers fire
        }
        const Action& a = actions[i];
        if (!a.primary_fire) {
            continue;
        }
        if (shooter.weapon.reloading || shooter.weapon.magazine == 0 ||
            shooter.weapon.fire_cooldown_ticks > 0) {
            continue;
        }
        // Hitscan ray.
        const Vec2 d{std::cos(shooter.aim_angle), std::sin(shooter.aim_angle)};
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
        X2_INVARIANT(victim.present, ErrorCode::CorruptState);
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

// --- Tick-pipeline step 14: respawn heroes whose timer elapsed. ---

void respawn_tick_update(HeroState& h, Tick now, const MatchConfig& config) {
    if (h.alive || !h.present) {
        return;
    }
    if (now < h.respawn_tick) {
        return;
    }
    // Respawn at original team spawn point; preserve kills/deaths counters.
    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;
    Vec2 spawn_pos{};
    float aim_angle = 0.0F;
    if (h.team == Team::A) {
        spawn_pos = Vec2{cx, config.map.min_y + 0.1F * span_y};
        aim_angle = 0.5F * common::kPi;
    } else {
        spawn_pos = Vec2{cx, config.map.min_y + 0.9F * span_y};
        aim_angle = -0.5F * common::kPi;
    }
    const std::uint32_t preserved_kills = h.kills;
    const std::uint32_t preserved_deaths = h.deaths;
    spawn_ranger(h, h.team, h.id, spawn_pos, aim_angle);
    h.kills = preserved_kills;
    h.deaths = preserved_deaths;
}

// --- Tick-pipeline step 15: objective state machine (5-case). ---

void objective_tick_update(ObjectiveState& obj,
                           const std::array<HeroState, kAgentsPerMatch>& heroes,
                           Tick now, const MapBounds& map) {
    // Initial lock: no state changes during the first kObjectiveLockTicks
    // ticks. The `now + 1` comparison is because state.tick is incremented
    // AFTER this function runs — callers observe tick N's state after
    // calling sim.step() N+1 times. Unlock during tick kObjectiveLockTicks-1
    // processing so "15 seconds of lock" is observable after exactly
    // kObjectiveLockTicks sim.step() calls.
    if (now + 1 < common::kObjectiveLockTicks) {
        obj.unlocked = false;
        return;
    }
    obj.unlocked = true;

    // Living heroes inside the objective circle, by team.
    bool present_a = false;
    bool present_b = false;
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        const HeroState& h = heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        if (!inside_objective(h.position, map)) {
            continue;
        }
        if (h.team == Team::A) {
            present_a = true;
        } else if (h.team == Team::B) {
            present_b = true;
        }
    }

    const std::uint32_t a_before = obj.team_a_score_ticks;
    const std::uint32_t b_before = obj.team_b_score_ticks;

    if (present_a && present_b) {
        // Case 1: contested — freeze. No changes.
    } else if (!present_a && !present_b) {
        // Case 2: empty — decay capture progress.
        if (obj.cap_progress_ticks > 0) {
            obj.cap_progress_ticks -= 1;
            if (obj.cap_progress_ticks == 0) {
                obj.cap_team = Team::Neutral;
            }
        }
    } else {
        const Team present_team = present_a ? Team::A : Team::B;
        const Team opp = (present_team == Team::A) ? Team::B : Team::A;
        if (obj.owner == present_team) {
            // Case 3: uncontested by owner — score.
            if (present_team == Team::A) {
                obj.team_a_score_ticks += 1;
            } else {
                obj.team_b_score_ticks += 1;
            }
            // Clean up any stale opposing cap progress.
            if (obj.cap_team == opp) {
                if (obj.cap_progress_ticks > 0) {
                    obj.cap_progress_ticks -= 1;
                }
                if (obj.cap_progress_ticks == 0) {
                    obj.cap_team = Team::Neutral;
                }
            } else {
                obj.cap_progress_ticks = 0;
                obj.cap_team = Team::Neutral;
            }
        } else {
            // Case 4: uncontested by non-owner — cap.
            if (obj.cap_team == present_team) {
                if (obj.cap_progress_ticks < common::kCaptureTicks) {
                    obj.cap_progress_ticks += 1;
                    if (obj.cap_progress_ticks == common::kCaptureTicks) {
                        obj.owner = present_team;
                        obj.cap_progress_ticks = 0;
                        obj.cap_team = Team::Neutral;
                    }
                }
            } else if (obj.cap_team == opp) {
                // Stale other-team progress — reset first.
                obj.cap_progress_ticks = 0;
                obj.cap_team = present_team;
            } else {
                obj.cap_team = present_team;
                obj.cap_progress_ticks = 0;
            }
        }
    }

    X2_INVARIANT(obj.team_a_score_ticks >= a_before, ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_b_score_ticks >= b_before, ErrorCode::CorruptState);
    X2_INVARIANT(obj.cap_progress_ticks <= common::kCaptureTicks, ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_a_score_ticks <= common::kWinTicks, ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_b_score_ticks <= common::kWinTicks, ErrorCode::CorruptState);
}

// --- Full tick pipeline. Steps 1–6 from Phase 0, 7 + 10–15 from Phase 1a. ---

void apply_one_tick(MatchState& state, const MatchConfig& config,
                    const std::array<Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    // Steps 1–3: validate + aim update.
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        const Action& a = actions[i];
        X2_INVARIANT(std::isfinite(a.move_x) && std::isfinite(a.move_y),
                     ErrorCode::NonFiniteFloat);
        X2_INVARIANT(std::isfinite(a.aim_delta), ErrorCode::NonFiniteFloat);

        if (!aim_consumed[i]) {
            const float delta =
                common::clampf(a.aim_delta, -common::kAimDeltaMax, common::kAimDeltaMax);
            h.aim_angle = common::wrap_angle(h.aim_angle + delta);
        }
    }

    // Steps 4–5: movement + wall/bounds collision.
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        const Action& a = actions[i];
        Vec2 move_vec = common::normalize_move_input(Vec2{a.move_x, a.move_y});
        const float speed = hero_speed(h.kind);
        h.velocity = common::scale(move_vec, speed);
        Vec2 next = common::add(h.position, common::scale(h.velocity, kDt));
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        h.position = next;
    }

    // Step 6: cooldown timers + weapon state advance (for living Rangers).
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
        if (h.alive && h.kind == HeroKind::Ranger) {
            weapon_tick_update(h.weapon);
        }
    }

    // Step 7: abilities — Combat Roll (Phase 1 only impulse ability wired up).
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = state.heroes[i];
        if (!h.present || h.kind != HeroKind::Ranger) {
            continue;
        }
        maybe_combat_roll(h, actions[i], aim_consumed[i], config.map);
    }

    // Steps 8–9 skipped in Phase 1: no spatial index needed at 2 heroes;
    // no fog until Phase 7.

    // Step 10: fire resolution (hitscan).
    DamageBuffer buf{};
    std::array<bool, kAgentsPerMatch> has_damage{};
    resolve_revolver_fire(state, actions, config.mechanics, buf, has_damage);

    // Steps 11–12: apply accumulated damage simultaneously.
    apply_damage_buffer(state, buf, has_damage);

    // Step 13: process deaths (credits kills).
    process_deaths(state, buf, has_damage, config);

    // Step 14: respawn timers.
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        respawn_tick_update(state.heroes[i], state.tick, config);
    }

    // Step 15: objective control.
    objective_tick_update(state.objective, state.heroes, state.tick, config.map);

    // Steps 16–18: rewards / obs / replay — deferred to Phase 1b/1c.

    state.tick += 1;
}

// --- Deterministic state hashing (FNV-1a 64). ---
// Manifest of included fields lives in docs/determinism_rules.md
// §"state_hash() manifest". Every new HeroState / ObjectiveState field
// gets a line here.

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

inline void hash_bytes(std::uint64_t& h, const void* data, std::size_t n) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= kFnvPrime;
    }
}

void hash_i32(std::uint64_t& h, std::int32_t v) { hash_bytes(h, &v, sizeof(v)); }
void hash_u32(std::uint64_t& h, std::uint32_t v) { hash_bytes(h, &v, sizeof(v)); }
void hash_u8 (std::uint64_t& h, std::uint8_t  v) { hash_bytes(h, &v, sizeof(v)); }

void hash_weapon(std::uint64_t& h, const RangerWeaponState& w) {
    hash_u8 (h, w.magazine);
    hash_u8 (h, w.reloading ? 1U : 0U);
    hash_u32(h, w.reload_ticks_left);
    hash_u32(h, w.ticks_since_last_shot);
    hash_u32(h, w.fire_cooldown_ticks);
}

void hash_hero(std::uint64_t& h, const HeroState& hero) {
    hash_u32(h, hero.id);
    hash_u8 (h, static_cast<std::uint8_t>(hero.team));
    hash_u8 (h, static_cast<std::uint8_t>(hero.kind));
    hash_u8 (h, static_cast<std::uint8_t>(hero.role));
    hash_u8 (h, hero.present ? 1U : 0U);
    hash_u8 (h, hero.alive ? 1U : 0U);
    hash_i32(h, common::quantize_pos(hero.position.x));
    hash_i32(h, common::quantize_pos(hero.position.y));
    hash_i32(h, common::quantize_pos(hero.velocity.x));
    hash_i32(h, common::quantize_pos(hero.velocity.y));
    const std::int32_t aim_q = static_cast<std::int32_t>(
        std::nearbyint(hero.aim_angle * 10000.0F));
    hash_i32(h, aim_q);
    hash_i32(h, hero.health_centi_hp);
    hash_u32(h, hero.respawn_tick);
    hash_u32(h, hero.cd_ability_1);
    hash_u32(h, hero.cd_ability_2);
    hash_weapon(h, hero.weapon);
    hash_u8 (h, hero.vanguard_barrier_active ? 1U : 0U);
    hash_i32(h, hero.vanguard_barrier_hp_centi);
    hash_u8 (h, static_cast<std::uint8_t>(hero.mender_weapon));
    hash_u32(h, hero.mender_beam_locked_on);
    hash_u32(h, hero.kills);
    hash_u32(h, hero.deaths);
}

void hash_objective(std::uint64_t& h, const ObjectiveState& obj) {
    hash_u8 (h, static_cast<std::uint8_t>(obj.owner));
    hash_u8 (h, static_cast<std::uint8_t>(obj.cap_team));
    hash_u32(h, obj.cap_progress_ticks);
    hash_u32(h, obj.team_a_score_ticks);
    hash_u32(h, obj.team_b_score_ticks);
    hash_u8 (h, obj.unlocked ? 1U : 0U);
}

// mt19937_64 exposes its full state via operator<<. Hash the textual form —
// out of the per-tick hot path, so a per-call stringstream alloc is OK.
void hash_rng(std::uint64_t& h, const std::mt19937_64& rng) {
    std::ostringstream os;
    os << rng;
    const std::string s = os.str();
    hash_bytes(h, s.data(), s.size());
}

// --- Config validation. Sim ctor rejects any MatchConfig whose
// Phase1MechanicsConfig still has a sentinel value. See
// docs/coding_philosophy.md §3. ---

void validate_mechanics(const Phase1MechanicsConfig& m) {
    X2_REQUIRE(m.revolver_damage_centi_hp != std::numeric_limits<std::uint32_t>::max(),
               ErrorCode::CorruptState);
    X2_REQUIRE(m.revolver_damage_centi_hp > 0U, ErrorCode::CorruptState);
    X2_REQUIRE(m.revolver_fire_cooldown_ticks != std::numeric_limits<std::uint32_t>::max(),
               ErrorCode::CorruptState);
    X2_REQUIRE(m.revolver_fire_cooldown_ticks >= 1U, ErrorCode::CorruptState);
    X2_REQUIRE(std::isfinite(m.revolver_hitbox_radius), ErrorCode::CorruptState);
    X2_REQUIRE(m.revolver_hitbox_radius > 0.0F, ErrorCode::CorruptState);
    X2_REQUIRE(m.respawn_ticks != std::numeric_limits<std::uint32_t>::max(),
               ErrorCode::CorruptState);
    X2_REQUIRE(m.respawn_ticks > 0U, ErrorCode::CorruptState);
}

}  // namespace

Sim::Sim(const MatchConfig& config) : config_(config) {
    X2_REQUIRE(config.action_repeat == 2 || config.action_repeat == 3,
               ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_x > config.map.min_x, ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_y > config.map.min_y, ErrorCode::CorruptState);
    validate_mechanics(config.mechanics);
    reset_state(state_, config_);
}

void Sim::reset() { reset_state(state_, config_); }

void Sim::reset(std::uint64_t seed) {
    config_.seed = seed;
    reset_state(state_, config_);
}

void Sim::step(std::array<Action, kAgentsPerMatch> actions) {
    for (auto& a : actions) {
        common::canonicalize_action(a);
    }
    std::array<bool, kAgentsPerMatch> aim_consumed{};
    apply_one_tick(state_, config_, actions, aim_consumed);
}

void Sim::step_decision(std::array<Action, kAgentsPerMatch> actions) {
    for (auto& a : actions) {
        common::canonicalize_action(a);
    }
    const std::uint32_t repeat = config_.action_repeat;
    X2_REQUIRE(repeat == 2 || repeat == 3, ErrorCode::CorruptState);

    std::array<bool, kAgentsPerMatch> aim_consumed{};  // starts all-false
    for (std::uint32_t k = 0; k < repeat; ++k) {
        apply_one_tick(state_, config_, actions, aim_consumed);
        for (auto& c : aim_consumed) {
            c = true;
        }
    }
}

bool Sim::episode_over() const noexcept {
    if (state_.objective.team_a_score_ticks >= common::kWinTicks ||
        state_.objective.team_b_score_ticks >= common::kWinTicks) {
        return true;
    }
    const Tick max_ticks =
        static_cast<Tick>(config_.round_length_seconds * kTickHz);
    return state_.tick >= max_ticks;
}

Team Sim::winner() const noexcept {
    if (state_.objective.team_a_score_ticks >= common::kWinTicks) {
        return Team::A;
    }
    if (state_.objective.team_b_score_ticks >= common::kWinTicks) {
        return Team::B;
    }
    // Timeout: higher score wins; exact tie = draw (Neutral).
    const Tick max_ticks =
        static_cast<Tick>(config_.round_length_seconds * kTickHz);
    if (state_.tick < max_ticks) {
        return Team::Neutral;  // episode not over
    }
    if (state_.objective.team_a_score_ticks > state_.objective.team_b_score_ticks) {
        return Team::A;
    }
    if (state_.objective.team_b_score_ticks > state_.objective.team_a_score_ticks) {
        return Team::B;
    }
    return Team::Neutral;
}

std::uint32_t Sim::team_a_kills() const noexcept {
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        if (state_.heroes[i].present && state_.heroes[i].team == Team::A) {
            total += state_.heroes[i].kills;
        }
    }
    return total;
}

std::uint32_t Sim::team_b_kills() const noexcept {
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        if (state_.heroes[i].present && state_.heroes[i].team == Team::B) {
            total += state_.heroes[i].kills;
        }
    }
    return total;
}

std::uint64_t Sim::state_hash() const noexcept {
    std::uint64_t h = kFnvOffset;
    hash_u32(h, state_.tick);
    for (const auto& hero : state_.heroes) {
        hash_hero(h, hero);
    }
    hash_objective(h, state_.objective);
    hash_rng(h, state_.rng);
    return h;
}

}  // namespace xushi2::sim
