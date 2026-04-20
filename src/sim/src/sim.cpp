#include <xushi2/sim/sim.h>

#include <sstream>
#include <string>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/math.hpp>

// Phase-0 playable slice. Two Rangers on a 50×50 arena. Movement + aim only,
// no combat, no objective scoring yet, no fog. This is the smallest
// deterministic slice that exercises reset / step / replay / state_hash.
//
// See docs/coding_philosophy.md §13 for the Tier 0 review checklist.

namespace xushi2::sim {

namespace {

using common::ErrorCode;
using common::HeroKind;
using common::Team;
using common::Vec2;

// Hero movement speeds per game-design §6 (u/s).
inline constexpr float kRangerSpeed = 4.2F;

// Ranger max HP per game-design §6.
inline constexpr float kRangerMaxHp = 150.0F;

float hero_speed(HeroKind kind) {
    switch (kind) {
        case HeroKind::Vanguard:
            return 3.6F;
        case HeroKind::Ranger:
            return 4.2F;
        case HeroKind::Mender:
            return 4.0F;
    }
    X2_UNREACHABLE();
}

void reset_state(MatchState& state, const MatchConfig& config) {
    state = MatchState{};
    state.rng.seed(config.seed);
    state.objective.owner = Team::Neutral;
    state.objective.cap_team = Team::Neutral;
    state.objective.cap_progress_ticks = 0;
    state.objective.team_a_score_ticks = 0;
    state.objective.team_b_score_ticks = 0;
    state.objective.unlocked = false;

    // Phase 0: two Rangers, one per team, at fixed spawn points on the
    // Y-axis of the arena. All other hero slots are absent.
    const float cx = 0.5F * (config.map.min_x + config.map.max_x);
    const float span_y = config.map.max_y - config.map.min_y;

    HeroState& a = state.heroes[0];
    a.id = 1;
    a.team = Team::A;
    a.kind = HeroKind::Ranger;
    a.role = Role::Damage;
    a.position = Vec2{cx, config.map.min_y + 0.1F * span_y};
    a.aim_angle = 0.5F * common::kPi;  // facing +y (toward center)
    a.health = kRangerMaxHp;
    a.max_health = kRangerMaxHp;
    a.alive = true;
    a.ranger_magazine = 6;
    a.present = true;

    HeroState& b = state.heroes[3];  // slot 3 = team B first damage
    b.id = 2;
    b.team = Team::B;
    b.kind = HeroKind::Ranger;
    b.role = Role::Damage;
    b.position = Vec2{cx, config.map.min_y + 0.9F * span_y};
    b.aim_angle = -0.5F * common::kPi;  // facing -y (toward center)
    b.health = kRangerMaxHp;
    b.max_health = kRangerMaxHp;
    b.alive = true;
    b.ranger_magazine = 6;
    b.present = true;
}

// Apply one sim tick with the given (canonical) actions.
// `aim_consumed` is a per-agent mask: if true, skip the aim-delta update
// because this is not the first tick of the policy decision. See
// docs/action_spec.md "Per-decision vs per-tick."
void apply_one_tick(MatchState& state, const MatchConfig& config,
                    const std::array<Action, kAgentsPerMatch>& actions,
                    const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    // 1–3. Validate + aim update.
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

    // 4–5. Movement + wall/bounds collision.
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
        // Clamp to arena bounds (Phase 0 has no interior walls).
        next.x = common::clampf(next.x, config.map.min_x, config.map.max_x);
        next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
        h.position = next;
    }

    // 6. Cooldown timers. (No ability logic wired yet; this just decrements.)
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
    }

    // 7–16. TODO: abilities, fire, damage, deaths, respawn, objective,
    //              rewards — land with subsequent phases.

    state.tick += 1;
}

// --- Deterministic state hashing (FNV-1a 64) ---
//
// Manifest of included fields lives in docs/determinism_rules.md §"state_hash()
// manifest". If you add state, update both that doc and this function.

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
void hash_u64(std::uint64_t& h, std::uint64_t v) { hash_bytes(h, &v, sizeof(v)); }
void hash_u8(std::uint64_t& h, std::uint8_t v) { hash_bytes(h, &v, sizeof(v)); }

void hash_hero(std::uint64_t& h, const HeroState& hero) {
    hash_u32(h, hero.id);
    hash_u8(h, static_cast<std::uint8_t>(hero.team));
    hash_u8(h, static_cast<std::uint8_t>(hero.kind));
    hash_u8(h, static_cast<std::uint8_t>(hero.role));
    hash_u8(h, hero.present ? 1U : 0U);
    hash_u8(h, hero.alive ? 1U : 0U);
    hash_i32(h, common::quantize_pos(hero.position.x));
    hash_i32(h, common::quantize_pos(hero.position.y));
    hash_i32(h, common::quantize_pos(hero.velocity.x));
    hash_i32(h, common::quantize_pos(hero.velocity.y));
    // Aim angle: quantize to 1/10000 rad (fine enough for ±π range).
    const std::int32_t aim_q = static_cast<std::int32_t>(
        std::nearbyint(hero.aim_angle * 10000.0F));
    hash_i32(h, aim_q);
    hash_i32(h, common::quantize_hp(hero.health));
    hash_u32(h, hero.respawn_tick);
    hash_u32(h, hero.cd_ability_1);
    hash_u32(h, hero.cd_ability_2);
    hash_u8(h, hero.vanguard_barrier_active ? 1U : 0U);
    hash_i32(h, common::quantize_hp(hero.vanguard_barrier_hp));
    hash_u8(h, hero.ranger_magazine);
    hash_u8(h, hero.ranger_reloading ? 1U : 0U);
    hash_u8(h, static_cast<std::uint8_t>(hero.mender_weapon));
    hash_u32(h, hero.mender_beam_locked_on);
}

void hash_objective(std::uint64_t& h, const ObjectiveState& obj) {
    hash_u8(h, static_cast<std::uint8_t>(obj.owner));
    hash_u8(h, static_cast<std::uint8_t>(obj.cap_team));
    hash_u32(h, obj.cap_progress_ticks);
    hash_u32(h, obj.team_a_score_ticks);
    hash_u32(h, obj.team_b_score_ticks);
    hash_u8(h, obj.unlocked ? 1U : 0U);
}

// mt19937_64 exposes its full state via operator<<. We hash the textual
// representation. This is Tier 0 debug/verification — out of the per-tick
// hot path — so a per-call stringstream allocation is acceptable.
void hash_rng(std::uint64_t& h, const std::mt19937_64& rng) {
    std::ostringstream os;
    os << rng;
    const std::string s = os.str();
    hash_bytes(h, s.data(), s.size());
}

}  // namespace

Sim::Sim(const MatchConfig& config) : config_(config) {
    X2_REQUIRE(config.action_repeat == 2 || config.action_repeat == 3,
               ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_x > config.map.min_x, ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_y > config.map.min_y, ErrorCode::CorruptState);
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
    std::array<bool, kAgentsPerMatch> aim_consumed{};  // all false
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
        // After the first tick of the decision, the aim delta has been
        // applied; subsequent ticks must not re-apply it.
        for (auto& c : aim_consumed) {
            c = true;
        }
    }
}

bool Sim::episode_over() const noexcept {
    // Win threshold: 100 s of scoring = 100 * TICK_HZ ticks per game-design §3.
    const std::uint32_t win_ticks =
        100U * static_cast<std::uint32_t>(kTickHz);
    if (state_.objective.team_a_score_ticks >= win_ticks ||
        state_.objective.team_b_score_ticks >= win_ticks) {
        return true;
    }
    const Tick max_ticks =
        static_cast<Tick>(config_.round_length_seconds * kTickHz);
    return state_.tick >= max_ticks;
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
