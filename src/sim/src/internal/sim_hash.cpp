#include "sim_hash.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>

#include <xushi2/common/math.hpp>

namespace xushi2::sim::internal {

// --- Deterministic state hashing (FNV-1a 64). ---
// Manifest of included fields lives in docs/determinism_rules.md
// §"state_hash() manifest". Every new HeroState / ObjectiveState field
// gets a line here.

static constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
static constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

static inline void hash_bytes(std::uint64_t& h, const void* data, std::size_t n) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= kFnvPrime;
    }
}

static void hash_i32(std::uint64_t& h, std::int32_t v) { hash_bytes(h, &v, sizeof(v)); }
static void hash_u32(std::uint64_t& h, std::uint32_t v) { hash_bytes(h, &v, sizeof(v)); }
static void hash_u8 (std::uint64_t& h, std::uint8_t  v) { hash_bytes(h, &v, sizeof(v)); }

static void hash_weapon(std::uint64_t& h, const RangerWeaponState& w) {
    hash_u8 (h, w.magazine);
    hash_u8 (h, w.reloading ? 1U : 0U);
    hash_u32(h, w.reload_ticks_left);
    hash_u32(h, w.ticks_since_last_shot);
    hash_u32(h, w.fire_cooldown_ticks);
}

static void hash_hero(std::uint64_t& h, const HeroState& hero) {
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

static void hash_objective(std::uint64_t& h, const ObjectiveState& obj) {
    hash_u8 (h, static_cast<std::uint8_t>(obj.owner));
    hash_u8 (h, static_cast<std::uint8_t>(obj.cap_team));
    hash_u32(h, obj.cap_progress_ticks);
    hash_u32(h, obj.team_a_score_ticks);
    hash_u32(h, obj.team_b_score_ticks);
    hash_u8 (h, obj.unlocked ? 1U : 0U);
}

// mt19937_64 exposes its full state via operator<<. Hash the textual form —
// out of the per-tick hot path, so a per-call stringstream alloc is OK.
static void hash_rng(std::uint64_t& h, const std::mt19937_64& rng) {
    std::ostringstream os;
    os << rng;
    const std::string s = os.str();
    hash_bytes(h, s.data(), s.size());
}

std::uint64_t compute_state_hash(const MatchState& state) {
    std::uint64_t h = kFnvOffset;
    hash_u32(h, state.tick);
    for (const auto& hero : state.heroes) {
        hash_hero(h, hero);
    }
    hash_objective(h, state.objective);
    hash_rng(h, state.rng);
    return h;
}

}  // namespace xushi2::sim::internal
