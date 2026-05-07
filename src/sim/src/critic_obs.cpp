// Phase-4 critic (centralized) observation builder.
//
// Layout (135 floats):
//   - 3 own-team actor mirrors (3 × kActorObsPhase1Dim = 93 floats),
//     emitted by calling `build_actor_obs_phase1` for each of the team's
//     three Ranger slots in ascending slot order.
//   - 3 enemy world-frame blocks (3 × 12 = 36 floats), one per enemy
//     Ranger in ascending slot order: (position, velocity, aim_unit,
//     hp_normalized, alive_flag, respawn_timer, ammo, reloading,
//     combat_roll_cd).
//   - 4 raw objective tick counters (cap_progress, team_a_score,
//     team_b_score, sim tick).
//   - 2 normalized seed bits (hi, lo), each in [-1, 1].
//
// Field order and widths MUST match python/xushi2/obs_manifest.py
// CRITIC_FIELDS. Requires `MatchConfig::team_size == 3`.

#include <xushi2/sim/obs.h>

#include <array>
#include <cmath>
#include <cstdint>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace xushi2::sim {

namespace {

struct Writer {
    float* out;
    std::uint32_t cursor;
    void push1(float v) noexcept {
        X2_ENSURE(std::isfinite(v), common::ErrorCode::NonFiniteFloat);
        out[cursor++] = v;
    }
    void push2(float a, float b) noexcept { push1(a); push1(b); }
};

float clamp01(float v) noexcept {
    if (v < 0.0F) return 0.0F;
    if (v > 1.0F) return 1.0F;
    return v;
}

float norm_u32(std::uint32_t v) noexcept {
    constexpr float kTwoPow32 = 4294967296.0F;
    return 2.0F * (static_cast<float>(v) / kTwoPow32) - 1.0F;
}

std::array<std::uint32_t, 3> find_team_ranger_slots(
        const MatchState& s, common::Team team) noexcept {
    std::array<std::uint32_t, 3> slots{
        static_cast<std::uint32_t>(s.heroes.size()),
        static_cast<std::uint32_t>(s.heroes.size()),
        static_cast<std::uint32_t>(s.heroes.size()),
    };
    std::uint32_t found = 0;
    for (std::uint32_t i = 0; i < s.heroes.size() && found < 3; ++i) {
        const auto& h = s.heroes[i];
        if (h.present && h.team == team) {
            slots[found++] = i;
        }
    }
    X2_REQUIRE(found == 3, common::ErrorCode::InvalidHeroId);
    return slots;
}

void emit_enemy_world_block(Writer& w,
                            const HeroState& h,
                            common::Tick now,
                            const MatchConfig& cfg) noexcept {
    // World-frame position and velocity, no mirror.
    w.push2(h.position.x, h.position.y);
    w.push2(h.velocity.x, h.velocity.y);

    // World-frame aim as (sin, cos) — no mirror, raw aim_angle.
    w.push2(std::sin(h.aim_angle), std::cos(h.aim_angle));

    // hp_normalized.
    const float hp = (h.max_health_centi_hp > 0)
        ? (static_cast<float>(h.health_centi_hp) /
           static_cast<float>(h.max_health_centi_hp))
        : 0.0F;
    w.push1(hp);

    // alive_flag.
    w.push1(h.alive ? 1.0F : 0.0F);

    // respawn_timer normalized to [0, 1] using cfg.mechanics.respawn_ticks.
    // Mirrors the actor's enemy_respawn_timer convention exactly.
    float respawn_norm = 0.0F;
    if (h.present && !h.alive && cfg.mechanics.respawn_ticks > 0U) {
        std::int64_t remaining =
            static_cast<std::int64_t>(h.respawn_tick) -
            static_cast<std::int64_t>(now);
        if (remaining < 0) remaining = 0;
        respawn_norm = clamp01(
            static_cast<float>(remaining) /
            static_cast<float>(cfg.mechanics.respawn_ticks));
    }
    w.push1(respawn_norm);

    // ammo.
    const float ammo = (common::kRangerMaxMagazine > 0)
        ? (static_cast<float>(h.weapon.magazine) /
           static_cast<float>(common::kRangerMaxMagazine))
        : 0.0F;
    w.push1(ammo);

    // reloading: same predicate the actor builder uses.
    w.push1(h.weapon.reloading ? 1.0F : 0.0F);

    // combat_roll_cd normalized.
    const float roll_cd = (common::kRangerCombatRollCooldownTicks > 0)
        ? clamp01(static_cast<float>(h.cd_ability_1) /
                  static_cast<float>(common::kRangerCombatRollCooldownTicks))
        : 0.0F;
    w.push1(roll_cd);
}

}  // namespace

void build_critic_obs(const Sim& sim,
                      common::Team team_perspective,
                      float* out_buffer,
                      std::uint32_t out_capacity) noexcept {
    X2_REQUIRE(out_buffer != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(out_capacity >= kCriticObsDim,
               common::ErrorCode::CapacityExceeded);
    X2_REQUIRE(team_perspective == common::Team::A ||
                   team_perspective == common::Team::B,
               common::ErrorCode::InvalidHeroId);

    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();
    X2_REQUIRE(cfg.team_size == 3, common::ErrorCode::CorruptState);

    // 1) Three own-team actor mirrors, each kActorObsPhase1Dim floats.
    const auto own_slots = find_team_ranger_slots(s, team_perspective);
    for (std::uint32_t i = 0; i < 3; ++i) {
        build_actor_obs_phase1(sim, own_slots[i],
                               out_buffer + i * kActorObsPhase1Dim,
                               kActorObsPhase1Dim);
    }

    Writer w{out_buffer, 3U * kActorObsPhase1Dim};

    // 2) Three enemy-team world-frame blocks, 12 floats each.
    const common::Team enemy_team =
        (team_perspective == common::Team::A) ? common::Team::B
                                              : common::Team::A;
    const auto enemy_slots = find_team_ranger_slots(s, enemy_team);
    for (std::uint32_t i = 0; i < 3; ++i) {
        emit_enemy_world_block(w, s.heroes[enemy_slots[i]], s.tick, cfg);
    }

    // 3) Raw objective counters.
    w.push1(static_cast<float>(s.objective.cap_progress_ticks));
    w.push1(static_cast<float>(s.objective.team_a_score_ticks));
    w.push1(static_cast<float>(s.objective.team_b_score_ticks));
    w.push1(static_cast<float>(s.tick));

    // 4) Seed bits, normalized to [-1, 1].
    const std::uint64_t seed = cfg.seed;
    const std::uint32_t seed_hi =
        static_cast<std::uint32_t>((seed >> 32) & 0xFFFFFFFFULL);
    const std::uint32_t seed_lo =
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL);
    w.push1(norm_u32(seed_hi));
    w.push1(norm_u32(seed_lo));

    X2_ENSURE(w.cursor == kCriticObsDim,
              common::ErrorCode::CapacityExceeded);
}

}  // namespace xushi2::sim
