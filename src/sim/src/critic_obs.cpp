// Phase-1 critic (centralized) observation builder.
//
// The critic obs is a superset of the actor obs for the given team's
// Ranger slot. The first kActorObsPhase1Dim floats are produced by the
// public actor builder so actor and critic stay aligned on that prefix by
// construction. The remainder carries privileged information (world-frame
// absolute positions and velocities, raw tick counters, match seed bits)
// that the actor MUST NEVER see.
//
// Field order and widths MUST match python/xushi2/obs_manifest.py
// CRITIC_PHASE1_FIELDS.

#include <xushi2/sim/obs.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/obs_utils.h>
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

// Find the Phase-1 Ranger slot for a team. Returns kAgentsPerMatch if the
// team has no occupied Ranger (should not happen at Phase 1).
std::uint32_t find_team_ranger_slot(const MatchState& s,
                                    common::Team team) noexcept {
    for (std::uint32_t i = 0; i < s.heroes.size(); ++i) {
        const auto& h = s.heroes[i];
        if (h.present && h.team == team) {
            return i;
        }
    }
    return static_cast<std::uint32_t>(s.heroes.size());
}

// Normalize a uint32 to [-1, 1] by dividing by 2^32. For seed bits we lose
// precision but that is fine — these fields are included so the critic can
// in principle distinguish maps; Phase-1 has a single map so precision is
// not load-bearing.
float norm_u32(std::uint32_t v) noexcept {
    constexpr float kTwoPow32 = 4294967296.0F;
    return 2.0F * (static_cast<float>(v) / kTwoPow32) - 1.0F;
}

}  // namespace

void build_critic_obs_phase1(const Sim& sim,
                             common::Team team_perspective,
                             float* out_buffer,
                             std::uint32_t out_capacity) noexcept {
    X2_REQUIRE(out_buffer != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(out_capacity >= kCriticObsPhase1Dim,
               common::ErrorCode::CapacityExceeded);
    X2_REQUIRE(team_perspective == common::Team::A ||
                   team_perspective == common::Team::B,
               common::ErrorCode::InvalidHeroId);

    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();

    const std::uint32_t team_slot =
        find_team_ranger_slot(s, team_perspective);
    X2_REQUIRE(team_slot < s.heroes.size(),
               common::ErrorCode::InvalidHeroId);

    // --- First kActorObsPhase1Dim floats: actor-shape mirror for this team.
    // We call the public actor builder because the critic's first N floats
    // must be exactly what the actor of that team would see. This keeps the
    // two surfaces aligned by construction.
    build_actor_obs_phase1(sim, team_slot, out_buffer,
                           kActorObsPhase1Dim);

    // --- Privileged section. Cursor picks up after the actor prefix. ---
    Writer w{out_buffer, kActorObsPhase1Dim};

    // Find the opposite team's Ranger slot. If missing, degenerate to zeros.
    const common::Team enemy_team =
        (team_perspective == common::Team::A) ? common::Team::B
                                              : common::Team::A;
    const std::uint32_t enemy_slot = find_team_ranger_slot(s, enemy_team);

    // world_own_position, world_enemy_position — no mirror, raw world coords.
    const HeroState& own = s.heroes[team_slot];
    w.push2(own.position.x, own.position.y);
    if (enemy_slot < s.heroes.size() && s.heroes[enemy_slot].present) {
        w.push2(s.heroes[enemy_slot].position.x,
                s.heroes[enemy_slot].position.y);
    } else {
        w.push2(0.0F, 0.0F);
    }

    // world_own_velocity, world_enemy_velocity — no mirror, raw.
    w.push2(own.velocity.x, own.velocity.y);
    if (enemy_slot < s.heroes.size() && s.heroes[enemy_slot].present) {
        w.push2(s.heroes[enemy_slot].velocity.x,
                s.heroes[enemy_slot].velocity.y);
    } else {
        w.push2(0.0F, 0.0F);
    }

    // Raw tick counters.
    w.push1(static_cast<float>(s.objective.cap_progress_ticks));
    w.push1(static_cast<float>(s.objective.team_a_score_ticks));
    w.push1(static_cast<float>(s.objective.team_b_score_ticks));
    w.push1(static_cast<float>(s.tick));

    // Seed bits, normalized to [-1, 1].
    const std::uint64_t seed = cfg.seed;
    const std::uint32_t seed_hi =
        static_cast<std::uint32_t>((seed >> 32) & 0xFFFFFFFFULL);
    const std::uint32_t seed_lo =
        static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL);
    w.push1(norm_u32(seed_hi));
    w.push1(norm_u32(seed_lo));

    X2_ENSURE(w.cursor == kCriticObsPhase1Dim,
              common::ErrorCode::CapacityExceeded);
}

}  // namespace xushi2::sim
