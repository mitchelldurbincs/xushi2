#include <xushi2/sim/sim.h>

#include <cmath>
#include <limits>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

#include "internal/sim_hash.h"
#include "internal/sim_spawn_reset.h"
#include "internal/sim_tick_pipeline.h"

// Phase-1a playable slice. 1v1 Ranger on a 50×50 arena. Movement + aim +
// combat (hitscan Revolver + magazine/reload state machine), Combat Roll,
// death + respawn, full 5-case objective state machine. No fog, no walls.
//
// See docs/coding_philosophy.md §13 for the Tier 0 review checklist applied
// here.

namespace xushi2::sim {

namespace {

using common::ErrorCode;

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
    X2_REQUIRE(config.team_size == 1 || config.team_size == 3,
               ErrorCode::CorruptState);
    validate_mechanics(config.mechanics);
    internal::reset_state(state_, config_);
}

void Sim::reset() { internal::reset_state(state_, config_); }

void Sim::reset(std::uint64_t seed) {
    config_.seed = seed;
    internal::reset_state(state_, config_);
}

void Sim::step(std::array<Action, kAgentsPerMatch> actions) {
    for (auto& a : actions) {
        common::canonicalize_action(a);
    }
    std::array<bool, kAgentsPerMatch> aim_consumed{};
    internal::apply_one_tick(state_, config_, actions, aim_consumed);
}

void Sim::step_decision(std::array<Action, kAgentsPerMatch> actions) {
    for (auto& a : actions) {
        common::canonicalize_action(a);
    }
    const std::uint32_t repeat = config_.action_repeat;
    X2_REQUIRE(repeat == 2 || repeat == 3, ErrorCode::CorruptState);

    std::array<bool, kAgentsPerMatch> aim_consumed{};  // starts all-false
    for (std::uint32_t k = 0; k < repeat; ++k) {
        internal::apply_one_tick(state_, config_, actions, aim_consumed);
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
    return internal::compute_state_hash(state_);
}

}  // namespace xushi2::sim
