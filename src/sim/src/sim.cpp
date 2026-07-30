#include <xushi2/sim/sim.h>

#include <cmath>
#include <limits>

#include <xushi2/common/action_canon.hpp>
#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

#include "internal/sim_combat.h"
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

void validate_cover(const MatchConfig& config) {
    X2_REQUIRE(config.num_cover_circles <= common::kMaxWalls,
               ErrorCode::CapacityExceeded);
    for (std::uint32_t i = 0; i < config.num_cover_circles; ++i) {
        const CoverCircle& cover = config.cover_circles[i];
        X2_REQUIRE(std::isfinite(cover.center.x) && std::isfinite(cover.center.y),
                   ErrorCode::NonFiniteFloat);
        X2_REQUIRE(std::isfinite(cover.radius) && cover.radius > 0.0F,
                   ErrorCode::CorruptState);
        X2_REQUIRE(cover.center.x - cover.radius >= config.map.min_x &&
                   cover.center.x + cover.radius <= config.map.max_x &&
                   cover.center.y - cover.radius >= config.map.min_y &&
                   cover.center.y + cover.radius <= config.map.max_y,
                   ErrorCode::CorruptState);
    }
    X2_REQUIRE(config.num_wall_segments <= common::kMaxWalls,
               ErrorCode::CapacityExceeded);
    for (std::uint32_t i = 0; i < config.num_wall_segments; ++i) {
        const WallSegment& wall = config.wall_segments[i];
        X2_REQUIRE(std::isfinite(wall.a.x) && std::isfinite(wall.a.y) &&
                   std::isfinite(wall.b.x) && std::isfinite(wall.b.y),
                   ErrorCode::NonFiniteFloat);
        X2_REQUIRE(std::isfinite(wall.half_width) && wall.half_width > 0.0F,
                   ErrorCode::CorruptState);
        const float dx = wall.b.x - wall.a.x;
        const float dy = wall.b.y - wall.a.y;
        X2_REQUIRE(dx * dx + dy * dy > 1e-6F, ErrorCode::CorruptState);
        X2_REQUIRE(wall.a.x - wall.half_width >= config.map.min_x &&
                   wall.a.x + wall.half_width <= config.map.max_x &&
                   wall.a.y - wall.half_width >= config.map.min_y &&
                   wall.a.y + wall.half_width <= config.map.max_y &&
                   wall.b.x - wall.half_width >= config.map.min_x &&
                   wall.b.x + wall.half_width <= config.map.max_x &&
                   wall.b.y - wall.half_width >= config.map.min_y &&
                   wall.b.y + wall.half_width <= config.map.max_y,
                   ErrorCode::CorruptState);
    }
}

void validate_objective_timing(const MatchConfig& config) {
    X2_REQUIRE(config.objective_unlock_ticks > 0U, ErrorCode::CorruptState);
    X2_REQUIRE(config.objective_capture_ticks > 0U, ErrorCode::CorruptState);
}

}  // namespace

Sim::Sim(const MatchConfig& config) : config_(config) {
    X2_REQUIRE(config.action_repeat == 2 || config.action_repeat == 3,
               ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_x > config.map.min_x, ErrorCode::CorruptState);
    X2_REQUIRE(config.map.max_y > config.map.min_y, ErrorCode::CorruptState);
    X2_REQUIRE(config.team_size == 1 || config.team_size == 3,
               ErrorCode::CorruptState);
    validate_objective_timing(config);
    validate_mechanics(config.mechanics);
    validate_cover(config);
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

void Sim::set_objective_timing_ticks(std::uint32_t unlock_ticks,
                                     std::uint32_t capture_ticks) {
    X2_REQUIRE(unlock_ticks > 0U, ErrorCode::CorruptState);
    X2_REQUIRE(capture_ticks > 0U, ErrorCode::CorruptState);
    config_.objective_unlock_ticks = unlock_ticks;
    config_.objective_capture_ticks = capture_ticks;
    if (state_.objective.cap_progress_ticks >= capture_ticks) {
        state_.objective.cap_progress_ticks = capture_ticks - 1U;
    }
}

bool Sim::score_threshold_reached() const noexcept {
    return state_.objective.team_a_score_ticks >= common::kWinTicks ||
           state_.objective.team_b_score_ticks >= common::kWinTicks;
}

bool Sim::round_timer_expired() const noexcept {
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

std::array<std::uint32_t, kAgentsPerMatch> Sim::kills_by_slot() const noexcept {
    std::array<std::uint32_t, kAgentsPerMatch> out{};
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        out[i] = state_.heroes[i].kills;
    }
    return out;
}

std::array<std::uint32_t, kAgentsPerMatch> Sim::deaths_by_slot() const noexcept {
    std::array<std::uint32_t, kAgentsPerMatch> out{};
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        out[i] = state_.heroes[i].deaths;
    }
    return out;
}

std::array<std::uint64_t, kAgentsPerMatch> Sim::damage_dealt_by_slot() const noexcept {
    std::array<std::uint64_t, kAgentsPerMatch> out{};
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        out[i] = state_.heroes[i].damage_dealt_centi_hp;
    }
    return out;
}

std::uint64_t Sim::state_hash() const noexcept {
    return internal::compute_state_hash(state_);
}

bool Sim::line_of_sight(std::uint32_t from_slot,
                        std::uint32_t to_slot) const noexcept {
    if (from_slot >= state_.heroes.size() || to_slot >= state_.heroes.size()) {
        return false;
    }
    const HeroState& from = state_.heroes[from_slot];
    const HeroState& to = state_.heroes[to_slot];
    if (!from.present || !to.present || !from.alive || !to.alive) {
        return false;
    }
    return !internal::segment_blocked_by_cover(from.position, to.position, config_);
}

}  // namespace xushi2::sim
