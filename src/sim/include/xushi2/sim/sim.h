#pragma once

// Xushi2 simulation core. Deterministic, headless, pure game-state update.
// See docs/game_design.md for rules, docs/determinism_rules.md for the
// float-determinism discipline, docs/action_spec.md for the action contract,
// and docs/coding_philosophy.md for the Tier 0 rules this file obeys.

#include <array>
#include <cstdint>
#include <random>

#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>

namespace xushi2::sim {

using common::Action;
using common::EntityId;
using common::HeroKind;
using common::Role;
using common::Team;
using common::Tick;
using common::Vec2;

// Tick pipeline (see game-design.md §11). 30 Hz fixed.
inline constexpr int kTickHz = static_cast<int>(common::kTickHz);
inline constexpr float kDt = 1.0F / static_cast<float>(kTickHz);

inline constexpr int kTeamSize = static_cast<int>(common::kTeamSize);
inline constexpr int kAgentsPerMatch = 2 * kTeamSize;

// Phase-0 playable slice: a tiny rectangular arena with fixed bounds.
// The real map comes online at Phase 5 (game_design.md §5).
struct MapBounds {
    float min_x = 0.0F;
    float min_y = 0.0F;
    float max_x = 50.0F;
    float max_y = 50.0F;
};

// Per-agent full state. Actor-side observations are a *subset* of this;
// critic-side observations can use all of it. See docs/observation_spec.md.
struct HeroState {
    EntityId id = 0;
    Team team = Team::Neutral;
    HeroKind kind = HeroKind::Vanguard;
    Role role = Role::Tank;
    Vec2 position{};
    Vec2 velocity{};
    float aim_angle = 0.0F;             // radians, absolute
    float health = 0.0F;
    float max_health = 0.0F;
    bool alive = true;
    Tick respawn_tick = 0;              // if !alive, tick when respawn fires
    // Cooldowns (ticks remaining, 0 means ready).
    Tick cd_ability_1 = 0;
    Tick cd_ability_2 = 0;
    // Hero-specific state; not all fields apply to all heroes.
    bool vanguard_barrier_active = false;
    float vanguard_barrier_hp = 0.0F;
    std::uint8_t ranger_magazine = 6;   // 0..6
    bool ranger_reloading = false;
    common::MenderWeapon mender_weapon = common::MenderWeapon::Staff;
    EntityId mender_beam_locked_on = 0;  // 0 = not locked
    // Whether this slot is actually occupied in the Phase-0 playable slice.
    // At Phase 1+, all six slots are occupied.
    bool present = false;
};

// Control-point state machine (see game-design.md §3). Integer tick math.
struct ObjectiveState {
    Team owner = Team::Neutral;
    Team cap_team = Team::Neutral;              // Neutral == "None"
    std::uint32_t cap_progress_ticks = 0;       // 0..CAPTURE_TICKS
    std::uint32_t team_a_score_ticks = 0;       // 0..WIN_TICKS
    std::uint32_t team_b_score_ticks = 0;
    bool unlocked = false;                      // true after the 15s lock window
};

// Match configuration — seed, match length, fog-of-war toggle, etc.
struct MatchConfig {
    std::uint64_t seed = 0;
    int round_length_seconds = 180;
    bool fog_of_war_enabled = true;
    bool randomize_map = false;     // per-episode wall randomization (default off until Phase 8)
    // Sim ticks held per policy decision (see action_spec.md).
    std::uint32_t action_repeat = common::kDefaultActionRepeat;
    MapBounds map{};
};

// Opaque match state. Copyable so snapshots can be taken trivially.
struct MatchState {
    Tick tick = 0;
    std::array<HeroState, kAgentsPerMatch> heroes{};
    ObjectiveState objective{};
    std::mt19937_64 rng{};  // seeded from MatchConfig::seed
};

// The Sim class is the only public handle into the simulation.
// Intentionally small API. Python bindings wrap this directly.
class Sim {
   public:
    explicit Sim(const MatchConfig& config);

    // Reset to a fresh initial state. Seed is taken from the config stored
    // at construction unless overridden by reset(seed).
    void reset();
    void reset(std::uint64_t seed);

    // Advance one simulation tick by applying the provided per-agent actions.
    // `actions` length must equal kAgentsPerMatch. Actions are canonicalized
    // before use (action_spec.md); the canonical form is what the sim sees.
    void step(std::array<Action, kAgentsPerMatch> actions);

    // Advance one policy decision (config.action_repeat sim ticks). Aim
    // delta is applied once at the start of the window; movement/held inputs
    // are applied every tick. See docs/action_spec.md "Per-decision vs
    // per-tick."
    void step_decision(std::array<Action, kAgentsPerMatch> actions);

    // Read-only accessors.
    const MatchState& state() const noexcept { return state_; }
    const MatchConfig& config() const noexcept { return config_; }
    bool episode_over() const noexcept;

    // Deterministic hash of the match state. Used by the golden-replay tests
    // (docs/determinism_rules.md). Manifest of included fields lives in
    // determinism_rules.md §"state_hash() manifest".
    std::uint64_t state_hash() const noexcept;

   private:
    MatchConfig config_{};
    MatchState state_{};
};

}  // namespace xushi2::sim
