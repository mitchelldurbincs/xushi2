#pragma once

// Xushi2 simulation core. Deterministic, headless, pure game-state update.
// See docs/game_design.md for rules and docs/determinism_rules.md for the
// float-determinism discipline.

#include <array>
#include <cstdint>
#include <random>
#include <vector>

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
inline constexpr int kTickHz = 30;
inline constexpr float kDt = 1.0F / static_cast<float>(kTickHz);

inline constexpr int kTeamSize = 3;
inline constexpr int kAgentsPerMatch = 2 * kTeamSize;

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
};

// Control-point state machine (see game-design.md §3).
struct ObjectiveState {
    Team owner = Team::Neutral;
    Team cap_team = Team::Neutral;  // Neutral = "None" here
    float cap_progress = 0.0F;
    int team_a_score = 0;
    int team_b_score = 0;
    bool unlocked = false;          // true after the 15s lock window
};

// Match configuration — seed, match length, fog-of-war toggle, etc.
struct MatchConfig {
    std::uint64_t seed = 0;
    int round_length_seconds = 180;
    bool fog_of_war_enabled = true;
    bool randomize_map = false;     // per-episode wall randomization (default off until Phase 8)
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
    // `actions` length must equal kAgentsPerMatch.
    void step(std::array<Action, kAgentsPerMatch> actions);

    // Read-only accessors.
    const MatchState& state() const noexcept { return state_; }
    const MatchConfig& config() const noexcept { return config_; }
    bool episode_over() const noexcept;

    // Deterministic hash of the match state. Used by the golden-replay tests
    // (docs/determinism_rules.md).
    std::uint64_t state_hash() const noexcept;

   private:
    MatchConfig config_{};
    MatchState state_{};
};

}  // namespace xushi2::sim
