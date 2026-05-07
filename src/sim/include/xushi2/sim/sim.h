#pragma once

// Xushi2 simulation core. Deterministic, headless, pure game-state update.
// See docs/game_design.md for rules, docs/determinism_rules.md for the
// float-determinism discipline, docs/action_spec.md for the action contract,
// and docs/coding_philosophy.md for the Tier 0 rules this file obeys.

#include <array>
#include <cstdint>
#include <limits>
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

// Phase-0/1 playable slice: a tiny rectangular arena with fixed bounds.
// The real map comes online at Phase 5 (game_design.md §5).
struct MapBounds {
    float min_x = 0.0F;
    float min_y = 0.0F;
    float max_x = 50.0F;
    float max_y = 50.0F;
};

// Phase-1 Ranger weapon sub-state. Narrow update functions in sim.cpp are
// the only way this advances. See docs/game_design.md §6 "Reload behavior."
struct RangerWeaponState {
    std::uint8_t magazine = 0;          // 0..kRangerMaxMagazine
    bool reloading = false;
    Tick reload_ticks_left = 0;         // valid only while reloading
    Tick ticks_since_last_shot = 0;     // drives auto-reload trigger
    Tick fire_cooldown_ticks = 0;       // min ticks until next shot permitted
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
    std::int32_t health_centi_hp = 0;       // HP × 100; 0 means dead
    std::int32_t max_health_centi_hp = 0;   // HP × 100
    bool alive = true;
    Tick respawn_tick = 0;              // if !alive, tick when respawn fires
    // Cooldowns (ticks remaining, 0 means ready).
    Tick cd_ability_1 = 0;
    Tick cd_ability_2 = 0;
    // Ranger weapon state (magazine, reload, fire-rate gate).
    RangerWeaponState weapon{};
    // Hero-specific state; not all fields apply to all heroes.
    bool vanguard_barrier_active = false;
    std::int32_t vanguard_barrier_hp_centi = 0;
    common::MenderWeapon mender_weapon = common::MenderWeapon::Staff;
    EntityId mender_beam_locked_on = 0;  // 0 = not locked
    // Lifetime combat counters (game-design §13 behavioral metrics use these).
    std::uint32_t kills = 0;
    std::uint32_t deaths = 0;
    // Whether this slot is actually occupied in the Phase-0/1 playable slice.
    // At Phase 4+, all six slots are occupied.
    bool present = false;
};

// Control-point state machine (see game-design.md §3). Integer tick math.
struct ObjectiveState {
    Team owner = Team::Neutral;
    Team cap_team = Team::Neutral;              // Neutral == "None"
    std::uint32_t cap_progress_ticks = 0;       // 0..kCaptureTicks
    std::uint32_t team_a_score_ticks = 0;       // 0..kWinTicks
    std::uint32_t team_b_score_ticks = 0;
    bool unlocked = false;                      // true after the 15s lock window
};

// Phase-1 mechanic values not pinned by docs — must be supplied by the
// caller. The Sim constructor rejects a MatchConfig whose mechanics
// fields are still sentinels. See docs/coding_philosophy.md §3.
struct Phase1MechanicsConfig {
    // Integer damage in centi-HP (damage × 100). UINT32_MAX = unset.
    std::uint32_t revolver_damage_centi_hp = std::numeric_limits<std::uint32_t>::max();
    // Minimum sim ticks between consecutive Revolver shots. 0 = invalid.
    std::uint32_t revolver_fire_cooldown_ticks = std::numeric_limits<std::uint32_t>::max();
    // Circular hitbox radius (u). NaN = unset. Must be > 0.
    float         revolver_hitbox_radius      = std::numeric_limits<float>::quiet_NaN();
    // Ticks between death and respawn. UINT32_MAX = unset. Must be > 0.
    std::uint32_t respawn_ticks               = std::numeric_limits<std::uint32_t>::max();
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
    // Required mechanic values; see Phase1MechanicsConfig docs.
    Phase1MechanicsConfig mechanics{};
    // Phase-4 toggle. team_size==1: single Ranger per team (Phase 1–3 path,
    // slots 0 and 3). team_size==3: full 3v3 (Phase 4+, slots 0–2 + 3–5).
    // Other values are rejected by the Sim ctor.
    std::uint32_t team_size = 1;
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

    // Winner of the (possibly terminal) match: Team::A, Team::B, or
    // Team::Neutral for a draw / in-progress. Meaningful only once
    // episode_over() is true.
    Team winner() const noexcept;

    // Aggregate kill counters across the team (lifetime of this episode).
    std::uint32_t team_a_kills() const noexcept;
    std::uint32_t team_b_kills() const noexcept;

    // Deterministic hash of the match state. Used by the golden-replay tests
    // (docs/determinism_rules.md). Manifest of included fields lives in
    // determinism_rules.md §"state_hash() manifest".
    std::uint64_t state_hash() const noexcept;

   private:
    MatchConfig config_{};
    MatchState state_{};
};

}  // namespace xushi2::sim
