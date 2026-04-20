#pragma once

#include <array>
#include <cstdint>

namespace xushi2::common {

// Stable integer IDs (see docs/determinism_rules.md).
using EntityId = std::uint32_t;
using Tick = std::uint32_t;

// Team enumeration. Underlying type fixed for deterministic serialization.
enum class Team : std::uint8_t { Neutral = 0, A = 1, B = 2 };

// Roles per game-design §6.
enum class Role : std::uint8_t { Tank = 0, Damage = 1, Support = 2 };

// Phase 1 hero roster (game-design §6).
enum class HeroKind : std::uint8_t { Vanguard = 0, Ranger = 1, Mender = 2 };

// Mender weapon state.
enum class MenderWeapon : std::uint8_t { Staff = 0, Sidearm = 1 };

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;
};

// Fixed-size action struct emitted by both RL agents and the human viewer.
// See docs/action_spec.md.
struct Action {
    float move_x = 0.0F;                // in [-1, 1]
    float move_y = 0.0F;                // in [-1, 1]
    float aim_delta = 0.0F;             // in [-π/4, π/4]; angular delta per decision
    bool primary_fire = false;
    bool ability_1 = false;
    bool ability_2 = false;
    std::uint8_t target_slot = 0;       // deferred until Phase 10; unused in Phase 1
};

}  // namespace xushi2::common
