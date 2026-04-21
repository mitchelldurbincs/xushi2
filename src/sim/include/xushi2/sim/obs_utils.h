#pragma once

// Low-level observation-building utilities. Intentionally primitive: these
// functions take `Vec2`, `float`, and enums — never `MatchState` — so they
// are provably incapable of iterating hidden enemy state. Both the actor
// and critic obs builders are allowed to call anything in this header.
// Functions that MUST see full state (critic-only work) live elsewhere.
//
// See docs/rl_design.md §§2–3 and docs/observation_spec.md invariant 1.

#include <cstdint>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace xushi2::sim::obs_utils {

// 180° rotation around map center. Applied for Team B viewers so the policy
// always sees its own side as "bottom." Idempotent: applying it twice yields
// the input (modulo float rounding). See docs/rl_design.md §6
// "Team-relative coordinate normalization."
common::Vec2 mirror_position_for_team(common::Vec2 world_pos,
                                      common::Team viewer_team,
                                      const MapBounds& map) noexcept;

// Mirror a world velocity for the team-relative frame. A 180° rotation of
// position implies a 180° rotation of velocity (both components flip sign).
common::Vec2 mirror_velocity_for_team(common::Vec2 world_vel,
                                      common::Team viewer_team) noexcept;

// Mirror an aim angle (radians) for a team-relative frame. For a 180°
// rotation this is `angle + π` wrapped into [-π, π].
float mirror_angle_for_team(float world_angle_radians,
                            common::Team viewer_team) noexcept;

// Normalize `world_pos` to `[-1, 1]` by mapping map bounds linearly.
// Caller should first mirror to the viewer's team frame.
common::Vec2 normalize_position_to_map(common::Vec2 pos,
                                       const MapBounds& map) noexcept;

// Linear normalization to [-1, 1]. Clamp-free: callers are expected to
// stay within bounds, so this is a pure affine transform.
float normalize_bounded(float value, float lo, float hi) noexcept;

// Wrap an angle to [-π, π].
float wrap_angle_pi(float angle_radians) noexcept;

// Write a 2-element (sin, cos) unit vector for the given angle into
// `out_sincos2`. Guaranteed unit-length up to float precision.
void angle_to_unit(float angle_radians, float* out_sincos2) noexcept;

// A narrow, leak-safe view of the single enemy slot relevant to a 1v1
// Ranger match. At Phase 1 there is no fog, so an alive enemy is always
// visible. When fog lands at Phase 7 this is the single location that
// gains LoS filtering. The actor obs builder is REQUIRED to route all
// enemy lookups through this function — it must never iterate
// `state.heroes` for enemy data itself.
struct VisibleEnemySlot {
    bool present;                          // false if the opposite team has no occupied slot
    common::EntityId id;
    bool alive;
    common::Tick respawn_tick;
    common::Vec2 world_position;
    common::Vec2 velocity;
    std::int32_t health_centi_hp;
    std::int32_t max_health_centi_hp;
};

// Return the single opposite-team occupied slot, or a zeroed struct with
// `present = false` if none exists. Intended for 1v1 Ranger at Phase 1;
// callers for 2v2+ (Phase 4+) must use a different helper.
VisibleEnemySlot visible_enemy_1v1(const MatchState& s,
                                   std::uint32_t viewer_slot) noexcept;

// Geometry helper exposed because both actor and critic need it. Wraps the
// private `inside_objective` used inside the sim tick pipeline.
bool position_on_objective(common::Vec2 world_pos,
                           const MapBounds& map) noexcept;

// Phase-1 convention for velocity normalization: divide per-component by
// the hero's max move speed, then mirror/clamp as needed by caller.
float ranger_max_speed() noexcept;

}  // namespace xushi2::sim::obs_utils
