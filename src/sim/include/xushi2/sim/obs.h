#pragma once

// Phase-1 flat observation layout — actor and critic.
//
// This header is the C++ mirror of python/xushi2/obs_manifest.py. The two
// files MUST stay in lockstep. Changing field order, field widths, or
// dimensions here without updating the Python manifest (or vice versa) is a
// breaking change that invalidates any trained checkpoints.
//
// See docs/observation_spec.md §Phase 1 for the field catalog and
// docs/rl_design.md §§2–3 for the actor/critic separation contract.

#include <cstdint>

#include <xushi2/common/types.h>

namespace xushi2::sim {

// Forward declarations. Builders take a `const Sim&` to reach both the
// current `MatchState` (for runtime fields) and the `MatchConfig` (for
// normalization bounds like round length and respawn window). Neither
// `MatchState` nor `MatchConfig` is exposed to Python — only these builders
// are callable across the FFI.
class Sim;

// Phase-1 actor obs tensor width. Mirrors ACTOR_PHASE1_DIM in
// python/xushi2/obs_manifest.py. See docs/observation_spec.md §Phase 1.
inline constexpr std::uint32_t kActorObsPhase1Dim = 31;

// Phase-4 critic obs tensor width. Mirrors CRITIC_DIM in
// python/xushi2/obs_manifest.py. Layout: 3 own-team actor mirrors
// (3 × kActorObsPhase1Dim = 93 floats) + 3 enemy world blocks
// (12 floats each = 36) + 4 raw objective counters + 2 seed bits = 135.
inline constexpr std::uint32_t kCriticObsDim = 135;

// Build the Phase-1 actor observation for the agent in `agent_slot` into the
// caller-provided `out_buffer` of capacity `out_capacity` float32 entries.
//
// Contract:
//  - Writes exactly `kActorObsPhase1Dim` floats starting at out_buffer[0].
//  - Requires `out_capacity >= kActorObsPhase1Dim`; aborts otherwise.
//  - Requires `agent_slot` to name a present, Phase-1 Ranger hero.
//  - Must not depend on, and must not leak, any field not listed in
//    observation_spec.md §Phase 1. The implementation routes all enemy
//    lookups through `obs_utils::visible_enemy_1v1` so that when fog of war
//    lands at Phase 7 the actor never sees hidden enemies.
//
// All spatial / velocity / aim fields are in a team-relative frame: Team A
// sees the world as-is, Team B sees the map mirrored across map center.
void build_actor_obs_phase1(const Sim& sim,
                            std::uint32_t agent_slot,
                            float* out_buffer,
                            std::uint32_t out_capacity) noexcept;

// Build the Phase-4 centralized-critic observation for the given team
// perspective into the caller-provided `out_buffer` of capacity
// `out_capacity` float32 entries.
//
// Contract:
//  - Writes exactly `kCriticObsDim` floats starting at out_buffer[0].
//  - Requires `out_capacity >= kCriticObsDim`; aborts otherwise.
//  - Requires the `Sim` to have been constructed with
//    `MatchConfig::team_size == 3`. With team_size == 1 the builder
//    will assert because the team has fewer than 3 present Rangers.
//  - `team_perspective` must be `Team::A` or `Team::B`.
//  - Layout: for the team's three Ranger slots in ascending order,
//    emit the 31-float actor obs of each (team-frame). Then for the
//    enemy team's three slots, emit a 12-float world-frame block
//    (position, velocity, aim_unit, hp, alive, respawn_timer, ammo,
//    reloading, combat_roll_cd). Then 4 raw objective tick counters
//    and 2 normalized seed bits. See python/xushi2/obs_manifest.py
//    CRITIC_FIELDS for the field-by-field catalog.
//  - The critic may iterate full sim state. It MUST NOT be called
//    from any code path that builds an actor obs.
void build_critic_obs(const Sim& sim,
                      common::Team team_perspective,
                      float* out_buffer,
                      std::uint32_t out_capacity) noexcept;

}  // namespace xushi2::sim
