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

// Phase-1 critic obs tensor width. Mirrors CRITIC_PHASE1_DIM in
// python/xushi2/obs_manifest.py. Actor surface plus world-frame positions,
// raw tick counters, and seed bits.
inline constexpr std::uint32_t kCriticObsPhase1Dim = 45;

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

// Build the Phase-1 critic observation for the given team perspective into
// the caller-provided `out_buffer` of capacity `out_capacity` float32
// entries.
//
// Contract:
//  - Writes exactly `kCriticObsPhase1Dim` floats starting at out_buffer[0].
//  - Requires `out_capacity >= kCriticObsPhase1Dim`; aborts otherwise.
//  - `team_perspective` must be `Team::A` or `Team::B`; the first
//    `kActorObsPhase1Dim` entries match the actor obs for that team's Ranger
//    slot, followed by privileged fields (world-frame absolute positions
//    and velocities, raw tick counters, match seed bits).
//  - The critic may iterate full sim state; it is the centralized-training
//    side. It MUST NOT be called from any code path that builds an actor
//    obs.
void build_critic_obs_phase1(const Sim& sim,
                             common::Team team_perspective,
                             float* out_buffer,
                             std::uint32_t out_capacity) noexcept;

}  // namespace xushi2::sim
