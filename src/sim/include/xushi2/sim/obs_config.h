#pragma once

// Observation policy — what a viewer is allowed to see. Deliberately NOT
// part of MatchConfig: observation policy never affects game state,
// Sim::state_hash(), or replay fixtures. It configures ObservationEngine
// (entity_obs.h), the single native owner of actor-visible entity assembly.
//
// The visibility rule enshrined here is the Phase-11 training rule
// (python/envs/phase11_current_selfplay_mappo.py _enemy_visibility_matrix,
// the rule live checkpoints were trained under):
//
//   visible(viewer, enemy) = enemy.alive
//                            AND norm_dist(viewer, enemy) <= visible_radius
//                            AND native LoS (obs_utils::observable_enemy,
//                                            which honors
//                                            MatchConfig::fog_of_war_enabled)
//
// with TeamShared taking the OR of the radius term and the OR of the LoS
// term independently across teammates (a teammate in radius plus a
// different teammate with LoS makes the enemy visible — this matches the
// Python rule exactly and is deliberate).
//
// See docs/rl_design.md §Phase 7 for the team_shared / per_agent split.

#include <cmath>
#include <cstdint>
#include <limits>

namespace xushi2::sim {

enum class FogMode : std::uint8_t {
    // Alive enemies are always visible: no LoS query, no radius term.
    None = 0,
    // Dota/OA5 fog (Phase 7a): visible-enemy sets are unioned across the
    // viewer's team before building each agent's observation.
    TeamShared = 1,
    // Genuine per-agent fog (Phase 7b): each agent sees only what it
    // directly has radius + line-of-sight to.
    PerAgent = 2,
};

struct ObsConfig {
    FogMode fog_mode = FogMode::PerAgent;

    // Radius in NORMALIZED team-frame units: the Euclidean norm of the
    // per-axis map-normalized delta (the Python 0.65 convention). On a
    // non-square map this is anisotropic — a known wart preserved for
    // checkpoint continuity; fixing it to world units is a training-
    // semantics change that must be its own flagged experiment.
    // NaN = unset: no radius term in the visibility rule.
    float visible_radius = std::numeric_limits<float>::quiet_NaN();

    // Maintain per-viewer last-seen enemy markers: when a previously seen
    // enemy is currently hidden, its token carries the stale-marker form
    // (alive = 0, frozen normalized position, aux = 0.5, mask = 1).
    bool last_seen_enabled = false;

    // Phase-4 information-ablation compatibility
    // (python/xushi2/multi_enemy_obs.py zero_masked_enemy_tokens): hidden
    // enemy tokens are fully zeroed, including the kind/team markers that
    // the default path always writes.
    bool zero_hidden_token_markers = false;
};

// NaN means unset — true only when the radius was explicitly set.
[[nodiscard]] inline bool has_visible_radius(const ObsConfig& cfg) noexcept {
    return !std::isnan(cfg.visible_radius);
}

}  // namespace xushi2::sim
