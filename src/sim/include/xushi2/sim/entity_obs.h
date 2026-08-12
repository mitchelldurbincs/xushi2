#pragma once

// Native entity-token/grid observation builder — the single owner of
// everything an actor is allowed to see about other entities.
//
// This is the C++ home of what python/xushi2/multi_enemy_obs.py
// `actor_obs_to_multi_enemy_entity_grid_obs` used to assemble in Python.
// Layout (per viewer, kEntityGridObsDim floats):
//   - kEntityTokenCount tokens × kEntityTokenDim floats
//     (self, enemy0, enemy1, enemy2, objective), each token:
//     kind[0:3], team[3:6], hp[6], alive[7], position[8:10],
//     velocity[10:12], aim[12:14], ammo[14], reloading[15],
//     ability_cd[16], aux[17]
//   - kEntityTokenCount mask bits
//   - kEntityGridChannels × kEntityGridSize² occupancy grid
//     (ch0 objective, ch1 self, ch2 enemies; painted 1.0 live / 0.5 stale)
//
// The layout MUST stay in lockstep with python/xushi2/obs_manifest.py and
// python/xushi2/multi_enemy_obs.py. See docs/observation_spec.md.
//
// Separation contract (docs/observation_spec.md invariant 1): the critic
// tensor is never an input here. Enemy data reaches token-writing code only
// through the FoggedEnemyView gate produced by the engine's single fog-gate
// function. entity_obs.cpp is the one translation unit permitted to iterate
// enemy HeroState for actor-destined data, and only inside that gate.

#include <array>
#include <cstdint>

#include <xushi2/common/types.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/sim.h>

namespace xushi2::sim {

inline constexpr std::uint32_t kEntityTokenDim = 18;
inline constexpr std::uint32_t kEntityTokenCount = 5;
inline constexpr std::uint32_t kEntityGridChannels = 3;
inline constexpr std::uint32_t kEntityGridSize = 32;
inline constexpr std::uint32_t kEntityGridFlatDim =
    kEntityGridChannels * kEntityGridSize * kEntityGridSize;
inline constexpr std::uint32_t kEntityGridObsDim =
    (kEntityTokenCount * kEntityTokenDim) + kEntityTokenCount +
    kEntityGridFlatDim;  // 5*18 + 5 + 3*32*32 = 3167

// Leak barrier: the ONLY structure token-writing code may read enemy data
// from. Produced by the engine's single fog gate. Exactly one of
// `visible` / `stale` may be set; when neither is set the enemy is hidden
// and contributes nothing beyond the (config-gated) kind/team markers.
struct FoggedEnemyView {
    bool visible = false;   // live fields below are populated
    bool stale = false;     // last-seen marker form (rel_position only)
    common::Vec2 rel_position{};  // viewer team frame, normalized, minus own pos
    float hp = 0.0F;
    common::Vec2 velocity_norm{};
    float aim_sin = 0.0F;
    float aim_cos = 0.0F;
    float ammo = 0.0F;
    float reloading = 0.0F;
    float ability_cd = 0.0F;
    float on_objective = 0.0F;  // aux: 1.0 live-on-point (0.5 stale form)
};

class ObservationEngine {
   public:
    explicit ObservationEngine(const ObsConfig& cfg) noexcept;

    [[nodiscard]] const ObsConfig& config() const noexcept { return cfg_; }

    // Clear last-seen memory. Must be called whenever the paired Sim resets.
    void reset() noexcept;

    // Compute visibility under the config, fold this tick's sightings into
    // last-seen memory (idempotent within a tick: calling twice on the same
    // sim state yields the same memory and output), and write
    // kEntityGridObsDim floats for one viewer slot.
    void build_entity_obs(const Sim& sim,
                          std::uint32_t viewer_slot,
                          float* out,
                          std::uint32_t capacity) noexcept;

    // All kAgentsPerMatch viewers in ascending slot order; `out` holds
    // kAgentsPerMatch * kEntityGridObsDim floats.
    void build_entity_obs_all(const Sim& sim,
                              float* out,
                              std::uint32_t capacity) noexcept;

    // Read-only visibility of the viewer's three enemies (ascending enemy
    // slot order), with no last-seen memory update. Diagnostics and tests.
    [[nodiscard]] std::array<bool, kTeamSize>
    visible_enemies(const Sim& sim, std::uint32_t viewer_slot) const noexcept;

    // Deterministic hash over last-seen memory only. NOT part of
    // Sim::state_hash() — observation memory is policy state, not game
    // state, and golden replay fixtures must not depend on it.
    [[nodiscard]] std::uint64_t obs_state_hash() const noexcept;

   private:
    struct LastSeen {
        common::Vec2 pos_norm{};  // viewer team frame, normalized
        bool valid = false;
    };

    void update_last_seen(const Sim& sim, std::uint32_t viewer_slot,
                          const std::array<bool, kTeamSize>& visible) noexcept;

    ObsConfig cfg_;
    std::array<std::array<LastSeen, kTeamSize>,
               static_cast<std::size_t>(kAgentsPerMatch)>
        last_seen_{};
};

}  // namespace xushi2::sim
