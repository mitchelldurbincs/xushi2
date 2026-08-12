#pragma once

// Reward feature block — everything the Python RewardCalculator reads from
// the sim per decision step, packed into one float32 vector so the batched
// SimPool boundary needs a single write per env instead of ~30 property
// reads plus re-derived actor observations.
//
// The layout MUST stay in lockstep with python/xushi2/obs_manifest.py
// REWARD_FEATURE_FIELDS. Values mirror what the legacy per-property path
// produced:
//   - on_point_by_slot uses the actor obs self_on_point rule
//     (alive && position_on_objective);
//   - dist_to_center_norm_by_slot is the hypot of the team-frame
//     map-normalized own position (mirror-invariant), matching the actor
//     obs own_position that reward.py's distance term consumed;
//   - owner/cap signs are Team-A-signed exactly like
//     ObsAccessor.objective_conversion_state (+1 us/A, -1 them/B, 0 none).
//
// All counters fit exactly in float32 (score ticks <= kWinTicks, episode
// damage sums stay far below 2^24 centi-HP).

#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim {

inline constexpr std::uint32_t kRewardFeatureDim = 48;

// Offsets into the block (widths in comments).
namespace reward_features {
inline constexpr std::uint32_t kTick = 0;
inline constexpr std::uint32_t kTeamAScoreTicks = 1;
inline constexpr std::uint32_t kTeamBScoreTicks = 2;
inline constexpr std::uint32_t kTeamAKills = 3;
inline constexpr std::uint32_t kTeamBKills = 4;
inline constexpr std::uint32_t kOwnerSignA = 5;
inline constexpr std::uint32_t kCapSignA = 6;
inline constexpr std::uint32_t kCapProgressFraction = 7;
inline constexpr std::uint32_t kCapProgressTicks = 8;
inline constexpr std::uint32_t kEpisodeOver = 9;
inline constexpr std::uint32_t kScoreThresholdReached = 10;
inline constexpr std::uint32_t kWinnerSign = 11;
inline constexpr std::uint32_t kKillsBySlot = 12;        // 6
inline constexpr std::uint32_t kDeathsBySlot = 18;       // 6
inline constexpr std::uint32_t kDamageCentiBySlot = 24;  // 6
inline constexpr std::uint32_t kAliveBySlot = 30;        // 6
inline constexpr std::uint32_t kOnPointBySlot = 36;      // 6
inline constexpr std::uint32_t kDistToCenterBySlot = 42; // 6
}  // namespace reward_features

// Write kRewardFeatureDim floats describing the sim's current state.
void write_reward_features(const Sim& sim,
                           float* out,
                           std::uint32_t capacity) noexcept;

}  // namespace xushi2::sim
