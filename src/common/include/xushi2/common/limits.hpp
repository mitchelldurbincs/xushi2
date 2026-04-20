#pragma once

// Project-wide fixed capacity constants. See docs/coding_philosophy.md §5.
// No dynamic allocation after initialization — all Tier 0 buffers size from
// these.

#include <cstdint>

namespace xushi2::common {

inline constexpr std::uint32_t kTickHz = 30;
inline constexpr std::uint32_t kTeamSize = 3;
inline constexpr std::uint32_t kMaxHeroes = 6;
inline constexpr std::uint32_t kMaxWalls = 128;
inline constexpr std::uint32_t kMaxBarriers = 6;
inline constexpr std::uint32_t kMaxEventsPerTick = 128;
inline constexpr std::uint32_t kMaxRaysPerTick = 32;

// At most 240 s @ 15 Hz policy rate; see docs/replay_format.md.
inline constexpr std::uint32_t kMaxReplayDecisionRecords = 240U * 15U;

// Default action-repeat window (sim ticks per policy decision).
// See docs/action_spec.md. Valid values: 2 or 3.
inline constexpr std::uint32_t kDefaultActionRepeat = 3;

}  // namespace xushi2::common
