#pragma once

#include <cstdint>

#include <xushi2/sim/sim.h>

namespace xushi2::sim::internal {

// Deterministic FNV-1a 64 hash of the full match state. Manifest of included
// fields lives in docs/determinism_rules.md §"state_hash() manifest".
std::uint64_t compute_state_hash(const MatchState& state);

}  // namespace xushi2::sim::internal
