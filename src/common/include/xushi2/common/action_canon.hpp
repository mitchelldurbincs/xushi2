#pragma once

// Action canonicalization — the Tier 0 boundary between policy/viewer/replay
// and the sim core. See docs/action_spec.md and docs/replay_format.md.
//
// Canonical form:
//   move_x, move_y, aim_delta : int16 quantized (scale 1/10000)
//   primary_fire, ability_1, ability_2 : bitfield in one byte
//   target_slot : uint8
//
// The sim consumes the canonical Action; training, human play, and replay
// all go through this single function. The live-sim action stream and the
// stored replay action stream are bit-identical.

#include <cmath>
#include <cstdint>

#include "math.hpp"
#include "types.h"

namespace xushi2::common {

inline constexpr float kAimDeltaMax = kPiOver4;  // ±45° per policy decision
inline constexpr float kCanonScale = 10000.0F;
inline constexpr std::int16_t kCanonInt16Max = 10000;  // == scale * 1.0
inline constexpr std::int16_t kCanonAimMaxQ = static_cast<std::int16_t>(
    // ±7854 ≈ round(π/4 · 10000)
    7854);

// Quantize a float in [-max, max] to int16 with scale 1/10000.
inline std::int16_t quantize_action_float(float v, float max_abs) {
    const float clamped = clampf(v, -max_abs, max_abs);
    const float scaled = clamped * kCanonScale;
    // Deterministic round to nearest, ties-to-even.
    const float rounded = std::nearbyint(scaled);
    return static_cast<std::int16_t>(rounded);
}

inline float dequantize_action_float(std::int16_t q) {
    return static_cast<float>(q) / kCanonScale;
}

// Canonicalize an Action in-place. Idempotent: canonicalize(canonicalize(a))
// is bitwise equal to canonicalize(a). target_slot is clamped to 0 while the
// action is disabled (Phase 1–9); callers that enable it in Phase 10+ must
// pass a valid value to the check below.
inline void canonicalize_action(Action& a) {
    const std::int16_t mx_q = quantize_action_float(a.move_x, 1.0F);
    const std::int16_t my_q = quantize_action_float(a.move_y, 1.0F);
    const std::int16_t ad_q = quantize_action_float(a.aim_delta, kAimDeltaMax);
    a.move_x = dequantize_action_float(mx_q);
    a.move_y = dequantize_action_float(my_q);
    a.aim_delta = dequantize_action_float(ad_q);
    // Booleans normalize to exactly 0 or 1 already by the type.
    // target_slot pass-through for now; Phase-gate enforcement happens in sim.
}

}  // namespace xushi2::common
