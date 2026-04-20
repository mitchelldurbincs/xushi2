#pragma once

// Tiny deterministic math helpers. No fast-math; no intrinsics.
// See docs/determinism_rules.md.

#include <cmath>
#include <cstdint>

#include "types.h"

namespace xushi2::common {

inline constexpr float kPi = 3.14159265358979323846F;
inline constexpr float kTwoPi = 2.0F * kPi;
inline constexpr float kPiOver4 = kPi / 4.0F;

inline constexpr float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline constexpr Vec2 add(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
inline constexpr Vec2 sub(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }
inline constexpr Vec2 scale(Vec2 a, float s) { return Vec2{a.x * s, a.y * s}; }

inline float length(Vec2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }

// Normalize move-input to [-1, 1]² (per game-design §8: "normalized if |v| > 1").
inline Vec2 normalize_move_input(Vec2 v) {
    const float len2 = v.x * v.x + v.y * v.y;
    if (len2 <= 1.0F) {
        return v;
    }
    const float inv_len = 1.0F / std::sqrt(len2);
    return Vec2{v.x * inv_len, v.y * inv_len};
}

// Wrap angle to (-π, π].
inline float wrap_angle(float a) {
    while (a > kPi) {
        a -= kTwoPi;
    }
    while (a <= -kPi) {
        a += kTwoPi;
    }
    return a;
}

// Quantize position to 1/1000 unit (per docs/determinism_rules.md §7).
inline std::int32_t quantize_pos(float v) {
    // nearbyint with the default rounding mode is deterministic under /fp:precise
    // and -fno-fast-math.
    return static_cast<std::int32_t>(std::nearbyint(v * 1000.0F));
}

// Quantize HP to 1/100 (game-design §10).
inline std::int32_t quantize_hp(float hp) {
    return static_cast<std::int32_t>(std::nearbyint(hp * 100.0F));
}

}  // namespace xushi2::common
