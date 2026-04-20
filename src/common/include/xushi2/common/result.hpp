#pragma once

// Result<T> / ErrorCode — the Tier 0 error-handling primitive.
// Tier 0 public / boundary functions return Result<T>, never throw.
// See docs/coding_philosophy.md §8.

#include <cstdint>
#include <type_traits>
#include <utility>

namespace xushi2::common {

enum class ErrorCode : std::uint16_t {
    Ok = 0,
    InvalidAction,
    InvalidHeroId,
    CorruptState,
    CorruptHeroHp,
    CorruptCooldown,
    NonFiniteFloat,
    NonFinitePosition,
    BadAction,
    CapacityExceeded,
    ReplayBadMagic,
    ReplayVersionMismatch,
    ReplayHashMismatch,
    ObservationLeakDetected,
    Unreachable,
};

template <typename T>
struct Result {
    bool ok = false;
    T value{};
    ErrorCode error = ErrorCode::Ok;

    static Result<T> success(T v) {
        Result<T> r;
        r.ok = true;
        r.value = std::move(v);
        r.error = ErrorCode::Ok;
        return r;
    }

    static Result<T> failure(ErrorCode e) {
        Result<T> r;
        r.ok = false;
        r.error = e;
        return r;
    }
};

// Specialization for void-like return.
template <>
struct Result<void> {
    bool ok = true;
    ErrorCode error = ErrorCode::Ok;

    static Result<void> success() { return Result<void>{true, ErrorCode::Ok}; }
    static Result<void> failure(ErrorCode e) { return Result<void>{false, e}; }
};

}  // namespace xushi2::common
