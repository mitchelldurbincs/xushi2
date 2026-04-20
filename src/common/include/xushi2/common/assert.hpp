#pragma once

// X2_REQUIRE / X2_ENSURE / X2_INVARIANT / X2_UNREACHABLE.
// See docs/coding_philosophy.md §3. Active in all sim builds (not compiled
// out in release). Internal-only failures abort; boundary functions should
// wrap checks in X2_CHECK_RET which returns a Result<...>::failure.

#include <cstdio>
#include <cstdlib>

#include "result.hpp"

namespace xushi2::common::detail {

[[noreturn]] inline void abort_with(const char* kind, const char* cond, const char* file,
                                    int line) {
    std::fprintf(stderr, "[xushi2] %s failed: %s  (%s:%d)\n", kind, cond, file, line);
    std::abort();
}

}  // namespace xushi2::common::detail

// Preconditions (argument / input checks). Internal code: abort on failure.
#define X2_REQUIRE(cond, /*error_code*/...)                                               \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            ::xushi2::common::detail::abort_with("REQUIRE", #cond, __FILE__, __LINE__);    \
        }                                                                                  \
    } while (0)

// Postconditions (return-value / output checks).
#define X2_ENSURE(cond, /*error_code*/...)                                                \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            ::xushi2::common::detail::abort_with("ENSURE", #cond, __FILE__, __LINE__);     \
        }                                                                                  \
    } while (0)

// State invariants.
#define X2_INVARIANT(cond, /*error_code*/...)                                             \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            ::xushi2::common::detail::abort_with("INVARIANT", #cond, __FILE__, __LINE__);  \
        }                                                                                  \
    } while (0)

// Impossible control path. Always aborts if reached.
#define X2_UNREACHABLE(/*error_code*/...)                                                 \
    do {                                                                                   \
        ::xushi2::common::detail::abort_with("UNREACHABLE", "reached", __FILE__, __LINE__);\
    } while (0)

// Boundary-function variant: on failure, return Result<T>::failure(err).
// Usage:
//   X2_CHECK_RET(ptr != nullptr, ErrorCode::CorruptState);
#define X2_CHECK_RET(cond, err)                                                           \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            return ::xushi2::common::Result<void>::failure(err);                           \
        }                                                                                  \
    } while (0)
