#pragma once

// Fixed-capacity vector. No dynamic allocation after construction.
// See docs/coding_philosophy.md §5.

#include <array>
#include <cstdint>

#include "assert.hpp"

namespace xushi2::common {

template <typename T, std::uint32_t N>
class FixedVector {
   public:
    using size_type = std::uint32_t;

    constexpr FixedVector() = default;

    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type capacity() const noexcept { return N; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    constexpr bool full() const noexcept { return size_ == N; }

    void clear() noexcept { size_ = 0; }

    void push_back(const T& v) {
        X2_REQUIRE(size_ < N, ErrorCode::CapacityExceeded);
        data_[size_] = v;
        ++size_;
    }

    T& operator[](size_type i) {
        X2_REQUIRE(i < size_, ErrorCode::CapacityExceeded);
        return data_[i];
    }
    const T& operator[](size_type i) const {
        X2_REQUIRE(i < size_, ErrorCode::CapacityExceeded);
        return data_[i];
    }

    T* begin() noexcept { return data_.data(); }
    T* end() noexcept { return data_.data() + size_; }
    const T* begin() const noexcept { return data_.data(); }
    const T* end() const noexcept { return data_.data() + size_; }

   private:
    std::array<T, N> data_{};
    size_type size_ = 0;
};

}  // namespace xushi2::common
