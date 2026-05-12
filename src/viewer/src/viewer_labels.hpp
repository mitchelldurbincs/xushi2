#pragma once

#include <cstdint>

namespace viewer_labels {

enum class TargetSlotLabelMode {
    Compact,
    Full,
};

const char* target_slot_label(std::uint8_t target_slot, TargetSlotLabelMode mode);

}  // namespace viewer_labels
