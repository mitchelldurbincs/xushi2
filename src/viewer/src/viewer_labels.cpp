#include "viewer_labels.hpp"

namespace viewer_labels {

const char* target_slot_label(std::uint8_t target_slot, TargetSlotLabelMode mode) {
    // target_slot semantics: 0=self, 1..3=enemies, 4=objective.
    if (mode == TargetSlotLabelMode::Compact) {
        switch (target_slot) {
            case 1: return "E1";
            case 2: return "E2";
            case 3: return "E3";
            default: return "-";
        }
    }

    switch (target_slot) {
        case 0: return "self";
        case 1: return "enemy0";
        case 2: return "enemy1";
        case 3: return "enemy2";
        case 4: return "objective";
        default: return "?";
    }
}

}  // namespace viewer_labels
