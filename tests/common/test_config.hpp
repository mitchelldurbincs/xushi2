#pragma once

// Test-only helper: build a MatchConfig with known-good Phase 1a mechanics
// values. Every test that wants to step the sim must set mechanics
// explicitly (the sim hard-fails otherwise per coding_philosophy.md §3).
// Centralizing the values here avoids duplication; change them in one place.

#include <xushi2/sim/sim.h>

namespace xushi2::test_support {

inline sim::Phase1MechanicsConfig default_mechanics() {
    sim::Phase1MechanicsConfig m{};
    m.revolver_damage_centi_hp = 7500U;     // 75.0 HP per shot
    m.revolver_fire_cooldown_ticks = 15U;   // 2 shots/sec at 30 Hz
    m.revolver_hitbox_radius = 0.75F;
    m.respawn_ticks = 240U;                 // 8 s at 30 Hz
    return m;
}

inline sim::MatchConfig make_test_config() {
    sim::MatchConfig cfg{};
    cfg.mechanics = default_mechanics();
    return cfg;
}

}  // namespace xushi2::test_support
