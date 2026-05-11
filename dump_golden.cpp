// Temporary dumper used to regenerate data/replays/golden_phase0_basic.txt.
// Mirrors the configuration in tests/replay/test_golden_replay.cpp.
#include <cstdint>
#include <cstdio>
#include <xushi2/bots/runner.h>
#include <xushi2/sim/sim.h>

namespace x = xushi2;

int main() {
    x::sim::MatchConfig cfg{};
    cfg.mechanics.revolver_damage_centi_hp = 7500U;
    cfg.mechanics.revolver_fire_cooldown_ticks = 15U;
    cfg.mechanics.revolver_hitbox_radius = 0.75F;
    cfg.mechanics.respawn_ticks = 240U;
    cfg.seed = 0xD1CEDA7AULL;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = false;
    cfg.randomize_map = false;
    const auto r = x::bots::run_scripted_episode(cfg, "basic", "basic");
    for (auto h : r.decision_hashes) {
        std::printf("%016lx\n", static_cast<unsigned long>(h));
    }
    return 0;
}
