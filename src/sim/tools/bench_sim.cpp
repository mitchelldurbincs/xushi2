#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace x = xushi2;

int main() {
    x::sim::MatchConfig cfg{};
    cfg.mechanics.revolver_damage_centi_hp = 7500U;
    cfg.mechanics.revolver_fire_cooldown_ticks = 15U;
    cfg.mechanics.revolver_hitbox_radius = 0.75F;
    cfg.mechanics.respawn_ticks = 240U;
    cfg.seed = 0xB3EC75EEDULL;
    cfg.round_length_seconds = 180;
    cfg.fog_of_war_enabled = true;
    cfg.randomize_map = false;

    x::sim::Sim sim(cfg);

    std::array<x::common::Action, x::sim::kAgentsPerMatch> actions{};

    constexpr std::uint32_t kDecisions = 50'000U;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < kDecisions; ++i) {
        sim.step_decision(actions);
        if (sim.episode_over()) {
            sim.reset(cfg.seed + static_cast<std::uint64_t>(i) + 1ULL);
        }
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const double decisions_per_sec = (elapsed_us > 0)
        ? (static_cast<double>(kDecisions) * 1'000'000.0 / static_cast<double>(elapsed_us))
        : 0.0;

    std::printf("bench_sim: %u decisions in %lld us (%.2f decisions/s)\n",
                kDecisions,
                static_cast<long long>(elapsed_us),
                decisions_per_sec);

    return 0;
}
