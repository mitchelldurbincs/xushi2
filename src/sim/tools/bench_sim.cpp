#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace x = xushi2;

namespace {

x::sim::MatchConfig make_bench_config(std::uint32_t team_size, std::uint64_t seed) {
    x::sim::MatchConfig cfg{};
    cfg.mechanics.revolver_damage_centi_hp = 7500U;
    cfg.mechanics.revolver_fire_cooldown_ticks = 15U;
    cfg.mechanics.revolver_hitbox_radius = 0.75F;
    cfg.mechanics.respawn_ticks = 240U;
    cfg.seed = seed;
    cfg.round_length_seconds = 180;
    cfg.fog_of_war_enabled = false;
    cfg.randomize_map = false;
    cfg.action_repeat = x::common::kDefaultActionRepeat;
    cfg.team_size = team_size;
    return cfg;
}

std::array<x::common::Action, x::sim::kAgentsPerMatch> make_noop_actions() {
    std::array<x::common::Action, x::sim::kAgentsPerMatch> actions{};
    for (auto& action : actions) {
        action.move_x = 0.0F;
        action.move_y = 0.0F;
        action.aim_delta = 0.0F;
        action.primary_fire = false;
        action.ability_1 = false;
        action.ability_2 = false;
        action.target_slot = 0;
    }
    return actions;
}

struct BenchResult {
    std::uint64_t ticks_executed = 0;
    std::uint64_t decisions_executed = 0;
    double elapsed_seconds = 0.0;
};

BenchResult run_step_bench(const x::sim::MatchConfig& cfg,
                           std::uint32_t warmup_ticks,
                           std::uint32_t measured_ticks,
                           volatile std::uint64_t& sink) {
    x::sim::Sim sim(cfg);
    const auto actions = make_noop_actions();

    for (std::uint32_t i = 0; i < warmup_ticks; ++i) {
        sim.step(actions);
        sink ^= sim.state_hash();
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < measured_ticks; ++i) {
        sim.step(actions);
        sink ^= sim.state_hash();
    }
    const auto end = std::chrono::steady_clock::now();

    BenchResult result{};
    result.ticks_executed = measured_ticks;
    result.decisions_executed = measured_ticks;
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

BenchResult run_step_decision_bench(const x::sim::MatchConfig& cfg,
                                    std::uint32_t warmup_decisions,
                                    std::uint32_t measured_decisions,
                                    volatile std::uint64_t& sink) {
    x::sim::Sim sim(cfg);
    const auto actions = make_noop_actions();

    for (std::uint32_t i = 0; i < warmup_decisions; ++i) {
        sim.step_decision(actions);
        sink ^= sim.state_hash();
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < measured_decisions; ++i) {
        sim.step_decision(actions);
        sink ^= sim.state_hash();
    }
    const auto end = std::chrono::steady_clock::now();

    BenchResult result{};
    result.ticks_executed =
        static_cast<std::uint64_t>(measured_decisions) * static_cast<std::uint64_t>(cfg.action_repeat);
    result.decisions_executed = measured_decisions;
    result.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

void print_result(const char* label, std::uint32_t team_size, const BenchResult& r, bool include_decisions_per_sec) {
    const double ticks_per_sec = (r.elapsed_seconds > 0.0)
                                     ? static_cast<double>(r.ticks_executed) / r.elapsed_seconds
                                     : 0.0;
    const double decisions_per_sec = (r.elapsed_seconds > 0.0)
                                         ? static_cast<double>(r.decisions_executed) / r.elapsed_seconds
                                         : 0.0;

    std::printf("%s team_size=%u ticks=%llu elapsed=%.6f sec ticks/sec=%.2f",
                label,
                static_cast<unsigned>(team_size),
                static_cast<unsigned long long>(r.ticks_executed),
                r.elapsed_seconds,
                ticks_per_sec);
    if (include_decisions_per_sec) {
        std::printf(" decisions=%llu decisions/sec=%.2f",
                    static_cast<unsigned long long>(r.decisions_executed),
                    decisions_per_sec);
    }
    std::printf("\n");
}

}  // namespace

int main() {
    constexpr std::uint32_t kWarmupTicks = 10'000;
    constexpr std::uint32_t kMeasureTicks = 200'000;
    constexpr std::uint32_t kWarmupDecisions = 2'500;
    constexpr std::uint32_t kMeasureDecisions = 50'000;

    volatile std::uint64_t sink = 0;

    for (const std::uint32_t team_size : {1U, 3U}) {
        const auto cfg = make_bench_config(team_size, 0xBEEFA11ULL + team_size);
        const auto step_result = run_step_bench(cfg, kWarmupTicks, kMeasureTicks, sink);
        const auto step_decision_result =
            run_step_decision_bench(cfg, kWarmupDecisions, kMeasureDecisions, sink);

        print_result("bench step", team_size, step_result, false);
        print_result("bench step_decision", team_size, step_decision_result, true);
    }

    std::printf("sink=%llu\n", static_cast<unsigned long long>(sink));
    return 0;
}
