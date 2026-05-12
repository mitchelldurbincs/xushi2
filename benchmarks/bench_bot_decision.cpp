#include <benchmark/benchmark.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/bots/bot.h>
#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;

void prime_sim(xushi2::sim::Sim& sim, int warmup_ticks) {
    std::array<Action, kAgentsPerMatch> actions{};
    for (int i = 0; i < warmup_ticks; ++i) {
        sim.step(actions);
    }
}

void bench_bot_decision(benchmark::State& state) {
    const int warmup_ticks = static_cast<int>(state.range(0));
    const int decisions_per_iter = static_cast<int>(state.range(1));

    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 17;
    cfg.round_length_seconds = 240;

    xushi2::sim::Sim sim(cfg);
    prime_sim(sim, warmup_ticks);

    auto basic = xushi2::bots::make_basic_bot();
    auto hold = xushi2::bots::make_hold_and_shoot_bot();

    for (auto _ : state) {
        for (int i = 0; i < decisions_per_iter; ++i) {
            const auto& s = sim.state();
            auto a0 = basic->decide(s, 0);
            auto a3 = hold->decide(s, 3);
            benchmark::DoNotOptimize(a0);
            benchmark::DoNotOptimize(a3);
        }
    }

    state.SetItemsProcessed(state.iterations() * decisions_per_iter * 2);
    state.SetLabel("warmup_ticks=" + std::to_string(warmup_ticks) +
                   ",bot_decisions=" + std::to_string(decisions_per_iter * 2));
}

BENCHMARK(bench_bot_decision)
    ->Args({0, 512})
    ->Args({300, 2048})
    ->Args({900, 4096})
    ->Complexity();

}  // namespace
