#include <benchmark/benchmark.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;

std::array<Action, kAgentsPerMatch> make_noop_actions() {
    std::array<Action, kAgentsPerMatch> actions{};
    return actions;
}

void bench_sim_tick(benchmark::State& state) {
    const int map_half_extent = static_cast<int>(state.range(0));
    const int ticks_per_iter = static_cast<int>(state.range(1));

    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 42;
    cfg.map.min_x = -static_cast<float>(map_half_extent);
    cfg.map.max_x = static_cast<float>(map_half_extent);
    cfg.map.min_y = -static_cast<float>(map_half_extent);
    cfg.map.max_y = static_cast<float>(map_half_extent);
    cfg.round_length_seconds = 300;

    auto actions = make_noop_actions();
    xushi2::sim::Sim sim(cfg);

    for (auto _ : state) {
        for (int t = 0; t < ticks_per_iter; ++t) {
            sim.step(actions);
        }
        benchmark::DoNotOptimize(sim.state_hash());
    }

    state.SetItemsProcessed(state.iterations() * ticks_per_iter);
    state.SetLabel("map=" + std::to_string(map_half_extent * 2) + "x" +
                   std::to_string(map_half_extent * 2) +
                   ",ticks=" + std::to_string(ticks_per_iter));
}

BENCHMARK(bench_sim_tick)
    ->Args({25, 30})
    ->Args({50, 300})
    ->Args({100, 900})
    ->Complexity();

}  // namespace
