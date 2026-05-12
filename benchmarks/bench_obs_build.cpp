#include <benchmark/benchmark.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/sim.h>

namespace {

using xushi2::common::Action;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;

void prime_state(xushi2::sim::Sim& sim, int warmup_ticks) {
    std::array<Action, kAgentsPerMatch> acts{};
    acts[0].move_y = 1.0F;
    acts[3].move_y = -1.0F;
    for (int i = 0; i < warmup_ticks; ++i) {
        sim.step(acts);
    }
}

void bench_obs_build(benchmark::State& state) {
    const int warmup_ticks = static_cast<int>(state.range(0));
    const int builds_per_iter = static_cast<int>(state.range(1));

    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 7;
    cfg.round_length_seconds = 180;

    xushi2::sim::Sim sim(cfg);
    prime_state(sim, warmup_ticks);

    std::array<float, kActorObsPhase1Dim> obs{};
    for (auto _ : state) {
        for (int i = 0; i < builds_per_iter; ++i) {
            xushi2::sim::build_actor_obs_phase1(sim, 0U, obs.data(),
                                                static_cast<std::uint32_t>(obs.size()));
            benchmark::DoNotOptimize(obs);
        }
    }

    state.SetItemsProcessed(state.iterations() * builds_per_iter);
    state.SetLabel("warmup_ticks=" + std::to_string(warmup_ticks) +
                   ",obs_builds=" + std::to_string(builds_per_iter));
}

BENCHMARK(bench_obs_build)
    ->Args({0, 128})
    ->Args({300, 512})
    ->Args({900, 2048})
    ->Complexity();

}  // namespace
