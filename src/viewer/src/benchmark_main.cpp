#include "replay_loader.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <xushi2/sim/sim.h>

namespace {

constexpr std::array<const char*, 3> kDefaultBenchmarkReplays = {
    "data/benchmarks/viewer/minimal_scene.replay",
    "data/benchmarks/viewer/typical_match_scene.replay",
    "data/benchmarks/viewer/stress_scene.replay",
};

void run_one(const std::string& path) {
    const auto replay = load_replay(path);
    if (!replay) {
        std::fprintf(stderr, "benchmark: failed to load replay %s\n", path.c_str());
        return;
    }

    xushi2::sim::Sim sim(replay->config);
    auto start = std::chrono::steady_clock::now();
    for (const auto& decision : replay->decisions) {
        sim.step_decision(decision.actions);
        if (sim.episode_over()) break;
    }
    auto end = std::chrono::steady_clock::now();
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::printf("replay=%s decisions=%zu elapsed_us=%lld\n", path.c_str(),
                replay->decisions.size(), static_cast<long long>(micros));
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> replays;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--replay") == 0 && i + 1 < argc) {
            replays.emplace_back(argv[++i]);
        }
    }

    if (replays.empty()) {
        replays.assign(kDefaultBenchmarkReplays.begin(), kDefaultBenchmarkReplays.end());
    }

    for (const auto& replay : replays) {
        run_one(replay);
    }
    return 0;
}
