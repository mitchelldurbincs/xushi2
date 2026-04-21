#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <test_config.hpp>
#include <xushi2/bots/runner.h>
#include <xushi2/sim/sim.h>

// Golden-replay CI test. Runs a canonical scripted-vs-scripted rollout and
// compares each per-decision state_hash to a committed list at
// data/replays/golden_phase0_basic.txt. If the sim, action canonicalization,
// bot logic, or state_hash manifest changes intentionally, regenerate with:
//     xushi2-eval --dump-golden > data/replays/golden_phase0_basic.txt
// See docs/determinism_rules.md and docs/replay_format.md.

namespace {

#ifndef XUSHI2_SOURCE_DIR
#error "XUSHI2_SOURCE_DIR must be defined via target_compile_definitions"
#endif

constexpr const char* kGoldenPath =
    XUSHI2_SOURCE_DIR "/data/replays/golden_phase0_basic.txt";

std::vector<std::uint64_t> load_golden(const std::string& path) {
    std::vector<std::uint64_t> out;
    std::ifstream in(path);
    if (!in) {
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        out.push_back(std::stoull(line, nullptr, 16));
    }
    return out;
}

}  // namespace

TEST(GoldenReplay, BasicVsBasicMatchesGoldenTrajectory) {
    const auto golden = load_golden(kGoldenPath);
    if (golden.empty()) {
        GTEST_SKIP() << "no golden file at " << kGoldenPath
                     << " — generate with `xushi2-eval --dump-golden`";
    }

    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xD1CEDA7AULL;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = false;
    cfg.randomize_map = false;

    const auto result = xushi2::bots::run_scripted_episode(cfg, "basic", "basic");
    ASSERT_EQ(result.decision_hashes.size(), golden.size())
        << "decision count changed — regenerate golden artifact";
    for (std::size_t i = 0; i < golden.size(); ++i) {
        ASSERT_EQ(result.decision_hashes[i], golden[i])
            << "divergence at decision " << i;
    }
}
