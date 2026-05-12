#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

namespace {

struct BenchCase {
    std::string benchmark_name;
    std::uint32_t team_size = 1;
    std::uint32_t action_repeat = 3;
    std::uint32_t iterations = 5000;
};

enum class OutputMode { Text, Json, Csv };

xushi2::sim::Phase1MechanicsConfig default_mechanics() {
    xushi2::sim::Phase1MechanicsConfig m{};
    m.revolver_damage_centi_hp = 7500U;
    m.revolver_fire_cooldown_ticks = 15U;
    m.revolver_hitbox_radius = 0.75F;
    m.respawn_ticks = 240U;
    return m;
}

struct BenchResult {
    BenchCase spec{};
    double elapsed_seconds = 0.0;
    std::uint64_t ticks = 0;
    double ticks_per_second = 0.0;
};

BenchResult run_case(const BenchCase& spec) {
    xushi2::sim::MatchConfig cfg{};
    cfg.seed = 1337U;
    cfg.mechanics = default_mechanics();
    cfg.team_size = spec.team_size;
    cfg.action_repeat = spec.action_repeat;

    xushi2::sim::Sim sim(cfg);
    std::array<xushi2::common::Action, xushi2::sim::kAgentsPerMatch> actions{};

    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < spec.iterations; ++i) {
        if (sim.episode_over()) {
            sim.reset();
        }
        sim.step_decision(actions);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    const auto total_ticks = static_cast<std::uint64_t>(spec.iterations) *
                             static_cast<std::uint64_t>(spec.action_repeat);

    BenchResult result{};
    result.spec = spec;
    result.elapsed_seconds = elapsed;
    result.ticks = total_ticks;
    result.ticks_per_second =
        (elapsed > 0.0) ? static_cast<double>(total_ticks) / elapsed : 0.0;
    return result;
}

OutputMode parse_output_mode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--json") {
            return OutputMode::Json;
        }
        if (arg == "--csv") {
            return OutputMode::Csv;
        }
    }
    return OutputMode::Text;
}

void print_text(const std::vector<BenchResult>& results) {
    std::cout << "benchmark_name team_size action_repeat iterations ticks elapsed_seconds ticks_per_second\n";
    for (const auto& r : results) {
        std::cout << r.spec.benchmark_name << ' ' << r.spec.team_size << ' '
                  << r.spec.action_repeat << ' ' << r.spec.iterations << ' '
                  << r.ticks << ' ' << std::fixed << std::setprecision(6)
                  << r.elapsed_seconds << ' ' << r.ticks_per_second << '\n';
    }
}

void print_csv(const std::vector<BenchResult>& results) {
    std::cout << "benchmark_name,team_size,action_repeat,iterations,ticks,elapsed_seconds,ticks_per_second\n";
    for (const auto& r : results) {
        std::cout << r.spec.benchmark_name << ',' << r.spec.team_size << ','
                  << r.spec.action_repeat << ',' << r.spec.iterations << ','
                  << r.ticks << ',' << std::fixed << std::setprecision(6)
                  << r.elapsed_seconds << ',' << r.ticks_per_second << '\n';
    }
}

void print_json(const std::vector<BenchResult>& results) {
    std::cout << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        std::cout << "  {\"benchmark_name\":\"" << r.spec.benchmark_name
                  << "\",\"team_size\":" << r.spec.team_size
                  << ",\"action_repeat\":" << r.spec.action_repeat
                  << ",\"iterations\":" << r.spec.iterations
                  << ",\"ticks\":" << r.ticks << ",\"elapsed_seconds\":"
                  << std::fixed << std::setprecision(6) << r.elapsed_seconds
                  << ",\"ticks_per_second\":" << r.ticks_per_second << "}";
        if (i + 1 != results.size()) {
            std::cout << ',';
        }
        std::cout << '\n';
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const auto mode = parse_output_mode(argc, argv);

    const std::vector<BenchCase> cases{
        {"sim_step_decision_team1_ar2", 1U, 2U, 7500U},
        {"sim_step_decision_team1_ar3", 1U, 3U, 5000U},
        {"sim_step_decision_team3_ar2", 3U, 2U, 7500U},
        {"sim_step_decision_team3_ar3", 3U, 3U, 5000U},
    };

    std::vector<BenchResult> results;
    results.reserve(cases.size());
    for (const auto& c : cases) {
        results.push_back(run_case(c));
    }

    switch (mode) {
        case OutputMode::Text:
            print_text(results);
            break;
        case OutputMode::Json:
            print_json(results);
            break;
        case OutputMode::Csv:
            print_csv(results);
            break;
    }

    return EXIT_SUCCESS;
}
