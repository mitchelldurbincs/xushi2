#include "benchmark_writer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>

namespace {

double percentile_ms(std::vector<double> samples, double p) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const double idx = (p / 100.0) * static_cast<double>(samples.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(idx));
    const auto hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) return samples[lo];
    const double w = idx - static_cast<double>(lo);
    return samples[lo] * (1.0 - w) + samples[hi] * w;
}

}  // namespace

void init_benchmark_state(BenchmarkState& state, int measured_frames) {
    state.measured_ms.clear();
    state.measured_ms.reserve(static_cast<std::size_t>(measured_frames));
    state.bench_frame = 0;
}

void record_benchmark_frame(BenchmarkState& state,
                            int warmup_frames,
                            int measured_frames,
                            double frame_ms) {
    if (state.bench_frame >= warmup_frames && static_cast<int>(state.measured_ms.size()) < measured_frames) {
        state.measured_ms.push_back(frame_ms);
    }
    ++state.bench_frame;
}

bool benchmark_complete(const BenchmarkState& state, int measured_frames) {
    return static_cast<int>(state.measured_ms.size()) >= measured_frames;
}

void write_bench_json(const std::string& path,
                      const std::string& replay_name,
                      int warmup,
                      int measured,
                      const std::vector<double>& frame_ms) {
    const double sum = std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0);
    const double avg = frame_ms.empty() ? 0.0 : sum / static_cast<double>(frame_ms.size());
    const double fps = avg > 0.0 ? 1000.0 / avg : 0.0;
    const char* git_sha = std::getenv("GIT_COMMIT");
    if (git_sha == nullptr) git_sha = std::getenv("GITHUB_SHA");

    std::ofstream os(path);
    os << "{\n";
    if (git_sha != nullptr) {
        os << "  \"git_commit\": \"" << git_sha << "\",\n";
    } else {
        os << "  \"git_commit\": null,\n";
    }
    os << "  \"replay_name\": \"" << replay_name << "\",\n";
    os << "  \"warmup_frames\": " << warmup << ",\n";
    os << "  \"measured_frames\": " << measured << ",\n";
    os << "  \"avg_ms\": " << avg << ",\n";
    os << "  \"p50_ms\": " << percentile_ms(frame_ms, 50.0) << ",\n";
    os << "  \"p95_ms\": " << percentile_ms(frame_ms, 95.0) << ",\n";
    os << "  \"p99_ms\": " << percentile_ms(frame_ms, 99.0) << ",\n";
    os << "  \"fps\": " << fps << "\n";
    os << "}\n";
}
