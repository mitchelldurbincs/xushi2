#pragma once

#include <string>
#include <vector>

struct BenchmarkState {
    std::vector<double> measured_ms{};
    int bench_frame = 0;
};

void init_benchmark_state(BenchmarkState& state, int measured_frames);
void record_benchmark_frame(BenchmarkState& state, int warmup_frames, int measured_frames, double frame_ms);
bool benchmark_complete(const BenchmarkState& state, int measured_frames);
void write_bench_json(const std::string& path,
                      const std::string& replay_name,
                      int warmup,
                      int measured,
                      const std::vector<double>& frame_ms);
