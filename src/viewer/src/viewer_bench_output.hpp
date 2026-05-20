#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace viewer_bench_output {

struct BenchJsonPayload {
    std::string replay_name;
    std::string mode;
    int warmup_frames = 0;
    int measured_frames = 0;
    std::vector<double> frame_ms;
};

std::optional<std::string> resolve_git_commit();
double percentile_ms(std::vector<double> samples, double p);
std::string json_escape(std::string_view input);

bool write_bench_json(const std::string& path,
                      const BenchJsonPayload& payload,
                      std::string* error_message);

}  // namespace viewer_bench_output
