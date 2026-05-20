#include "viewer_bench_output.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>

namespace viewer_bench_output {

std::optional<std::string> resolve_git_commit() {
    const char* git_commit = std::getenv("GIT_COMMIT");
    if (git_commit != nullptr && git_commit[0] != '\0') {
        return std::string(git_commit);
    }
    const char* github_sha = std::getenv("GITHUB_SHA");
    if (github_sha != nullptr && github_sha[0] != '\0') {
        return std::string(github_sha);
    }
    return std::nullopt;
}

double percentile_ms(std::vector<double> samples, const double p) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    const double idx = (p / 100.0) * static_cast<double>(samples.size() - 1U);
    const auto lo = static_cast<std::size_t>(std::floor(idx));
    const auto hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) {
        return samples[lo];
    }
    const double w = idx - static_cast<double>(lo);
    return samples[lo] * (1.0 - w) + samples[hi] * w;
}

std::string json_escape(const std::string_view input) {
    std::string out;
    out.reserve(input.size() + 8U);
    for (const char c : input) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
                break;
        }
    }
    return out;
}

bool write_bench_json(const std::string& path,
                      const BenchJsonPayload& payload,
                      std::string* error_message) {
    const double sum = std::accumulate(payload.frame_ms.begin(), payload.frame_ms.end(), 0.0);
    const double avg = payload.frame_ms.empty() ? 0.0 : sum / static_cast<double>(payload.frame_ms.size());
    const double fps = avg > 0.0 ? 1000.0 / avg : 0.0;

    std::ofstream os(path);
    if (!os.is_open()) {
        if (error_message != nullptr) {
            *error_message = "failed to open output path for write: \"" + json_escape(path) + "\"";
        }
        return false;
    }

    const std::optional<std::string> git_commit = resolve_git_commit();

    os << "{\n";
    if (git_commit.has_value()) {
        os << "  \"git_commit\": \"" << json_escape(*git_commit) << "\",\n";
    } else {
        os << "  \"git_commit\": null,\n";
    }
    os << "  \"replay_name\": \"" << json_escape(payload.replay_name) << "\",\n";
    os << "  \"mode\": \"" << json_escape(payload.mode) << "\",\n";
    os << "  \"warmup_frames\": " << payload.warmup_frames << ",\n";
    os << "  \"measured_frames\": " << payload.measured_frames << ",\n";
    os << "  \"avg_ms\": " << avg << ",\n";
    os << "  \"p50_ms\": " << percentile_ms(payload.frame_ms, 50.0) << ",\n";
    os << "  \"p95_ms\": " << percentile_ms(payload.frame_ms, 95.0) << ",\n";
    os << "  \"p99_ms\": " << percentile_ms(payload.frame_ms, 99.0) << ",\n";
    os << "  \"fps\": " << fps << "\n";
    os << "}\n";

    if (!os.good()) {
        if (error_message != nullptr) {
            *error_message = "failed to flush output JSON to path: \"" + json_escape(path) + "\"";
        }
        return false;
    }

    return true;
}

}  // namespace viewer_bench_output
