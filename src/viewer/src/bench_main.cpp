#include <raylib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <numeric>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

#include "panel.hpp"
#include "render_arena.hpp"
#include "render_debug.hpp"
#include "replay_loader.hpp"
#include "viewer_layout.hpp"
#include "viewer_bench_output.hpp"

namespace {

constexpr int kWindowWidth = viewer_layout::kWindowWidth;
constexpr int kWindowHeight = viewer_layout::kWindowHeight;
constexpr std::uint32_t kActionRepeat = 3U;

struct BenchOptions {
    std::string replay_path{};
    int warmup_frames = 300;
    int measured_frames = 3000;
    std::string mode = "render";
    std::string json_out_path{};
};

void print_usage(const char* exe) {
    std::fprintf(stderr,
                 "Usage: %s --replay <path> [--warmup N] [--frames N] [--mode render|sim] "
                 "[--json-out <path>]\n",
                 exe);
}

std::optional<BenchOptions> parse_cli(int argc, char** argv) {
    BenchOptions opt{};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--replay") == 0 && i + 1 < argc) {
            opt.replay_path = argv[++i];
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opt.warmup_frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            opt.measured_frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            opt.mode = argv[++i];
        } else if (std::strcmp(argv[i], "--json-out") == 0 && i + 1 < argc) {
            opt.json_out_path = argv[++i];
        } else {
            print_usage(argv[0]);
            return std::nullopt;
        }
    }

    if (opt.replay_path.empty() || opt.warmup_frames < 0 || opt.measured_frames <= 0 ||
        (opt.mode != "render" && opt.mode != "sim")) {
        print_usage(argv[0]);
        return std::nullopt;
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    const std::optional<BenchOptions> options = parse_cli(argc, argv);
    if (!options) {
        return 2;
    }

    const std::optional<Replay> replay = load_replay(options->replay_path);
    if (!replay) {
        std::fprintf(stderr, "failed to load replay: %s\n", options->replay_path.c_str());
        return 3;
    }

    xushi2::sim::Sim sim(replay->config);
    std::unique_ptr<xushi2::bots::IBot> bot_a = xushi2::bots::make_basic_bot();
    std::unique_ptr<xushi2::bots::IBot> bot_b = xushi2::bots::make_basic_bot();

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWindowWidth, kWindowHeight, "xushi2 viewer bench");

    const ArenaTransform arena = make_arena_transform(replay->config.map);
    const xushi2::common::Vec2 obj_center{
        0.5F * (replay->config.map.min_x + replay->config.map.max_x),
        0.5F * (replay->config.map.min_y + replay->config.map.max_y),
    };

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch> shots{};
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch> tethers{};
    auto prev_heroes = sim.state().heroes;
    std::size_t replay_idx = 0;

    const auto step_once = [&]() {
        actions = {};
        if (replay_idx < replay->decisions.size()) {
            actions = replay->decisions[replay_idx].actions;
            ++replay_idx;
        } else {
            actions[0] = bot_a->decide(sim.state(), sim.config(), 0);
            actions[3] = bot_b->decide(sim.state(), sim.config(), 3);
        }
        sim.step_decision(actions);
        update_shot_tracers(shots, prev_heroes, sim.state().heroes, sim.state().tick);
        update_tether_trails(tethers, prev_heroes, sim.state().heroes, sim.state().tick);
        prev_heroes = sim.state().heroes;
    };

    const int total_frames = options->warmup_frames + options->measured_frames;
    int measured_count = 0;
    std::vector<double> frame_ms;
    frame_ms.reserve(static_cast<std::size_t>(options->measured_frames));

    for (int frame = 0; frame < total_frames; ++frame) {
        if (sim.episode_over()) {
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();

        step_once();

        if (options->mode == "render") {
            BeginDrawing();
            ClearBackground(Color{8, 10, 14, 255});

            const auto& s = sim.state();
            draw_arena(arena);
            draw_wall_markers(arena, replay->wall_markers);
            draw_cover_markers(arena, replay->cover_markers);
            const LosDebugCounts los_counts = replay->fog ? draw_line_of_sight_debug(arena, sim)
                                                          : LosDebugCounts{};
            draw_objective(arena, s.objective, obj_center);
            draw_tether_trails(arena, tethers, s.tick);
            draw_mender_beams(arena, s.heroes);
            if (replay->target_slot) {
                draw_target_token_debug(arena, s, actions);
            }
            draw_shot_tracers(arena, shots, s.tick);
            for (const auto& h : s.heroes) {
                draw_hero(arena, h);
            }
            const PanelViewModel panel_model{
                .state = s,
                .replay_mode = true,
                .paused = false,
                .playback_speed = 1.0F,
                .replay_phase = replay->phase,
                .replay_fog = replay->fog,
                .replay_fog_mode = replay->fog_mode,
                .replay_layout_hash = replay->layout_hash,
                .replay_match_type = replay->match_type,
                .replay_schedule_summary = replay->schedule_summary,
                .replay_league_summary = replay->league_summary,
                .replay_snapshot_group = replay->snapshot_group,
                .replay_snapshot_name = replay->snapshot_name,
                .replay_loss_mask = replay->loss_mask,
                .replay_target_slot = replay->target_slot,
                .replay_last_seen = replay->last_seen,
                .replay_cover_count = replay->cover_markers.size(),
                .replay_wall_count = replay->wall_markers.size(),
                .replay_los_counts = los_counts,
                .replay_actions = actions,
                .replay_idx = replay_idx,
                .replay_total = replay->decisions.size(),
            };
            draw_panel(panel_model);
            EndDrawing();
        }

        const auto t1 = std::chrono::steady_clock::now();
        if (frame >= options->warmup_frames) {
            const std::chrono::duration<double, std::milli> dt = t1 - t0;
            frame_ms.push_back(dt.count());
            ++measured_count;
        }
    }

    CloseWindow();

    if (measured_count != options->measured_frames) {
        std::fprintf(stderr,
                     "measured frame count mismatch: got=%d expected=%d\n",
                     measured_count,
                     options->measured_frames);
        return 4;
    }

    const double sum = std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0);
    const double avg = sum / static_cast<double>(frame_ms.size());
    const double p50 = viewer_bench_output::percentile_ms(frame_ms, 50.0);
    const double p95 = viewer_bench_output::percentile_ms(frame_ms, 95.0);
    const double p99 = viewer_bench_output::percentile_ms(frame_ms, 99.0);
    const double fps = 1000.0 / avg;

    std::printf("bench_result replay=%s mode=%s warmup=%d frames=%d\n",
                options->replay_path.c_str(),
                options->mode.c_str(),
                options->warmup_frames,
                options->measured_frames);
    std::printf("avg_ms=%.4f p50_ms=%.4f p95_ms=%.4f p99_ms=%.4f fps=%.2f\n",
                avg,
                p50,
                p95,
                p99,
                fps);

    if (!options->json_out_path.empty()) {
        const viewer_bench_output::BenchJsonPayload payload{
            .replay_name = options->replay_path,
            .mode = options->mode,
            .warmup_frames = options->warmup_frames,
            .measured_frames = options->measured_frames,
            .frame_ms = frame_ms,
        };
        std::string err;
        if (!viewer_bench_output::write_bench_json(options->json_out_path, payload, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 5;
        }
    }

    return 0;
}
