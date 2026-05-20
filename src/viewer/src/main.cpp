#include <raylib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

#include "benchmark_writer.hpp"
#include "panel.hpp"
#include "playback_controller.hpp"
#include "render_arena.hpp"
#include "render_debug.hpp"
#include "replay_loader.hpp"
#include "viewer_layout.hpp"

namespace {
constexpr int kWindowWidth = viewer_layout::kWindowWidth;
constexpr int kWindowHeight = viewer_layout::kWindowHeight;
constexpr int kTargetFps = 60;
constexpr std::uint32_t kActionRepeat = 3U;
constexpr float kDecisionSeconds =
    static_cast<float>(kActionRepeat) / static_cast<float>(xushi2::sim::kTickHz);

struct CliArgs {
    std::optional<std::string> replay_path;
    std::optional<std::string> json_out;
    int warmup_frames = 120;
    int measured_frames = 600;
    bool benchmark = false;
};

xushi2::sim::MatchConfig make_viewer_config() {
    xushi2::sim::MatchConfig c{};
    c.seed = 42;
    c.round_length_seconds = 30;
    c.fog_of_war_enabled = false;
    c.randomize_map = false;
    c.action_repeat = kActionRepeat;
    c.mechanics.revolver_damage_centi_hp = 7500U;
    c.mechanics.revolver_fire_cooldown_ticks = 15U;
    c.mechanics.revolver_hitbox_radius = 0.75F;
    c.mechanics.respawn_ticks = 240U;
    return c;
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args{};
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--replay" && i + 1 < argc) {
            args.replay_path = argv[++i];
        } else if (a == "--json-out" && i + 1 < argc) {
            args.json_out = argv[++i];
            args.benchmark = true;
        } else if (a == "--bench-warmup-frames" && i + 1 < argc) {
            args.warmup_frames = std::max(0, std::atoi(argv[++i]));
            args.benchmark = true;
        } else if (a == "--bench-measured-frames" && i + 1 < argc) {
            args.measured_frames = std::max(1, std::atoi(argv[++i]));
            args.benchmark = true;
        }
    }
    return args;
}

PanelViewModel make_panel_model(
    const xushi2::sim::MatchState& state,
    const std::optional<Replay>& replay,
    const PlaybackState& playback,
    LosDebugCounts los_counts,
    const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions) {
    return PanelViewModel{
        .state = state,
        .replay_mode = replay.has_value(),
        .paused = playback.paused,
        .playback_speed = playback.playback_speed,
        .replay_phase = replay ? replay->phase : 0,
        .replay_fog = replay ? replay->fog : false,
        .replay_fog_mode = replay ? replay->fog_mode : std::string_view{},
        .replay_layout_hash = replay ? replay->layout_hash : std::string_view{},
        .replay_match_type = replay ? replay->match_type : std::string_view{},
        .replay_schedule_summary = replay ? replay->schedule_summary : std::string_view{},
        .replay_league_summary = replay ? replay->league_summary : std::string_view{},
        .replay_snapshot_group = replay ? replay->snapshot_group : std::string_view{},
        .replay_snapshot_name = replay ? replay->snapshot_name : std::string_view{},
        .replay_loss_mask = replay ? replay->loss_mask : std::string_view{},
        .replay_target_slot = replay ? replay->target_slot : false,
        .replay_last_seen = replay ? replay->last_seen : false,
        .replay_cover_count = replay ? replay->cover_markers.size() : 0U,
        .replay_wall_count = replay ? replay->wall_markers.size() : 0U,
        .replay_los_counts = los_counts,
        .replay_actions = actions,
        .replay_idx = playback.replay_idx,
        .replay_total = replay ? replay->decisions.size() : 0U,
    };
}

void draw_frame(const ArenaTransform& arena,
                const xushi2::common::Vec2& obj_center,
                xushi2::sim::Sim& sim,
                const std::optional<Replay>& replay,
                const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions,
                const std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
                const std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& tethers,
                const PlaybackState& playback) {
    BeginDrawing();
    ClearBackground(Color{8, 10, 14, 255});

    const auto& s = sim.state();
    draw_arena(arena);
    if (replay) {
        draw_wall_markers(arena, replay->wall_markers);
        draw_cover_markers(arena, replay->cover_markers);
    }

    const LosDebugCounts los_counts =
        (replay && replay->fog) ? draw_line_of_sight_debug(arena, sim) : LosDebugCounts{};

    draw_objective(arena, s.objective, obj_center);
    draw_tether_trails(arena, tethers, s.tick);
    draw_mender_beams(arena, s.heroes);
    if (replay && replay->target_slot) {
        draw_target_token_debug(arena, s, actions);
    }
    draw_shot_tracers(arena, shots, s.tick);
    for (const auto& h : s.heroes) {
        draw_hero(arena, h);
    }

    const PanelViewModel panel_model = make_panel_model(s, replay, playback, los_counts, actions);
    draw_panel(panel_model);

    EndDrawing();
}
}  // namespace

int main(int argc, char** argv) {
    const CliArgs cli = parse_args(argc, argv);
    std::optional<Replay> replay;
    if (cli.replay_path) replay = load_replay(*cli.replay_path);

    if (cli.benchmark) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWindowWidth, kWindowHeight, "xushi2 viewer");
    SetTargetFPS(kTargetFps);

    const xushi2::sim::MatchConfig config = replay ? replay->config : make_viewer_config();
    xushi2::sim::Sim sim(config);
    std::unique_ptr<xushi2::bots::IBot> bot_a = xushi2::bots::make_basic_bot();
    std::unique_ptr<xushi2::bots::IBot> bot_b = xushi2::bots::make_basic_bot();
    const ArenaTransform arena = make_arena_transform(config.map);
    const xushi2::common::Vec2 obj_center{
        0.5F * (config.map.min_x + config.map.max_x),
        0.5F * (config.map.min_y + config.map.max_y)};
    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch> shots{};
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch> tethers{};
    auto prev_heroes = sim.state().heroes;

    PlaybackState playback{};
    PlaybackContext playback_ctx{.sim = &sim,
                                 .replay = &replay,
                                 .bot_a = &bot_a,
                                 .bot_b = &bot_b,
                                 .actions = &actions,
                                 .shots = &shots,
                                 .tethers = &tethers,
                                 .prev_heroes = &prev_heroes};

    BenchmarkState benchmark{};
    init_benchmark_state(benchmark, cli.measured_frames);
    while (!WindowShouldClose()) {
        const auto frame_start = std::chrono::steady_clock::now();

        handle_input(playback_ctx, playback);
        advance_playback(playback_ctx, playback, GetFrameTime(), kDecisionSeconds);
        draw_frame(arena, obj_center, sim, replay, actions, shots, tethers, playback);
        if (sim.episode_over()) reset_playback(playback_ctx, playback);

        if (cli.benchmark) {
            const auto frame_end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            record_benchmark_frame(benchmark, cli.warmup_frames, cli.measured_frames, ms);
            if (benchmark_complete(benchmark, cli.measured_frames)) break;
        }
    }

    if (cli.benchmark && cli.json_out) {
        const std::string replay_name = cli.replay_path ? *cli.replay_path : std::string("<none>");
        write_bench_json(
            *cli.json_out, replay_name, cli.warmup_frames, cli.measured_frames, benchmark.measured_ms);
    }
    CloseWindow();
    return 0;
}
