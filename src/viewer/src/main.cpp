#include <raylib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

#include "render_arena.hpp"
#include "render_debug.hpp"
#include "panel.hpp"
#include "viewer_layout.hpp"
#include "replay_loader.hpp"

namespace {
constexpr int kWindowWidth = viewer_layout::kWindowWidth;
constexpr int kWindowHeight = viewer_layout::kWindowHeight;
constexpr int kTargetFps = 60;
constexpr std::uint32_t kActionRepeat = 3U;
constexpr float kDecisionSeconds = static_cast<float>(kActionRepeat) / static_cast<float>(xushi2::sim::kTickHz);

struct CliArgs { std::optional<std::string> replay_path; std::optional<std::string> json_out; int warmup_frames=120; int measured_frames=600; bool benchmark=false; };

xushi2::sim::MatchConfig make_viewer_config() { xushi2::sim::MatchConfig c{}; c.seed=42; c.round_length_seconds=30; c.fog_of_war_enabled=false; c.randomize_map=false; c.action_repeat=kActionRepeat; c.mechanics.revolver_damage_centi_hp=7500U; c.mechanics.revolver_fire_cooldown_ticks=15U; c.mechanics.revolver_hitbox_radius=0.75F; c.mechanics.respawn_ticks=240U; return c; }

CliArgs parse_args(int argc, char** argv){ CliArgs args{}; for(int i=1;i<argc;++i){ std::string_view a=argv[i]; if(a=="--replay" && i+1<argc){ args.replay_path=argv[++i]; } else if(a=="--json-out" && i+1<argc){ args.json_out=argv[++i]; args.benchmark=true; } else if(a=="--bench-warmup-frames" && i+1<argc){ args.warmup_frames=std::max(0, std::atoi(argv[++i])); args.benchmark=true; } else if(a=="--bench-measured-frames" && i+1<argc){ args.measured_frames=std::max(1, std::atoi(argv[++i])); args.benchmark=true; } } return args; }

double percentile_ms(std::vector<double> samples, double p){ if(samples.empty()) return 0.0; std::sort(samples.begin(), samples.end()); double idx=(p/100.0)*static_cast<double>(samples.size()-1); auto lo=static_cast<size_t>(std::floor(idx)); auto hi=static_cast<size_t>(std::ceil(idx)); if(lo==hi) return samples[lo]; double w=idx-static_cast<double>(lo); return samples[lo]*(1.0-w)+samples[hi]*w; }

void write_bench_json(const std::string& path, const std::string& replay_name, int warmup, int measured, const std::vector<double>& frame_ms){
    const double sum=std::accumulate(frame_ms.begin(), frame_ms.end(), 0.0);
    const double avg=frame_ms.empty()?0.0:sum/static_cast<double>(frame_ms.size());
    const double fps=avg>0.0?1000.0/avg:0.0;
    const char* git_sha=std::getenv("GIT_COMMIT");
    if(git_sha==nullptr) git_sha=std::getenv("GITHUB_SHA");
    std::ofstream os(path);
    os << "{\n";
    if(git_sha!=nullptr){ os << "  \"git_commit\": \"" << git_sha << "\",\n"; } else { os << "  \"git_commit\": null,\n"; }
    os << "  \"replay_name\": \"" << replay_name << "\",\n";
    os << "  \"warmup_frames\": " << warmup << ",\n";
    os << "  \"measured_frames\": " << measured << ",\n";
    os << "  \"avg_ms\": " << avg << ",\n";
    os << "  \"p50_ms\": " << percentile_ms(frame_ms,50.0) << ",\n";
    os << "  \"p95_ms\": " << percentile_ms(frame_ms,95.0) << ",\n";
    os << "  \"p99_ms\": " << percentile_ms(frame_ms,99.0) << ",\n";
    os << "  \"fps\": " << fps << "\n";
    os << "}\n";
}

template <typename StepOnce, typename ResetPlayback>
void handle_input(xushi2::sim::Sim& sim,
                  bool& paused,
                  float& playback_speed,
                  float& decision_accum,
                  StepOnce& step_once,
                  ResetPlayback& reset_playback) {
    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_R)) reset_playback();
    if (IsKeyPressed(KEY_ONE)) playback_speed = 0.5F;
    if (IsKeyPressed(KEY_TWO)) playback_speed = 1.0F;
    if (IsKeyPressed(KEY_THREE)) playback_speed = 2.0F;
    if (IsKeyPressed(KEY_RIGHT) && !sim.episode_over()) {
        step_once();
        decision_accum = 0.0F;
    }
}

template <typename StepOnce>
void advance_playback(xushi2::sim::Sim& sim,
                      bool paused,
                      float playback_speed,
                      float& decision_accum,
                      StepOnce& step_once) {
    if (paused) return;

    decision_accum += GetFrameTime() * playback_speed;
    while (decision_accum >= kDecisionSeconds && !sim.episode_over()) {
        step_once();
        decision_accum -= kDecisionSeconds;
    }
}

PanelViewModel make_panel_model(
    const xushi2::sim::MatchState& state,
    const std::optional<Replay>& replay,
    bool paused,
    float playback_speed,
    LosDebugCounts los_counts,
    const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions,
    std::size_t replay_idx) {
    return PanelViewModel{
        .state = state,
        .replay_mode = replay.has_value(),
        .paused = paused,
        .playback_speed = playback_speed,
        .replay_phase = replay ? replay->phase : 0,
        .replay_fog = replay ? replay->fog : false,
        .replay_fog_mode = replay ? replay->fog_mode : std::string{},
        .replay_layout_hash = replay ? replay->layout_hash : std::string{},
        .replay_match_type = replay ? replay->match_type : std::string{},
        .replay_schedule_summary = replay ? replay->schedule_summary : std::string{},
        .replay_league_summary = replay ? replay->league_summary : std::string{},
        .replay_snapshot_group = replay ? replay->snapshot_group : std::string{},
        .replay_snapshot_name = replay ? replay->snapshot_name : std::string{},
        .replay_loss_mask = replay ? replay->loss_mask : std::string{},
        .replay_target_slot = replay ? replay->target_slot : false,
        .replay_last_seen = replay ? replay->last_seen : false,
        .replay_cover_count = replay ? replay->cover_markers.size() : 0U,
        .replay_wall_count = replay ? replay->wall_markers.size() : 0U,
        .replay_los_counts = los_counts,
        .replay_actions = actions,
        .replay_idx = replay_idx,
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
                bool paused,
                float playback_speed,
                std::size_t replay_idx) {
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

    const PanelViewModel panel_model =
        make_panel_model(s, replay, paused, playback_speed, los_counts, actions, replay_idx);
    draw_panel(panel_model);

    EndDrawing();
}
}

int main(int argc, char** argv) {
    const CliArgs cli = parse_args(argc, argv);
    std::optional<Replay> replay;
    if (cli.replay_path) replay = load_replay(*cli.replay_path);

    if (cli.benchmark) SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(kWindowWidth, kWindowHeight, "xushi2 viewer");
    SetTargetFPS(kTargetFps);

    xushi2::sim::MatchConfig config = replay ? replay->config : make_viewer_config();
    xushi2::sim::Sim sim(config);
    std::unique_ptr<xushi2::bots::IBot> bot_a = xushi2::bots::make_basic_bot();
    std::unique_ptr<xushi2::bots::IBot> bot_b = xushi2::bots::make_basic_bot();
    std::size_t replay_idx = 0;
    const ArenaTransform arena = make_arena_transform(config.map);
    const xushi2::common::Vec2 obj_center{0.5F*(config.map.min_x+config.map.max_x),0.5F*(config.map.min_y+config.map.max_y)};
    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch> shots{};
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch> tethers{};
    auto prev_heroes = sim.state().heroes;
    float decision_accum = 0.0F;
    bool paused = false;
    float playback_speed = 1.0F;

    auto reset_playback = [&]() {
        sim.reset();
        replay_idx = 0;
        actions = {};
        shots = {};
        tethers = {};
        prev_heroes = sim.state().heroes;
        decision_accum = 0.0F;
    };

    auto step_once = [&]() {
        actions = {};
        if (replay && replay_idx < replay->decisions.size()) {
            actions = replay->decisions[replay_idx].actions;
            ++replay_idx;
        } else if (!replay) {
            actions[0] = bot_a->decide(sim.state(), sim.config(), 0);
            actions[3] = bot_b->decide(sim.state(), sim.config(), 3);
        }

        sim.step_decision(actions);
        update_shot_tracers(shots, prev_heroes, sim.state().heroes, sim.state().tick);
        update_tether_trails(tethers, prev_heroes, sim.state().heroes, sim.state().tick);
        prev_heroes = sim.state().heroes;
    };

    std::vector<double> measured_ms; measured_ms.reserve(static_cast<size_t>(cli.measured_frames)); int bench_frame=0;
    while (!WindowShouldClose()) {
        const auto frame_start = std::chrono::steady_clock::now();
        handle_input(sim, paused, playback_speed, decision_accum, step_once, reset_playback);
        advance_playback(sim, paused, playback_speed, decision_accum, step_once);
        draw_frame(arena, obj_center, sim, replay, actions, shots, tethers, paused, playback_speed, replay_idx);
        if (sim.episode_over()) reset_playback();
        if (cli.benchmark) {
            const auto frame_end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            if (bench_frame >= cli.warmup_frames && static_cast<int>(measured_ms.size()) < cli.measured_frames) measured_ms.push_back(ms);
            ++bench_frame;
            if (static_cast<int>(measured_ms.size()) >= cli.measured_frames) break;
        }
    }
    if (cli.benchmark && cli.json_out) {
        const std::string replay_name = cli.replay_path ? *cli.replay_path : std::string("<none>");
        write_bench_json(*cli.json_out, replay_name, cli.warmup_frames, cli.measured_frames, measured_ms);
    }
    CloseWindow();
    return 0;
}
