// Xushi2 raylib viewer.
//
// Phase-0/1 scaffold: opens a window, runs the deterministic sim at 30 Hz,
// and renders a top-down view of the arena with the objective, both teams'
// Rangers (position, facing, HP), and live score / cap progress. Real
// debug overlays for vision cones, fog, raycasts, shields, cooldowns,
// last-seen ghosts, and reward events are specified in game-design.md §15
// and rl-design.md §9 and land as the sim logic does.

#include <raylib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
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
constexpr int kTargetFps = 60;  // render rate; sim runs at 30 Hz internally
constexpr std::uint32_t kActionRepeat = 3U;
constexpr float kDecisionSeconds =
    static_cast<float>(kActionRepeat) / static_cast<float>(xushi2::sim::kTickHz);

// Layout: a square arena viewport on the left, an info panel on the right.
constexpr int kArenaPx = viewer_layout::kArenaPx;
constexpr int kArenaMarginPx = viewer_layout::kArenaMarginPx;
constexpr int kPanelX = viewer_layout::kPanelX;

constexpr std::uint32_t kShotFadeTicks = 12U;  // 0.4s at 30 Hz
struct ShotTracer {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

struct TetherTrail {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

const char* target_token_label(std::uint8_t target_slot) {
    switch (target_slot) {
        case 0: return "self";
        case 1: return "enemy0";
        case 2: return "enemy1";
        case 3: return "enemy2";
        case 4: return "objective";
        default: return "?";
    }
}


}  // namespace

int main(int argc, char** argv) {
    // CLI: --replay <path> drives the sim from a dumped greedy episode.
    std::optional<Replay> replay;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--replay") == 0) {
            replay = load_replay(argv[i + 1]);
            ++i;
        }
    }

    InitWindow(kWindowWidth, kWindowHeight, "xushi2 viewer");
    SetTargetFPS(kTargetFps);

    xushi2::sim::MatchConfig config =
        replay ? replay->config : make_viewer_config();
    xushi2::sim::Sim sim(config);
    std::unique_ptr<xushi2::bots::IBot> bot_a = xushi2::bots::make_basic_bot();
    std::unique_ptr<xushi2::bots::IBot> bot_b = xushi2::bots::make_basic_bot();
    std::size_t replay_idx = 0;

    const ArenaTransform arena = make_arena_transform(config.map);
    const xushi2::common::Vec2 obj_center{
        0.5F * (config.map.min_x + config.map.max_x),
        0.5F * (config.map.min_y + config.map.max_y),
    };

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch> shots{};
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch> tethers{};
    auto prev_heroes = sim.state().heroes;
    float decision_accum = 0.0F;
    bool paused = false;
    float playback_speed = 1.0F;

    const auto reset_playback = [&]() {
        sim.reset();
        replay_idx = 0;
        actions = {};
        shots = {};
        tethers = {};
        prev_heroes = sim.state().heroes;
        decision_accum = 0.0F;
    };

    const auto step_once = [&]() {
        actions = {};
        if (replay && replay_idx < replay->decisions.size()) {
            actions = replay->decisions[replay_idx].actions;
            ++replay_idx;
        } else if (!replay) {
            actions[0] = bot_a->decide(sim.state(), 0);
            actions[3] = bot_b->decide(sim.state(), 3);
        }
        // (replay exhausted: no-op actions; sim coasts to round end)
        sim.step_decision(actions);
        update_shot_tracers(shots, prev_heroes, sim.state().heroes,
                            sim.state().tick);
        update_tether_trails(tethers, prev_heroes, sim.state().heroes,
                             sim.state().tick);
        prev_heroes = sim.state().heroes;
    };

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) reset_playback();
        if (IsKeyPressed(KEY_ONE)) playback_speed = 0.5F;
        if (IsKeyPressed(KEY_TWO)) playback_speed = 1.0F;
        if (IsKeyPressed(KEY_THREE)) playback_speed = 2.0F;
        if (IsKeyPressed(KEY_RIGHT) && !sim.episode_over()) {
            step_once();
            decision_accum = 0.0F;
        }

        if (!paused) {
            decision_accum += GetFrameTime() * playback_speed;
            while (decision_accum >= kDecisionSeconds && !sim.episode_over()) {
                step_once();
                decision_accum -= kDecisionSeconds;
            }
        }

        BeginDrawing();
        ClearBackground(Color{8, 10, 14, 255});

        const auto& s = sim.state();
        draw_arena(arena);
        if (replay) {
            draw_wall_markers(arena, replay->wall_markers);
            draw_cover_markers(arena, replay->cover_markers);
        }
        const LosDebugCounts los_counts =
            (replay && replay->fog) ? draw_line_of_sight_debug(arena, sim)
                                    : LosDebugCounts{};
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
        const PanelViewModel panel_model{
            .state = s,
            .replay_mode = replay.has_value(),
            .paused = paused,
            .playback_speed = playback_speed,
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
            .replay_idx = replay_idx,
            .replay_total = replay ? replay->decisions.size() : 0U,
        };
        draw_panel(panel_model);

        EndDrawing();

        if (sim.episode_over()) {
            reset_playback();
        }
    }

    CloseWindow();
    return 0;
}
