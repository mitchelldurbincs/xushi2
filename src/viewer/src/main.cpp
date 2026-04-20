// Xushi2 raylib viewer.
//
// Phase-0 scaffold: opens a window, runs an empty sim loop at 30 Hz, and
// renders a placeholder frame. Real debug overlays (vision cones, fog,
// raycasts, shields, cooldowns, last-seen ghosts, reward events) are
// specified in game-design.md §15 and rl-design.md §9 and land as the sim
// logic does.

#include <raylib.h>

#include <xushi2/bots/bot.h>
#include <xushi2/sim/sim.h>

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kTargetFps = 60;  // render rate; sim runs at 30 Hz internally

}  // namespace

int main() {
    InitWindow(kWindowWidth, kWindowHeight, "xushi2 viewer");
    SetTargetFPS(kTargetFps);

    xushi2::sim::MatchConfig config{};
    config.seed = 42;
    xushi2::sim::Sim sim(config);

    std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch> actions{};

    while (!WindowShouldClose()) {
        // TODO: translate keyboard / mouse into actions[0] (human-controlled
        // hero slot). Until then all agents idle.
        sim.step(actions);

        BeginDrawing();
        ClearBackground(BLACK);

        DrawText("xushi2 — viewer scaffold", 40, 40, 24, RAYWHITE);
        DrawText(TextFormat("tick: %u", sim.state().tick), 40, 80, 18, LIGHTGRAY);
        DrawText(TextFormat("score  A: %u   B: %u",
                            sim.state().objective.team_a_score_ticks,
                            sim.state().objective.team_b_score_ticks),
                 40, 110, 18, LIGHTGRAY);
        DrawText("sim pipeline not yet implemented — see src/sim/src/sim.cpp",
                 40, kWindowHeight - 40, 14, GRAY);

        EndDrawing();

        if (sim.episode_over()) {
            sim.reset();
        }
    }

    CloseWindow();
    return 0;
}
