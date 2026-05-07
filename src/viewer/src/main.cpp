// Xushi2 raylib viewer.
//
// Phase-0/1 scaffold: opens a window, runs the deterministic sim at 30 Hz,
// and renders a top-down view of the arena with the objective, both teams'
// Rangers (position, facing, HP), and live score / cap progress. Real
// debug overlays for vision cones, fog, raycasts, shields, cooldowns,
// last-seen ghosts, and reward events are specified in game-design.md §15
// and rl-design.md §9 and land as the sim logic does.

#include <raylib.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kTargetFps = 60;  // render rate; sim runs at 30 Hz internally
constexpr std::uint32_t kActionRepeat = 3U;
constexpr float kDecisionSeconds =
    static_cast<float>(kActionRepeat) / static_cast<float>(xushi2::sim::kTickHz);

// Layout: a square arena viewport on the left, an info panel on the right.
constexpr int kArenaPx = 720;       // square; matches window height
constexpr int kArenaMarginPx = 12;  // padding inside the arena viewport
constexpr int kPanelX = kArenaPx;
constexpr int kPanelW = kWindowWidth - kArenaPx;

// Shot tracers: a brief line from shooter origin along aim direction, fading
// over kShotFadeTicks (sim ticks). One slot per hero is plenty since the
// minimum revolver fire cadence (15 ticks) is well above the fade window.
constexpr std::uint32_t kShotFadeTicks = 12U;  // 0.4s at 30 Hz
struct ShotTracer {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

xushi2::sim::MatchConfig make_viewer_config() {
    xushi2::sim::MatchConfig config{};
    config.seed = 42;
    config.round_length_seconds = 30;
    config.fog_of_war_enabled = false;
    config.randomize_map = false;
    config.action_repeat = kActionRepeat;

    config.mechanics.revolver_damage_centi_hp = 7500U;
    config.mechanics.revolver_fire_cooldown_ticks = 15U;
    config.mechanics.revolver_hitbox_radius = 0.75F;
    config.mechanics.respawn_ticks = 240U;
    return config;
}

// --- Replay support ---------------------------------------------------------

struct ReplayDecision {
    std::uint32_t tick;
    xushi2::common::Action slot0;
    xushi2::common::Action slot3;
};

struct Replay {
    xushi2::sim::MatchConfig config;
    std::vector<ReplayDecision> decisions;
};

bool parse_kv_double(const std::string& s, const char* key, double& out) {
    const std::string needle = std::string(key) + "=";
    const auto pos = s.find(needle);
    if (pos == std::string::npos) return false;
    const auto start = pos + needle.size();
    const auto end = s.find(' ', start);
    const std::string val = s.substr(start, end - start);
    try {
        out = std::stod(val);
    } catch (...) {
        return false;
    }
    return true;
}

std::optional<Replay> load_replay(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        TraceLog(LOG_ERROR, "replay: cannot open %s", path.c_str());
        return std::nullopt;
    }
    std::string header;
    if (!std::getline(in, header)) {
        TraceLog(LOG_ERROR, "replay: empty file");
        return std::nullopt;
    }
    Replay rep{};
    rep.config = make_viewer_config();  // seeded with sane defaults; header overrides

    double v = 0.0;
    if (parse_kv_double(header, "seed", v))           rep.config.seed = static_cast<std::uint64_t>(v);
    if (parse_kv_double(header, "round_seconds", v))  rep.config.round_length_seconds = static_cast<int>(v);
    if (parse_kv_double(header, "action_repeat", v))  rep.config.action_repeat = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "mech_dmg", v))       rep.config.mechanics.revolver_damage_centi_hp = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "mech_fcd", v))       rep.config.mechanics.revolver_fire_cooldown_ticks = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "mech_hbr", v))       rep.config.mechanics.revolver_hitbox_radius = static_cast<float>(v);
    if (parse_kv_double(header, "mech_resp", v))      rep.config.mechanics.respawn_ticks = static_cast<std::uint32_t>(v);

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        ReplayDecision d{};
        std::uint32_t tick = 0;
        float mx0=0, my0=0, ad0=0, pf0=0, a10=0, a20=0;
        float mx3=0, my3=0, ad3=0, pf3=0, a13=0, a23=0;
        if (!(iss >> tick >> mx0 >> my0 >> ad0 >> pf0 >> a10 >> a20
                  >> mx3 >> my3 >> ad3 >> pf3 >> a13 >> a23)) {
            TraceLog(LOG_WARNING, "replay: skipping malformed line");
            continue;
        }
        d.tick = tick;
        d.slot0 = xushi2::common::Action{mx0, my0, ad0, pf0 >= 0.5F, a10 >= 0.5F, a20 >= 0.5F, 0};
        d.slot3 = xushi2::common::Action{mx3, my3, ad3, pf3 >= 0.5F, a13 >= 0.5F, a23 >= 0.5F, 0};
        rep.decisions.push_back(d);
    }
    TraceLog(LOG_INFO, "replay: loaded %zu decisions from %s", rep.decisions.size(), path.c_str());
    return rep;
}

// World → screen mapping for the square arena viewport. The world is a
// rectangle [min_x, max_x] × [min_y, max_y] (Phase-1 default 0..50 × 0..50).
// World +Y points up; screen +Y points down — flip at conversion time.
struct ArenaTransform {
    float world_min_x;
    float world_min_y;
    float world_w;
    float world_h;
    float pixels_per_unit;  // assumes square arena; world_w == world_h
    float screen_origin_x;  // top-left corner of the arena rect in screen px
    float screen_origin_y;
};

ArenaTransform make_arena_transform(const xushi2::sim::MapBounds& m) {
    const float ww = m.max_x - m.min_x;
    const float wh = m.max_y - m.min_y;
    const float inner = static_cast<float>(kArenaPx - 2 * kArenaMarginPx);
    const float scale = inner / std::max(ww, wh);
    return ArenaTransform{
        m.min_x, m.min_y, ww, wh, scale,
        static_cast<float>(kArenaMarginPx),
        static_cast<float>(kArenaMarginPx),
    };
}

Vector2 world_to_screen(const ArenaTransform& t, xushi2::common::Vec2 p) {
    const float sx = t.screen_origin_x + (p.x - t.world_min_x) * t.pixels_per_unit;
    // Flip Y so world +Y appears upward on screen.
    const float sy = t.screen_origin_y + (t.world_h - (p.y - t.world_min_y)) * t.pixels_per_unit;
    return Vector2{sx, sy};
}

float world_len_to_screen(const ArenaTransform& t, float u) {
    return u * t.pixels_per_unit;
}

Color team_color(xushi2::common::Team team) {
    switch (team) {
        case xushi2::common::Team::A: return Color{82, 156, 255, 255};   // blue
        case xushi2::common::Team::B: return Color{255, 96, 96, 255};    // red
        default: return GRAY;
    }
}

void draw_arena(const ArenaTransform& t) {
    // Arena background.
    DrawRectangle(static_cast<int>(t.screen_origin_x),
                  static_cast<int>(t.screen_origin_y),
                  static_cast<int>(t.world_w * t.pixels_per_unit),
                  static_cast<int>(t.world_h * t.pixels_per_unit),
                  Color{18, 22, 28, 255});
    // Arena border.
    DrawRectangleLines(static_cast<int>(t.screen_origin_x),
                       static_cast<int>(t.screen_origin_y),
                       static_cast<int>(t.world_w * t.pixels_per_unit),
                       static_cast<int>(t.world_h * t.pixels_per_unit),
                       Color{60, 70, 84, 255});
}

void draw_objective(const ArenaTransform& t, const xushi2::sim::ObjectiveState& obj,
                    xushi2::common::Vec2 center) {
    const Vector2 c = world_to_screen(t, center);
    const float r = world_len_to_screen(t, xushi2::common::kObjectiveRadius);
    // Filled disc tinted by current owner.
    Color fill = Color{40, 50, 64, 180};
    if (obj.owner == xushi2::common::Team::A) fill = Color{40, 70, 120, 200};
    else if (obj.owner == xushi2::common::Team::B) fill = Color{120, 50, 50, 200};
    DrawCircleV(c, r, fill);
    DrawCircleLinesV(c, r, Color{200, 200, 80, 255});

    // Capture-progress arc (cap_progress_ticks / kCaptureTicks of a full ring).
    if (obj.cap_progress_ticks > 0 && obj.cap_team != xushi2::common::Team::Neutral) {
        const float frac = static_cast<float>(obj.cap_progress_ticks) /
                           static_cast<float>(xushi2::common::kCaptureTicks);
        const float sweep = 360.0F * frac;
        DrawRing(c, r * 0.85F, r * 0.95F, -90.0F, -90.0F + sweep, 64,
                 team_color(obj.cap_team));
    }
}

void draw_hero(const ArenaTransform& t, const xushi2::sim::HeroState& h) {
    if (!h.present) return;
    const Vector2 c = world_to_screen(t, h.position);
    const float body_r = world_len_to_screen(t, 0.6F);
    const Color color = team_color(h.team);

    if (!h.alive) {
        // Greyed-out dead marker.
        DrawCircleV(c, body_r, Color{60, 60, 60, 200});
        DrawCircleLinesV(c, body_r, Color{120, 120, 120, 255});
        return;
    }

    // Body.
    DrawCircleV(c, body_r, color);
    DrawCircleLinesV(c, body_r, RAYWHITE);

    // Facing arrow: aim_angle is in radians, 0 = +x. World +Y is up, so on
    // screen we negate the y-component (matches world_to_screen's flip).
    const float arrow_world_len = 1.4F;
    const float ax = h.position.x + std::cos(h.aim_angle) * arrow_world_len;
    const float ay = h.position.y + std::sin(h.aim_angle) * arrow_world_len;
    const Vector2 tip = world_to_screen(t, xushi2::common::Vec2{ax, ay});
    DrawLineEx(c, tip, 2.5F, RAYWHITE);

    // HP bar above the hero.
    const int bar_w = static_cast<int>(world_len_to_screen(t, 1.6F));
    const int bar_h = 5;
    const int bar_x = static_cast<int>(c.x) - bar_w / 2;
    const int bar_y = static_cast<int>(c.y - body_r) - bar_h - 4;
    const float hp_frac = h.max_health_centi_hp > 0
        ? std::max(0.0F, static_cast<float>(h.health_centi_hp) /
                        static_cast<float>(h.max_health_centi_hp))
        : 0.0F;
    DrawRectangle(bar_x, bar_y, bar_w, bar_h, Color{30, 30, 30, 220});
    DrawRectangle(bar_x, bar_y, static_cast<int>(bar_w * hp_frac), bar_h,
                  Color{120, 220, 120, 255});
    DrawRectangleLines(bar_x, bar_y, bar_w, bar_h, Color{80, 80, 80, 200});
}

void draw_shot_tracers(const ArenaTransform& t,
                       const std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
                       xushi2::sim::Tick now) {
    for (const auto& sh : shots) {
        if (!sh.active) continue;
        const std::uint32_t age = now - sh.fired_tick;
        if (age >= kShotFadeTicks) continue;
        const float alpha = 1.0F - (static_cast<float>(age) /
                                    static_cast<float>(kShotFadeTicks));
        Color base = team_color(sh.team);
        base.a = static_cast<unsigned char>(220.0F * alpha);
        const Vector2 a = world_to_screen(t, sh.start);
        const Vector2 b = world_to_screen(t, sh.end);
        DrawLineEx(a, b, 2.0F, base);
    }
}

void update_shot_tracers(
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
    xushi2::sim::Tick now) {
    for (std::size_t i = 0; i < curr.size(); ++i) {
        const auto& p = prev[i];
        const auto& c = curr[i];
        if (!c.present) continue;
        // A shot fires when magazine decrements (and the hero was alive on
        // the previous tick — reload jumps magazine 0 → max which is an
        // increment, so we don't trip on it).
        if (p.alive && c.alive && c.weapon.magazine + 1U == p.weapon.magazine) {
            const float ax = std::cos(c.aim_angle);
            const float ay = std::sin(c.aim_angle);
            shots[i] = ShotTracer{
                true,
                c.position,
                xushi2::common::Vec2{
                    c.position.x + ax * xushi2::common::kRangerRevolverRange,
                    c.position.y + ay * xushi2::common::kRangerRevolverRange,
                },
                c.team,
                now,
            };
        }
    }
}

void draw_panel(const xushi2::sim::MatchState& s) {
    const int x = kPanelX + 24;
    int y = 32;
    DrawText("xushi2 viewer", x, y, 22, RAYWHITE); y += 32;
    DrawText("(basic vs basic)", x, y, 14, GRAY); y += 28;

    DrawText(TextFormat("tick     %u", s.tick), x, y, 18, LIGHTGRAY); y += 24;
    const float seconds = static_cast<float>(s.tick) /
                          static_cast<float>(xushi2::common::kTickHz);
    DrawText(TextFormat("time     %.1fs", seconds), x, y, 18, LIGHTGRAY); y += 32;

    DrawText("score", x, y, 16, GRAY); y += 22;
    DrawText(TextFormat("  A  %u", s.objective.team_a_score_ticks),
             x, y, 18, team_color(xushi2::common::Team::A)); y += 22;
    DrawText(TextFormat("  B  %u", s.objective.team_b_score_ticks),
             x, y, 18, team_color(xushi2::common::Team::B)); y += 28;

    DrawText("objective", x, y, 16, GRAY); y += 22;
    const char* owner_label = "neutral";
    Color owner_col = GRAY;
    if (s.objective.owner == xushi2::common::Team::A) {
        owner_label = "team A"; owner_col = team_color(xushi2::common::Team::A);
    } else if (s.objective.owner == xushi2::common::Team::B) {
        owner_label = "team B"; owner_col = team_color(xushi2::common::Team::B);
    }
    DrawText(TextFormat("  owner   %s", owner_label), x, y, 16, owner_col); y += 22;
    DrawText(TextFormat("  cap     %u/%u",
                        s.objective.cap_progress_ticks,
                        xushi2::common::kCaptureTicks),
             x, y, 16, LIGHTGRAY); y += 22;
    DrawText(TextFormat("  unlocked %s", s.objective.unlocked ? "yes" : "no"),
             x, y, 16, LIGHTGRAY); y += 32;

    DrawText("heroes", x, y, 16, GRAY); y += 22;
    for (std::size_t i = 0; i < s.heroes.size(); ++i) {
        const auto& h = s.heroes[i];
        if (!h.present) continue;
        const Color c = team_color(h.team);
        const char* status = h.alive ? "alive" : "dead";
        const int hp_show = h.health_centi_hp / 100;
        const int hp_max  = h.max_health_centi_hp / 100;
        DrawText(TextFormat("  slot %zu  %s  %d/%d", i, status, hp_show, hp_max),
                 x, y, 14, c);
        y += 18;
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
    auto prev_heroes = sim.state().heroes;
    float decision_accum = 0.0F;

    while (!WindowShouldClose()) {
        decision_accum += GetFrameTime();
        while (decision_accum >= kDecisionSeconds && !sim.episode_over()) {
            // TODO: translate keyboard / mouse into actions[0] for a human-
            // controlled hero. For now, drive the present Phase-1 Ranger slots
            // with deterministic baseline bots so the placeholder is live.
            actions = {};
            if (replay && replay_idx < replay->decisions.size()) {
                actions[0] = replay->decisions[replay_idx].slot0;
                actions[3] = replay->decisions[replay_idx].slot3;
                ++replay_idx;
            } else if (!replay) {
                actions[0] = bot_a->decide(sim.state(), 0);
                actions[3] = bot_b->decide(sim.state(), 3);
            }
            // (replay exhausted: no-op actions; sim coasts to round end)
            sim.step_decision(actions);
            update_shot_tracers(shots, prev_heroes, sim.state().heroes,
                                sim.state().tick);
            prev_heroes = sim.state().heroes;
            decision_accum -= kDecisionSeconds;
        }

        BeginDrawing();
        ClearBackground(Color{8, 10, 14, 255});

        const auto& s = sim.state();
        draw_arena(arena);
        draw_objective(arena, s.objective, obj_center);
        draw_shot_tracers(arena, shots, s.tick);
        for (const auto& h : s.heroes) {
            draw_hero(arena, h);
        }
        draw_panel(s);

        EndDrawing();

        if (sim.episode_over()) {
            sim.reset();
            replay_idx = 0;
            decision_accum = 0.0F;
        }
    }

    CloseWindow();
    return 0;
}
