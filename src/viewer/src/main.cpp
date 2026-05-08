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
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/sim/sim.h>

#include "panel.hpp"

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

struct TetherTrail {
    bool active = false;
    xushi2::common::Vec2 start{};
    xushi2::common::Vec2 end{};
    xushi2::common::Team team = xushi2::common::Team::Neutral;
    xushi2::sim::Tick fired_tick = 0;
};

struct CoverMarker {
    xushi2::common::Vec2 center{};
    float radius = 1.0F;
};

struct WallMarker {
    xushi2::common::Vec2 a{};
    xushi2::common::Vec2 b{};
    float half_width = 0.25F;
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
    std::array<xushi2::common::Action, xushi2::sim::kAgentsPerMatch> actions{};
};

struct Replay {
    xushi2::sim::MatchConfig config;
    int phase = 0;
    bool fog = false;
    std::string fog_mode;
    std::string layout_hash;
    std::string match_type;
    std::string schedule_summary;
    std::string league_summary;
    std::string snapshot_group;
    std::string snapshot_name;
    std::string loss_mask;
    bool target_slot = false;
    bool last_seen = false;
    std::vector<CoverMarker> cover_markers;
    std::vector<WallMarker> wall_markers;
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

std::optional<std::string> parse_kv_string(const std::string& s, const char* key) {
    const std::string needle = std::string(key) + "=";
    const auto pos = s.find(needle);
    if (pos == std::string::npos) return std::nullopt;
    const auto start = pos + needle.size();
    const auto end = s.find(' ', start);
    return s.substr(start, end - start);
}

std::vector<CoverMarker> parse_cover_markers(std::string_view s) {
    std::vector<CoverMarker> markers;
    std::size_t start = 0;
    while (start < s.size()) {
        const auto comma = s.find(',', start);
        const auto len = (comma == std::string_view::npos)
            ? std::string_view::npos
            : comma - start;
        const std::string_view token = s.substr(start, len);
        const auto colon = token.find(':');
        if (colon != std::string_view::npos) {
            try {
                const float x = std::stof(std::string(token.substr(0, colon)));
                const auto radius_sep = token.find(':', colon + 1);
                const float y = std::stof(std::string(
                    token.substr(
                        colon + 1,
                        radius_sep == std::string_view::npos
                            ? std::string_view::npos
                            : radius_sep - colon - 1)));
                const float radius =
                    radius_sep == std::string_view::npos
                        ? 1.0F
                        : std::stof(std::string(token.substr(radius_sep + 1)));
                markers.push_back(CoverMarker{xushi2::common::Vec2{x, y}, radius});
            } catch (...) {
                TraceLog(LOG_WARNING, "replay: skipping malformed cover marker");
            }
        }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return markers;
}

std::vector<WallMarker> parse_wall_markers(std::string_view s) {
    std::vector<WallMarker> markers;
    std::size_t start = 0;
    while (start < s.size()) {
        const std::size_t comma = s.find(',', start);
        const std::size_t len = (comma == std::string_view::npos)
            ? std::string_view::npos
            : comma - start;
        const std::string text(s.substr(start, len));
        float x1 = 0.0F;
        float y1 = 0.0F;
        float x2 = 0.0F;
        float y2 = 0.0F;
        float half_width = 0.25F;
        if (std::sscanf(text.c_str(), "%f:%f:%f:%f:%f",
                        &x1, &y1, &x2, &y2, &half_width) == 5) {
            markers.push_back(WallMarker{
                xushi2::common::Vec2{x1, y1},
                xushi2::common::Vec2{x2, y2},
                half_width,
            });
        } else {
            TraceLog(LOG_WARNING, "replay: skipping malformed wall marker");
        }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return markers;
}

xushi2::common::HeroKind parse_hero_kind(std::string_view s) {
    if (s == "vanguard") return xushi2::common::HeroKind::Vanguard;
    if (s == "mender") return xushi2::common::HeroKind::Mender;
    return xushi2::common::HeroKind::Ranger;
}

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
    if (parse_kv_double(header, "phase", v))          rep.phase = static_cast<int>(v);
    if (parse_kv_double(header, "fog", v))            rep.fog = (v != 0.0);
    if (parse_kv_double(header, "target_slot", v))    rep.target_slot = (v != 0.0);
    if (parse_kv_double(header, "last_seen", v))      rep.last_seen = (v != 0.0);
    if (const auto fog_mode = parse_kv_string(header, "fog_mode")) {
        rep.fog_mode = *fog_mode;
    }
    if (const auto layout = parse_kv_string(header, "layout")) {
        rep.layout_hash = *layout;
    }
    if (const auto match_type = parse_kv_string(header, "match_type")) {
        rep.match_type = *match_type;
    }
    if (const auto league = parse_kv_string(header, "league")) {
        rep.league_summary = *league;
    }
    if (const auto schedule = parse_kv_string(header, "schedule")) {
        rep.schedule_summary = *schedule;
    }
    if (const auto group = parse_kv_string(header, "snapshot_group")) {
        rep.snapshot_group = *group;
    }
    if (const auto snapshot = parse_kv_string(header, "snapshot")) {
        rep.snapshot_name = *snapshot;
    }
    if (const auto loss_mask = parse_kv_string(header, "loss_mask")) {
        rep.loss_mask = *loss_mask;
    }
    if (parse_kv_double(header, "round_seconds", v))  rep.config.round_length_seconds = static_cast<int>(v);
    if (parse_kv_double(header, "action_repeat", v))  rep.config.action_repeat = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "team_size", v))       rep.config.team_size = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "map_min_x", v))       rep.config.map.min_x = static_cast<float>(v);
    if (parse_kv_double(header, "map_min_y", v))       rep.config.map.min_y = static_cast<float>(v);
    if (parse_kv_double(header, "map_max_x", v))       rep.config.map.max_x = static_cast<float>(v);
    if (parse_kv_double(header, "map_max_y", v))       rep.config.map.max_y = static_cast<float>(v);
    if (const auto covers = parse_kv_string(header, "cover")) {
        rep.cover_markers = parse_cover_markers(*covers);
    }
    if (const auto walls = parse_kv_string(header, "walls")) {
        rep.wall_markers = parse_wall_markers(*walls);
    }
    if (parse_kv_double(header, "mech_dmg", v))       rep.config.mechanics.revolver_damage_centi_hp = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "mech_fcd", v))       rep.config.mechanics.revolver_fire_cooldown_ticks = static_cast<std::uint32_t>(v);
    if (parse_kv_double(header, "mech_hbr", v))       rep.config.mechanics.revolver_hitbox_radius = static_cast<float>(v);
    if (parse_kv_double(header, "mech_resp", v))      rep.config.mechanics.respawn_ticks = static_cast<std::uint32_t>(v);
    if (const auto heroes = parse_kv_string(header, "heroes")) {
        std::size_t start = 0;
        for (std::size_t slot = 0; slot < rep.config.hero_kinds.size(); ++slot) {
            const auto end = heroes->find(',', start);
            const auto len = (end == std::string::npos) ? std::string::npos : end - start;
            rep.config.hero_kinds[slot] = parse_hero_kind(std::string_view(*heroes).substr(start, len));
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }

    const auto make_action = [](const std::vector<float>& values,
                                std::size_t offset,
                                std::size_t stride) {
        return xushi2::common::Action{
            values[offset + 0],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3] >= 0.5F,
            values[offset + 4] >= 0.5F,
            values[offset + 5] >= 0.5F,
            static_cast<std::uint8_t>(
                stride >= 7U ? std::clamp(values[offset + 6], 0.0F, 255.0F)
                             : 0.0F),
        };
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        ReplayDecision d{};
        std::uint32_t tick = 0;
        if (!(iss >> tick)) {
            TraceLog(LOG_WARNING, "replay: skipping malformed line");
            continue;
        }
        std::vector<float> values;
        float value = 0.0F;
        while (iss >> value) {
            values.push_back(value);
        }
        d.tick = tick;
        if (values.size() == 12U) {
            // Legacy text replay: slot 0 then slot 3.
            d.actions[0] = make_action(values, 0U, 6U);
            d.actions[3] = make_action(values, 6U, 6U);
        } else if (values.size() == xushi2::sim::kAgentsPerMatch * 6U) {
            for (std::size_t slot = 0; slot < xushi2::sim::kAgentsPerMatch; ++slot) {
                d.actions[slot] = make_action(values, slot * 6U, 6U);
            }
        } else if (values.size() == xushi2::sim::kAgentsPerMatch * 7U) {
            for (std::size_t slot = 0; slot < xushi2::sim::kAgentsPerMatch; ++slot) {
                d.actions[slot] = make_action(values, slot * 7U, 7U);
            }
        } else {
            TraceLog(LOG_WARNING, "replay: skipping line with %zu action values", values.size());
            continue;
        }
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

void draw_cover_markers(const ArenaTransform& t,
                        const std::vector<CoverMarker>& markers) {
    for (const auto marker : markers) {
        const Vector2 c = world_to_screen(t, marker.center);
        const float r = world_len_to_screen(t, marker.radius);
        DrawCircleV(c, r, Color{95, 105, 112, 190});
        DrawCircleLinesV(c, r, Color{190, 200, 210, 220});
    }
}

void draw_wall_markers(const ArenaTransform& t,
                       const std::vector<WallMarker>& markers) {
    for (const auto marker : markers) {
        const Vector2 a = world_to_screen(t, marker.a);
        const Vector2 b = world_to_screen(t, marker.b);
        const float width =
            std::max(2.0F, world_len_to_screen(t, marker.half_width * 2.0F));
        DrawLineEx(a, b, width, Color{90, 98, 106, 220});
        DrawLineEx(a, b, 1.0F, Color{205, 214, 222, 220});
    }
}

LosDebugCounts draw_line_of_sight_debug(const ArenaTransform& t,
                                        const xushi2::sim::Sim& sim) {
    LosDebugCounts counts{};
    const auto& heroes = sim.state().heroes;
    for (std::size_t a_slot = 0; a_slot < xushi2::common::kTeamSize; ++a_slot) {
        const auto& a = heroes[a_slot];
        if (!a.present || !a.alive) continue;
        for (std::size_t b_slot = xushi2::common::kTeamSize;
             b_slot < xushi2::sim::kAgentsPerMatch;
             ++b_slot) {
            const auto& b = heroes[b_slot];
            if (!b.present || !b.alive) continue;
            const bool visible = sim.line_of_sight(
                static_cast<std::uint32_t>(a_slot),
                static_cast<std::uint32_t>(b_slot));
            const Color col = visible ? Color{90, 230, 135, 90}
                                      : Color{255, 100, 100, 80};
            DrawLineEx(world_to_screen(t, a.position),
                       world_to_screen(t, b.position),
                       visible ? 1.5F : 1.0F,
                       col);
            if (visible) {
                ++counts.visible;
            } else {
                ++counts.blocked;
            }
        }
    }
    return counts;
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
    if (h.vanguard_barrier_active && h.vanguard_barrier_hp_centi > 0) {
        Color barrier_col = color;
        barrier_col.a = 90;
        DrawCircleV(c, world_len_to_screen(t, xushi2::common::kVanguardBarrierRadius),
                    barrier_col);
        DrawCircleLinesV(c, world_len_to_screen(t, xushi2::common::kVanguardBarrierRadius),
                         Color{180, 220, 255, 220});
    }
    if (h.ranger_marked_ticks > 0) {
        DrawCircleLinesV(c, body_r + 6.0F, Color{255, 214, 92, 255});
    }

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

void draw_mender_beams(
    const ArenaTransform& t,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& heroes) {
    for (const auto& h : heroes) {
        if (!h.present || !h.alive || h.kind != xushi2::common::HeroKind::Mender ||
            h.mender_beam_locked_on == 0) {
            continue;
        }
        const auto target_it = std::find_if(
            heroes.begin(), heroes.end(),
            [&](const xushi2::sim::HeroState& other) {
                return other.present && other.id == h.mender_beam_locked_on;
            });
        if (target_it == heroes.end() || !target_it->alive) {
            continue;
        }
        const Vector2 a = world_to_screen(t, h.position);
        const Vector2 b = world_to_screen(t, target_it->position);
        DrawLineEx(a, b, 3.0F, Color{120, 255, 170, 210});
        DrawCircleV(b, 5.0F, Color{160, 255, 190, 180});
    }
}

void draw_target_token_debug(
    const ArenaTransform& t,
    const xushi2::sim::MatchState& s,
    const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions) {
    for (std::uint32_t slot = 0; slot < s.heroes.size(); ++slot) {
        const auto& h = s.heroes[slot];
        if (!h.present || !h.alive) {
            continue;
        }
        const auto target = actions[slot].target_slot;
        if (target == 0 && !actions[slot].ability_2) {
            continue;
        }
        const Vector2 p = world_to_screen(t, h.position);
        Color col = Color{245, 225, 120, 220};
        DrawText(TextFormat("t:%s", target_token_label(target)),
                 static_cast<int>(p.x + 8.0F),
                 static_cast<int>(p.y - 22.0F),
                 13,
                 col);
        if (target >= 1 && target <= 3) {
            const std::uint32_t enemy_idx = static_cast<std::uint32_t>(target - 1U);
            const std::uint32_t enemy_slot = slot < 3U ? enemy_idx + 3U : enemy_idx;
            if (enemy_slot < s.heroes.size() && s.heroes[enemy_slot].present &&
                s.heroes[enemy_slot].alive) {
                DrawLineEx(p, world_to_screen(t, s.heroes[enemy_slot].position),
                           1.0F, Color{245, 225, 120, 90});
            }
        }
    }
}

void draw_tether_trails(
    const ArenaTransform& t,
    const std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
    xushi2::sim::Tick now) {
    for (const auto& tr : trails) {
        if (!tr.active) continue;
        const std::uint32_t age = now - tr.fired_tick;
        if (age >= kShotFadeTicks) continue;
        const float alpha = 1.0F - (static_cast<float>(age) /
                                    static_cast<float>(kShotFadeTicks));
        Color col = team_color(tr.team);
        col.a = static_cast<unsigned char>(180.0F * alpha);
        DrawLineEx(world_to_screen(t, tr.start), world_to_screen(t, tr.end), 4.0F, col);
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
        } else if (p.alive && c.alive && c.kind == xushi2::common::HeroKind::Mender &&
                   c.mender_weapon == xushi2::common::MenderWeapon::Sidearm &&
                   p.weapon.fire_cooldown_ticks == 0 &&
                   c.weapon.fire_cooldown_ticks == xushi2::common::kMenderSidearmCooldownTicks) {
            const float ax = std::cos(c.aim_angle);
            const float ay = std::sin(c.aim_angle);
            shots[i] = ShotTracer{
                true,
                c.position,
                xushi2::common::Vec2{
                    c.position.x + ax * xushi2::common::kMenderSidearmRange,
                    c.position.y + ay * xushi2::common::kMenderSidearmRange,
                },
                c.team,
                now,
            };
        }
    }
}

void update_tether_trails(
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
    xushi2::sim::Tick now) {
    for (std::size_t i = 0; i < curr.size(); ++i) {
        const auto& p = prev[i];
        const auto& c = curr[i];
        if (!p.present || !c.present || !p.alive || !c.alive ||
            c.kind != xushi2::common::HeroKind::Mender) {
            continue;
        }
        const float dx = c.position.x - p.position.x;
        const float dy = c.position.y - p.position.y;
        const bool tether_cd_armed =
            p.cd_ability_2 == 0 &&
            c.cd_ability_2 == xushi2::common::kMenderTetherCooldownTicks;
        if (tether_cd_armed && (dx * dx + dy * dy) > 1.0F) {
            trails[i] = TetherTrail{true, p.position, c.position, c.team, now};
        }
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
