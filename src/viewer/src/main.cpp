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
#include <charconv>
#include <cerrno>
#include <cstdlib>
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

#include "render_arena.hpp"
#include "render_debug.hpp"
#include "panel.hpp"
#include "viewer_layout.hpp"

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


bool parse_float_token(std::string_view token, float& out) {
#if defined(__cpp_lib_to_chars) && (__cpp_lib_to_chars >= 201611L)
    const char* begin = token.data();
    const char* end = token.data() + token.size();
    const auto parsed = std::from_chars(begin, end, out);
    return parsed.ec == std::errc{} && parsed.ptr == end;
#else
    std::string tmp(token);
    char* parse_end = nullptr;
    errno = 0;
    const float value = std::strtof(tmp.c_str(), &parse_end);
    if (parse_end != tmp.c_str() + tmp.size() || errno == ERANGE) return false;
    out = value;
    return true;
#endif
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
            const auto radius_sep = token.find(':', colon + 1);
            float x = 0.0F;
            float y = 0.0F;
            float radius = 1.0F;
            const bool parsed_xy =
                parse_float_token(token.substr(0, colon), x) &&
                parse_float_token(
                    token.substr(
                        colon + 1,
                        radius_sep == std::string_view::npos
                            ? std::string_view::npos
                            : radius_sep - colon - 1),
                    y);
            const bool parsed_radius =
                radius_sep == std::string_view::npos ||
                parse_float_token(token.substr(radius_sep + 1), radius);
            if (parsed_xy && parsed_radius) {
                markers.push_back(CoverMarker{xushi2::common::Vec2{x, y}, radius});
            } else {
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
        const std::string_view token = s.substr(start, len);
        std::array<float, 5> values{0.0F, 0.0F, 0.0F, 0.0F, 0.25F};
        std::size_t value_index = 0;
        std::size_t token_start = 0;
        bool malformed = false;
        while (token_start <= token.size() && value_index < values.size()) {
            const std::size_t sep = token.find(':', token_start);
            const std::size_t part_len =
                (sep == std::string_view::npos) ? std::string_view::npos : sep - token_start;
            if (!parse_float_token(token.substr(token_start, part_len), values[value_index])) {
                malformed = true;
                break;
            }
            ++value_index;
            if (sep == std::string_view::npos) break;
            token_start = sep + 1;
        }
        if (!malformed && value_index == values.size() && token.find(':', token_start) == std::string_view::npos) {
            markers.push_back(WallMarker{
                xushi2::common::Vec2{values[0], values[1]},
                xushi2::common::Vec2{values[2], values[3]},
                values[4],
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
