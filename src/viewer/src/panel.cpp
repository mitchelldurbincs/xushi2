#include "panel.hpp"

#include <xushi2/common/limits.hpp>

#include "viewer_layout.hpp"

namespace {

constexpr int kPanelX = viewer_layout::kPanelX;

Color team_color(xushi2::common::Team team) {
    switch (team) {
        case xushi2::common::Team::A: return Color{255, 95, 95, 255};
        case xushi2::common::Team::B: return Color{95, 160, 255, 255};
        case xushi2::common::Team::Neutral: return Color{200, 200, 200, 255};
    }
    return LIGHTGRAY;
}

const char* hero_kind_label(xushi2::common::HeroKind kind) {
    switch (kind) {
        case xushi2::common::HeroKind::Vanguard: return "vanguard";
        case xushi2::common::HeroKind::Ranger: return "ranger";
        case xushi2::common::HeroKind::Mender: return "mender";
    }
    return "unknown";
}

const char* mender_weapon_label(xushi2::common::MenderWeapon weapon) {
    switch (weapon) {
        case xushi2::common::MenderWeapon::Staff: return "staff";
        case xushi2::common::MenderWeapon::Sidearm: return "sidearm";
    }
    return "unknown";
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

}  // namespace

void draw_panel_header(const PanelViewModel& model, int x, int& y) {
    DrawText("xushi2 viewer", x, y, 22, RAYWHITE); y += 32;
    DrawText(model.replay_mode ? "replay playback" : "basic vs basic", x, y, 14, GRAY); y += 24;
    DrawText(TextFormat("state    %s  %.1fx", model.paused ? "paused" : "running", static_cast<double>(model.playback_speed)), x, y, 16, LIGHTGRAY); y += 22;
    DrawText("keys     Space pause  Right step  R reset  1/2/3 speed", x, y, 13, GRAY); y += 28;

    DrawText(TextFormat("tick     %u", model.state.tick), x, y, 18, LIGHTGRAY); y += 24;
    const float seconds = static_cast<float>(model.state.tick) /
                          static_cast<float>(xushi2::common::kTickHz);
    DrawText(TextFormat("time     %.1fs", seconds), x, y, 18, LIGHTGRAY); y += 32;
}

void draw_panel_score_objective(const PanelViewModel& model, int x, int& y) {
    const auto& s = model.state;
    DrawText("score", x, y, 16, GRAY); y += 22;
    DrawText(TextFormat("  A  %u", s.objective.team_a_score_ticks), x, y, 18, team_color(xushi2::common::Team::A)); y += 22;
    DrawText(TextFormat("  B  %u", s.objective.team_b_score_ticks), x, y, 18, team_color(xushi2::common::Team::B)); y += 28;

    DrawText("objective", x, y, 16, GRAY); y += 22;
    const char* owner_label = "neutral";
    Color owner_col = GRAY;
    if (s.objective.owner == xushi2::common::Team::A) {
        owner_label = "team A"; owner_col = team_color(xushi2::common::Team::A);
    } else if (s.objective.owner == xushi2::common::Team::B) {
        owner_label = "team B"; owner_col = team_color(xushi2::common::Team::B);
    }
    DrawText(TextFormat("  owner   %s", owner_label), x, y, 16, owner_col); y += 22;
    DrawText(TextFormat("  cap     %u/%u", s.objective.cap_progress_ticks, xushi2::common::kCaptureTicks), x, y, 16, LIGHTGRAY); y += 22;
    DrawText(TextFormat("  unlocked %s", s.objective.unlocked ? "yes" : "no"), x, y, 16, LIGHTGRAY); y += 32;
}

void draw_panel_replay_section(const PanelViewModel& model, int x, int& y) {
    if (!model.replay_mode) return;
    DrawText("replay", x, y, 16, GRAY); y += 22;
    if (model.replay_phase > 0) { DrawText(TextFormat("  phase    %d", model.replay_phase), x, y, 16, LIGHTGRAY); y += 22; }
    if (model.replay_fog) {
        const char* mode = model.replay_fog_mode.empty() ? "diagnostic" : model.replay_fog_mode.data();
        DrawText(TextFormat("  fog      %s", mode), x, y, 16, LIGHTGRAY); y += 22;
    }
    if (!model.replay_layout_hash.empty()) { DrawText(TextFormat("  layout   %s", std::string(model.replay_layout_hash).c_str()), x, y, 16, LIGHTGRAY); y += 22; }
    if (!model.replay_match_type.empty()) { DrawText(TextFormat("  match    %s", std::string(model.replay_match_type).c_str()), x, y, 16, LIGHTGRAY); y += 22; }
    if (!model.replay_snapshot_group.empty()) { DrawText(TextFormat("  league   %s", std::string(model.replay_snapshot_group).c_str()), x, y, 16, LIGHTGRAY); y += 22; }
    if (!model.replay_snapshot_name.empty()) { DrawText(TextFormat("  snapshot %.28s", std::string(model.replay_snapshot_name).c_str()), x, y, 16, LIGHTGRAY); y += 22; }
    if (!model.replay_league_summary.empty()) { DrawText(TextFormat("  weights  %.30s", std::string(model.replay_league_summary).c_str()), x, y, 13, GRAY); y += 18; }
    if (!model.replay_schedule_summary.empty()) { DrawText(TextFormat("  schedule %.30s", std::string(model.replay_schedule_summary).c_str()), x, y, 13, GRAY); y += 18; }
    if (!model.replay_loss_mask.empty()) { DrawText(TextFormat("  lossmask %s", std::string(model.replay_loss_mask).c_str()), x, y, 16, LIGHTGRAY); y += 22; }
    if (model.replay_target_slot) {
        DrawText("  target   enabled", x, y, 16, LIGHTGRAY); y += 22;
        DrawText(TextFormat("  tokens   0:%s 1:%s 2:%s", target_token_label(model.replay_actions[0].target_slot), target_token_label(model.replay_actions[1].target_slot), target_token_label(model.replay_actions[2].target_slot)), x, y, 13, GRAY); y += 18;
    }
    if (model.replay_last_seen) { DrawText("  lastseen enabled", x, y, 16, LIGHTGRAY); y += 22; }
    if (model.replay_cover_count > 0) { DrawText(TextFormat("  cover    %zu", model.replay_cover_count), x, y, 16, LIGHTGRAY); y += 22; }
    if (model.replay_wall_count > 0) { DrawText(TextFormat("  walls    %zu", model.replay_wall_count), x, y, 16, LIGHTGRAY); y += 22; }
    if (model.replay_los_counts.visible + model.replay_los_counts.blocked > 0) {
        DrawText(TextFormat("  los      %zu/%zu", model.replay_los_counts.visible, model.replay_los_counts.visible + model.replay_los_counts.blocked), x, y, 16, LIGHTGRAY); y += 22;
    }
    DrawText(TextFormat("  decision %zu/%zu", model.replay_idx, model.replay_total), x, y, 16, LIGHTGRAY); y += 28;
}

void draw_panel_heroes(const PanelViewModel& model, int x, int& y) {
    DrawText("heroes", x, y, 16, GRAY); y += 22;
    for (std::size_t i = 0; i < model.state.heroes.size(); ++i) {
        const auto& h = model.state.heroes[i];
        if (!h.present) continue;
        const Color c = team_color(h.team);
        const char* status = h.alive ? "alive" : "dead";
        const int hp_show = h.health_centi_hp / 100;
        const int hp_max = h.max_health_centi_hp / 100;
        DrawText(TextFormat("  slot %zu  %.3s  %s  %d/%d", i, hero_kind_label(h.kind), status, hp_show, hp_max), x, y, 14, c);
        y += 18;
        if (h.kind == xushi2::common::HeroKind::Mender) {
            DrawText(TextFormat("    weapon  %s", mender_weapon_label(h.mender_weapon)), x, y, 13, LIGHTGRAY);
            y += 16;
            if (h.mender_beam_locked_on != 0) {
                DrawText(TextFormat("    beam    id %u", h.mender_beam_locked_on), x, y, 13, Color{120, 255, 170, 255});
                y += 16;
            }
        }
        if (h.ranger_marked_ticks > 0) {
            DrawText(TextFormat("    marked  %u", h.ranger_marked_ticks), x, y, 13, Color{255, 214, 92, 255});
            y += 16;
        }
    }
}

void draw_panel(const PanelViewModel& model) {
    const int x = kPanelX + 24;
    int y = 32;
    draw_panel_header(model, x, y);
    draw_panel_score_objective(model, x, y);
    draw_panel_replay_section(model, x, y);
    draw_panel_heroes(model, x, y);
}
