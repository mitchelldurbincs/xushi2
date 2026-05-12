#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include <raylib.h>

#include <xushi2/sim/sim.h>

#include "render_types.hpp"

struct PanelViewModel {
    const xushi2::sim::MatchState& state;
    bool replay_mode = false;
    bool paused = false;
    float playback_speed = 1.0F;
    int replay_phase = 0;
    bool replay_fog = false;
    std::string_view replay_fog_mode;
    std::string_view replay_layout_hash;
    std::string_view replay_match_type;
    std::string_view replay_schedule_summary;
    std::string_view replay_league_summary;
    std::string_view replay_snapshot_group;
    std::string_view replay_snapshot_name;
    std::string_view replay_loss_mask;
    bool replay_target_slot = false;
    bool replay_last_seen = false;
    std::size_t replay_cover_count = 0;
    std::size_t replay_wall_count = 0;
    LosDebugCounts replay_los_counts{};
    const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& replay_actions;
    std::size_t replay_idx = 0;
    std::size_t replay_total = 0;
};

void draw_panel(const PanelViewModel& model);

void draw_panel_header(const PanelViewModel& model, int x, int& y);
void draw_panel_score_objective(const PanelViewModel& model, int x, int& y);
void draw_panel_replay_section(const PanelViewModel& model, int x, int& y);
void draw_panel_heroes(const PanelViewModel& model, int x, int& y);
