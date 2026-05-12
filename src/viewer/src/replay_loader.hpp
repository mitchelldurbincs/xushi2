#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <xushi2/common/types.h>
#include <xushi2/sim/sim.h>

#include "render_types.hpp"

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

// Load a text replay file and return decoded metadata + per-tick actions.
// Accepted decision line payloads (after tick token):
// - 12 values legacy format (slot 0 then slot 3, 6 values each).
// - 6 values per agent (kAgentsPerMatch * 6 total).
// - 7 values per agent (kAgentsPerMatch * 7 total, includes target token).
std::optional<Replay> load_replay(const std::string& path);
