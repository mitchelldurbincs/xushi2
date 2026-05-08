#pragma once

#include <array>
#include <vector>

#include <xushi2/sim/sim.h>

#include "render_types.hpp"

struct CoverMarker {
    xushi2::common::Vec2 center{};
    float radius = 1.0F;
};

struct WallMarker {
    xushi2::common::Vec2 a{};
    xushi2::common::Vec2 b{};
    float half_width = 0.25F;
};

void draw_cover_markers(const ArenaTransform& t,
                        const std::vector<CoverMarker>& markers);
void draw_wall_markers(const ArenaTransform& t,
                       const std::vector<WallMarker>& markers);
LosDebugCounts draw_line_of_sight_debug(const ArenaTransform& t,
                                        const xushi2::sim::Sim& sim);

void draw_shot_tracers(const ArenaTransform& t,
                       const std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
                       xushi2::sim::Tick now);
void update_shot_tracers(
    std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
    xushi2::sim::Tick now);

void draw_tether_trails(
    const ArenaTransform& t,
    const std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
    xushi2::sim::Tick now);
void update_tether_trails(
    std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
    xushi2::sim::Tick now);

void draw_mender_beams(
    const ArenaTransform& t,
    const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& heroes);

void draw_target_token_debug(
    const ArenaTransform& t,
    const xushi2::sim::MatchState& s,
    const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions);
