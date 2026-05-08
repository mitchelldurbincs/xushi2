#pragma once

#include <raylib.h>

#include <xushi2/common/types.hpp>
#include <xushi2/common/vec2.hpp>
#include <xushi2/sim/sim.h>

#include "render_types.hpp"

ArenaTransform make_arena_transform(const xushi2::sim::MapBounds& m);
Vector2 world_to_screen(const ArenaTransform& t, xushi2::common::Vec2 p);
float world_len_to_screen(const ArenaTransform& t, float u);
Color team_color(xushi2::common::Team team);
void draw_arena(const ArenaTransform& t);
void draw_objective(const ArenaTransform& t, const xushi2::sim::ObjectiveState& obj,
                    xushi2::common::Vec2 center);
void draw_hero(const ArenaTransform& t, const xushi2::sim::HeroState& h);
