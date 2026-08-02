#include "render_arena.hpp"

#include <algorithm>
#include <cmath>

#include "viewer_layout.hpp"

ArenaTransform make_arena_transform(const xushi2::sim::MapBounds& m) {
    const float ww = m.max_x - m.min_x;
    const float wh = m.max_y - m.min_y;
    const float inner = static_cast<float>(viewer_layout::kArenaPx - 2 * viewer_layout::kArenaMarginPx);
    const float scale = inner / std::max(ww, wh);
    return ArenaTransform{
        m.min_x, m.min_y, ww, wh, scale,
        static_cast<float>(viewer_layout::kArenaMarginPx),
        static_cast<float>(viewer_layout::kArenaMarginPx),
    };
}

Vector2 world_to_screen(const ArenaTransform& t, xushi2::common::Vec2 p) {
    const float sx = t.screen_origin_x + (p.x - t.world_min_x) * t.pixels_per_unit;
    const float sy = t.screen_origin_y + (t.world_h - (p.y - t.world_min_y)) * t.pixels_per_unit;
    return Vector2{sx, sy};
}

float world_len_to_screen(const ArenaTransform& t, float u) { return u * t.pixels_per_unit; }

// Must agree with panel.cpp's team_color — Team A (the learner in every
// replay we dump) is red, Team B is blue. These were opposite conventions
// until 2026-08-02, so the arena painted the learner blue while the panel
// called Team A red.
Color team_color(xushi2::common::Team team) {
    switch (team) {
        case xushi2::common::Team::A: return Color{255, 96, 96, 255};
        case xushi2::common::Team::B: return Color{82, 156, 255, 255};
        default: return GRAY;
    }
}

void draw_arena(const ArenaTransform& t) {
    DrawRectangle(static_cast<int>(t.screen_origin_x), static_cast<int>(t.screen_origin_y),
                  static_cast<int>(t.world_w * t.pixels_per_unit),
                  static_cast<int>(t.world_h * t.pixels_per_unit),
                  Color{18, 22, 28, 255});
    DrawRectangleLines(static_cast<int>(t.screen_origin_x), static_cast<int>(t.screen_origin_y),
                       static_cast<int>(t.world_w * t.pixels_per_unit),
                       static_cast<int>(t.world_h * t.pixels_per_unit),
                       Color{60, 70, 84, 255});
}

void draw_objective(const ArenaTransform& t, const xushi2::sim::ObjectiveState& obj,
                    xushi2::common::Vec2 center) {
    const Vector2 c = world_to_screen(t, center);
    const float r = world_len_to_screen(t, xushi2::common::kObjectiveRadius);
    Color fill = Color{40, 50, 64, 180};
    if (obj.owner == xushi2::common::Team::A) fill = Color{120, 50, 50, 200};
    else if (obj.owner == xushi2::common::Team::B) fill = Color{40, 70, 120, 200};
    DrawCircleV(c, r, fill);
    DrawCircleLinesV(c, r, Color{200, 200, 80, 255});

    if (obj.cap_progress_ticks > 0 && obj.cap_team != xushi2::common::Team::Neutral) {
        const float frac = static_cast<float>(obj.cap_progress_ticks) /
                           static_cast<float>(xushi2::common::kCaptureTicks);
        const float sweep = 360.0F * frac;
        DrawRing(c, r * 0.85F, r * 0.95F, -90.0F, -90.0F + sweep, 64, team_color(obj.cap_team));
    }
}

void draw_hero(const ArenaTransform& t, const xushi2::sim::HeroState& h) {
    if (!h.present) return;
    const Vector2 c = world_to_screen(t, h.position);
    const float body_r = world_len_to_screen(t, 0.6F);
    const Color color = team_color(h.team);
    if (!h.alive) {
        DrawCircleV(c, body_r, Color{60, 60, 60, 200});
        DrawCircleLinesV(c, body_r, Color{120, 120, 120, 255});
        return;
    }
    DrawCircleV(c, body_r, color);
    DrawCircleLinesV(c, body_r, RAYWHITE);
    if (h.vanguard_barrier_active && h.vanguard_barrier_hp_centi > 0) {
        Color barrier_col = color;
        barrier_col.a = 90;
        DrawCircleV(c, world_len_to_screen(t, xushi2::common::kVanguardBarrierRadius), barrier_col);
        DrawCircleLinesV(c, world_len_to_screen(t, xushi2::common::kVanguardBarrierRadius),
                         Color{180, 220, 255, 220});
    }
    if (h.ranger_marked_ticks > 0) DrawCircleLinesV(c, body_r + 6.0F, Color{255, 214, 92, 255});

    const float ax = h.position.x + std::cos(h.aim_angle) * 1.4F;
    const float ay = h.position.y + std::sin(h.aim_angle) * 1.4F;
    DrawLineEx(c, world_to_screen(t, xushi2::common::Vec2{ax, ay}), 2.5F, RAYWHITE);

    const int bar_w = static_cast<int>(world_len_to_screen(t, 1.6F));
    const int bar_x = static_cast<int>(c.x) - bar_w / 2;
    const int bar_y = static_cast<int>(c.y - body_r) - 9;
    const float hp_frac = h.max_health_centi_hp > 0
        ? std::max(0.0F, static_cast<float>(h.health_centi_hp) / static_cast<float>(h.max_health_centi_hp))
        : 0.0F;
    DrawRectangle(bar_x, bar_y, bar_w, 5, Color{30, 30, 30, 220});
    DrawRectangle(bar_x, bar_y, static_cast<int>(bar_w * hp_frac), 5, Color{120, 220, 120, 255});
    DrawRectangleLines(bar_x, bar_y, bar_w, 5, Color{80, 80, 80, 200});
}
