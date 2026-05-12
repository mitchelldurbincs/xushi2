#include "render_debug.hpp"

#include <algorithm>
#include <cmath>

#include <raylib.h>

#include "render_arena.hpp"
#include "viewer_labels.hpp"

namespace {
constexpr std::uint32_t kShotFadeTicks = 12U;

}  // namespace

void draw_cover_markers(const ArenaTransform& t, const std::vector<CoverMarker>& markers) {
    for (const auto marker : markers) {
        const Vector2 c = world_to_screen(t, marker.center);
        const float r = world_len_to_screen(t, marker.radius);
        DrawCircleV(c, r, Color{95, 105, 112, 190});
        DrawCircleLinesV(c, r, Color{190, 200, 210, 220});
    }
}

void draw_wall_markers(const ArenaTransform& t, const std::vector<WallMarker>& markers) {
    for (const auto marker : markers) {
        const Vector2 a = world_to_screen(t, marker.a);
        const Vector2 b = world_to_screen(t, marker.b);
        const float width = std::max(2.0F, world_len_to_screen(t, marker.half_width * 2.0F));
        DrawLineEx(a, b, width, Color{90, 98, 106, 220});
        DrawLineEx(a, b, 1.0F, Color{205, 214, 222, 220});
    }
}

LosDebugCounts draw_line_of_sight_debug(const ArenaTransform& t, const xushi2::sim::Sim& sim) {
    LosDebugCounts counts{};
    const auto& heroes = sim.state().heroes;
    for (std::size_t a_slot = 0; a_slot < xushi2::common::kTeamSize; ++a_slot) {
        const auto& a = heroes[a_slot];
        if (!a.present || !a.alive) continue;
        for (std::size_t b_slot = xushi2::common::kTeamSize; b_slot < xushi2::sim::kAgentsPerMatch; ++b_slot) {
            const auto& b = heroes[b_slot];
            if (!b.present || !b.alive) continue;
            const bool visible = sim.line_of_sight(static_cast<std::uint32_t>(a_slot), static_cast<std::uint32_t>(b_slot));
            const Color col = visible ? Color{90, 230, 135, 90} : Color{255, 100, 100, 80};
            DrawLineEx(world_to_screen(t, a.position), world_to_screen(t, b.position), visible ? 1.5F : 1.0F, col);
            visible ? ++counts.visible : ++counts.blocked;
        }
    }
    return counts;
}

void draw_shot_tracers(const ArenaTransform& t, const std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
                       xushi2::sim::Tick now) {
    for (const auto& sh : shots) {
        if (!sh.active) continue;
        const std::uint32_t age = now - sh.fired_tick;
        if (age >= kShotFadeTicks) continue;
        const float alpha = 1.0F - (static_cast<float>(age) / static_cast<float>(kShotFadeTicks));
        Color base = team_color(sh.team);
        base.a = static_cast<unsigned char>(220.0F * alpha);
        DrawLineEx(world_to_screen(t, sh.start), world_to_screen(t, sh.end), 2.0F, base);
    }
}

void update_shot_tracers(std::array<ShotTracer, xushi2::sim::kAgentsPerMatch>& shots,
                         const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
                         const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
                         xushi2::sim::Tick now) {
    for (std::size_t i = 0; i < curr.size(); ++i) {
        const auto& p = prev[i];
        const auto& c = curr[i];
        if (!c.present) continue;
        if (p.alive && c.alive && c.weapon.magazine + 1U == p.weapon.magazine) {
            const float ax = std::cos(c.aim_angle);
            const float ay = std::sin(c.aim_angle);
            shots[i] = ShotTracer{true, c.position,
                                  xushi2::common::Vec2{c.position.x + ax * xushi2::common::kRangerRevolverRange,
                                                       c.position.y + ay * xushi2::common::kRangerRevolverRange},
                                  c.team, now};
        } else if (p.alive && c.alive && c.kind == xushi2::common::HeroKind::Mender &&
                   c.mender_weapon == xushi2::common::MenderWeapon::Sidearm &&
                   p.weapon.fire_cooldown_ticks == 0 &&
                   c.weapon.fire_cooldown_ticks == xushi2::common::kMenderSidearmCooldownTicks) {
            const float ax = std::cos(c.aim_angle);
            const float ay = std::sin(c.aim_angle);
            shots[i] = ShotTracer{true, c.position,
                                  xushi2::common::Vec2{c.position.x + ax * xushi2::common::kMenderSidearmRange,
                                                       c.position.y + ay * xushi2::common::kMenderSidearmRange},
                                  c.team, now};
        }
    }
}

void draw_tether_trails(const ArenaTransform& t,
                        const std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
                        xushi2::sim::Tick now) {
    for (const auto& tr : trails) {
        if (!tr.active) continue;
        const std::uint32_t age = now - tr.fired_tick;
        if (age >= kShotFadeTicks) continue;
        const float alpha = 1.0F - (static_cast<float>(age) / static_cast<float>(kShotFadeTicks));
        Color col = team_color(tr.team);
        col.a = static_cast<unsigned char>(180.0F * alpha);
        DrawLineEx(world_to_screen(t, tr.start), world_to_screen(t, tr.end), 4.0F, col);
    }
}

void update_tether_trails(std::array<TetherTrail, xushi2::sim::kAgentsPerMatch>& trails,
                          const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& prev,
                          const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& curr,
                          xushi2::sim::Tick now) {
    for (std::size_t i = 0; i < curr.size(); ++i) {
        const auto& p = prev[i];
        const auto& c = curr[i];
        if (!p.present || !c.present || !p.alive || !c.alive || c.kind != xushi2::common::HeroKind::Mender) continue;
        const float dx = c.position.x - p.position.x;
        const float dy = c.position.y - p.position.y;
        const bool tether_cd_armed = p.cd_ability_2 == 0 && c.cd_ability_2 == xushi2::common::kMenderTetherCooldownTicks;
        if (tether_cd_armed && (dx * dx + dy * dy) > 1.0F) trails[i] = TetherTrail{true, p.position, c.position, c.team, now};
    }
}

void draw_mender_beams(const ArenaTransform& t,
                       const std::array<xushi2::sim::HeroState, xushi2::sim::kAgentsPerMatch>& heroes) {
    for (const auto& h : heroes) {
        if (!h.present || !h.alive || h.kind != xushi2::common::HeroKind::Mender || h.mender_beam_locked_on == 0) continue;
        const auto target_it = std::find_if(heroes.begin(), heroes.end(), [&](const xushi2::sim::HeroState& other) {
            return other.present && other.id == h.mender_beam_locked_on;
        });
        if (target_it == heroes.end() || !target_it->alive) continue;
        const Vector2 a = world_to_screen(t, h.position);
        const Vector2 b = world_to_screen(t, target_it->position);
        DrawLineEx(a, b, 3.0F, Color{120, 255, 170, 210});
        DrawCircleV(b, 5.0F, Color{160, 255, 190, 180});
    }
}

void draw_target_token_debug(const ArenaTransform& t, const xushi2::sim::MatchState& s,
                             const std::array<xushi2::sim::Action, xushi2::sim::kAgentsPerMatch>& actions) {
    for (std::uint32_t slot = 0; slot < s.heroes.size(); ++slot) {
        const auto& h = s.heroes[slot];
        if (!h.present || !h.alive) continue;
        const auto target = actions[slot].target_slot;
        if (target == 0 && !actions[slot].ability_2) continue;
        const Vector2 p = world_to_screen(t, h.position);
        DrawText(TextFormat("t:%s", viewer_labels::target_slot_label(target, viewer_labels::TargetSlotLabelMode::Compact)), static_cast<int>(p.x + 8.0F), static_cast<int>(p.y - 22.0F), 13,
                 Color{245, 225, 120, 220});
        if (target >= 1 && target <= 3) {
            const std::uint32_t enemy_idx = static_cast<std::uint32_t>(target - 1U);
            const std::uint32_t enemy_slot = slot < 3U ? enemy_idx + 3U : enemy_idx;
            if (enemy_slot < s.heroes.size() && s.heroes[enemy_slot].present && s.heroes[enemy_slot].alive) {
                DrawLineEx(p, world_to_screen(t, s.heroes[enemy_slot].position), 1.0F, Color{245, 225, 120, 90});
            }
        }
    }
}
