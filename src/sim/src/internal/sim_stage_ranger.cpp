#include "sim_stage_ranger.h"

#include <cmath>

#include <xushi2/common/math.hpp>

#include "sim_combat.h"
#include "sim_movement_geometry.h"
#include "sim_target_query.h"
#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

static void maybe_combat_roll(HeroState& h, const common::Action& a, bool aim_consumed,
                              const MatchConfig& config) {
    if (aim_consumed || !a.ability_1 || !h.alive || h.cd_ability_1 != 0) return;
    common::Vec2 dir{}; const float move_mag_sq=a.move_x*a.move_x+a.move_y*a.move_y;
    if (move_mag_sq > 1e-6F) { const float inv=1.0F/std::sqrt(move_mag_sq); dir={a.move_x*inv,a.move_y*inv}; }
    else { dir={std::cos(h.aim_angle),std::sin(h.aim_angle)}; }
    common::Vec2 next{h.position.x + dir.x * common::kRangerCombatRollDistance,h.position.y + dir.y * common::kRangerCombatRollDistance};
    next.x = common::clampf(next.x, config.map.min_x, config.map.max_x); next.y = common::clampf(next.y, config.map.min_y, config.map.max_y);
    next = prevent_wall_crossing(h.position, next, config); h.position = resolve_cover_overlap(next, dir, config);
    weapon_on_combat_roll(h.weapon); h.cd_ability_1 = common::kRangerCombatRollCooldownTicks;
}

void stage_abilities_combat_roll(const TickContext& ctx){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& h=ctx.state.heroes[i]; if(!h.present||h.kind!=common::HeroKind::Ranger) continue; maybe_combat_roll(h,ctx.actions[i],ctx.aim_consumed[i],ctx.config);} }

void stage_abilities_ranger_mark_target(const TickContext& ctx){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& ranger=ctx.state.heroes[i]; if(!ranger.present||!ranger.alive||ranger.kind!=common::HeroKind::Ranger) continue; const auto& a=ctx.actions[i]; if(ctx.aim_consumed[i]||!a.ability_2||a.target_slot!=1||ranger.cd_ability_2!=0) continue; int best_slot=nearest_enemy_in_range_with_los(ctx.state,i,common::kRangerRevolverRange,ctx.config); ranger.cd_ability_2=common::kRangerMarkTargetCooldownTicks; if(best_slot>=0){HeroState& target=ctx.state.heroes[(std::uint32_t)best_slot]; target.ranger_marked_ticks=common::kRangerMarkTargetDurationTicks; target.ranger_marked_by=ranger.team;}} }

}  // namespace
