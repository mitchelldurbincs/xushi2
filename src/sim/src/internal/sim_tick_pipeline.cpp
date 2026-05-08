#include "sim_tick_pipeline.h"

#include <cmath>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/math.hpp>

#include "sim_movement_geometry.h"
#include "sim_objective.h"
#include "sim_spawn_reset.h"
#include "sim_stage_mender.h"
#include "sim_stage_ranger.h"
#include "sim_stage_vanguard.h"
#include "sim_weapon_ranger.h"

namespace xushi2::sim::internal {

static constexpr float kVanguardSpeed = 3.6F;
static constexpr float kRangerSpeed = 4.2F;
static constexpr float kMenderSpeed = 4.0F;

static float hero_speed(common::HeroKind kind) { switch(kind){case common::HeroKind::Vanguard:return kVanguardSpeed;case common::HeroKind::Ranger:return kRangerSpeed;case common::HeroKind::Mender:return kMenderSpeed;} X2_UNREACHABLE(); }

static void stage_validate_and_aim(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = ctx.state.heroes[i];
        if (!h.present || !h.alive) continue;
        const common::Action& a = ctx.actions[i];
        X2_INVARIANT(std::isfinite(a.move_x) && std::isfinite(a.move_y), common::ErrorCode::NonFiniteFloat);
        X2_INVARIANT(std::isfinite(a.aim_delta), common::ErrorCode::NonFiniteFloat);
        if (!ctx.aim_consumed[i]) h.aim_angle = common::wrap_angle(h.aim_angle + common::clampf(a.aim_delta, -common::kAimDeltaMax, common::kAimDeltaMax));
    }
}

static void stage_movement_and_bounds(const TickContext& ctx) {
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        HeroState& h = ctx.state.heroes[i]; if (!h.present || !h.alive) continue;
        common::Vec2 move_vec = common::normalize_move_input(common::Vec2{ctx.actions[i].move_x, ctx.actions[i].move_y});
        float speed = hero_speed(h.kind); if (h.kind == common::HeroKind::Vanguard && h.vanguard_barrier_active) speed *= 0.7F;
        h.velocity = common::scale(move_vec, speed);
        common::Vec2 next = common::add(h.position, common::scale(h.velocity, kDt));
        next.x = common::clampf(next.x, ctx.config.map.min_x, ctx.config.map.max_x);
        next.y = common::clampf(next.y, ctx.config.map.min_y, ctx.config.map.max_y);
        next = prevent_wall_crossing(h.position, next, ctx.config);
        h.position = resolve_cover_overlap(next, move_vec, ctx.config);
    }
}

static void stage_cooldowns_and_weapon_tick(MatchState& state) { for (std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& h=state.heroes[i]; if(!h.present) continue; if(h.cd_ability_1>0)--h.cd_ability_1; if(h.cd_ability_2>0)--h.cd_ability_2; if(h.ranger_marked_ticks>0){--h.ranger_marked_ticks; if(h.ranger_marked_ticks==0) h.ranger_marked_by=common::Team::Neutral;} if(h.alive&&h.kind==common::HeroKind::Ranger) weapon_tick_update(h.weapon); else if(h.alive&&(h.kind==common::HeroKind::Vanguard||h.kind==common::HeroKind::Mender)&&h.weapon.fire_cooldown_ticks>0)--h.weapon.fire_cooldown_ticks; }}

static void stage_fire_resolution(MatchState& state,const std::array<common::Action,kAgentsPerMatch>& actions,const Phase1MechanicsConfig& m,const MatchConfig& config,DamageBuffer& buf,std::array<bool,kAgentsPerMatch>& has_damage){ resolve_revolver_fire(state,actions,m,config,buf,has_damage);} 
static void stage_apply_damage(MatchState& state,const DamageBuffer& buf,const std::array<bool,kAgentsPerMatch>& has_damage){ apply_damage_buffer(state,buf,has_damage);} 
static void stage_process_deaths(MatchState& state,const DamageBuffer& buf,const std::array<bool,kAgentsPerMatch>& has_damage,const MatchConfig& config){ process_deaths(state,buf,has_damage,config);} 
static void stage_respawn(MatchState& state,const MatchConfig& config){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i) respawn_tick_update(state.heroes[i],i,state.tick,config);} 
static void stage_objective(MatchState& state,const MapBounds& map){ objective_tick_update(state.objective,state.heroes,state.tick,map);} 

void apply_one_tick(MatchState& state, const MatchConfig& config,const std::array<common::Action, kAgentsPerMatch>& actions,const std::array<bool, kAgentsPerMatch>& aim_consumed) {
    TickContext ctx{state, config, actions, aim_consumed};
    stage_validate_and_aim(ctx);
    stage_abilities_vanguard_barrier(ctx);
    stage_movement_and_bounds(ctx);
    stage_cooldowns_and_weapon_tick(state);
    stage_abilities_combat_roll(ctx);
    stage_abilities_vanguard_guard_step(ctx);
    stage_abilities_ranger_mark_target(ctx);
    stage_abilities_mender_weapon_swap(ctx);
    stage_mender_staff_beam(ctx);
    stage_abilities_mender_tether(ctx);
    DamageBuffer buf{}; std::array<bool, kAgentsPerMatch> has_damage{};
    stage_fire_resolution(state, actions, config.mechanics, config, buf, has_damage);
    resolve_mender_sidearm_fire(state, actions, config.mechanics, config, buf, has_damage);
    stage_vanguard_warhammer(ctx, buf, has_damage);
    stage_apply_damage(state, buf, has_damage);
    stage_process_deaths(state, buf, has_damage, config);
    stage_respawn(state, config);
    stage_objective(state, config.map);
    state.tick += 1;
}

}  // namespace xushi2::sim::internal
