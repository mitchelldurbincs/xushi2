#include "sim_tick_pipeline.h"

// This file is orchestration-only. Keep game-rule implementations in dedicated
// stage modules and leave this pipeline as an ordered sequence of stage calls.

#include "sim_objective.h"
#include "sim_spawn_reset.h"
#include "sim_stage_mender.h"
#include "sim_stage_ranger.h"
#include "sim_stage_vanguard.h"
#include "sim_tick_stages.h"

namespace xushi2::sim::internal {

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
