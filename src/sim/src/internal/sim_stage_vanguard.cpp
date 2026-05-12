#include "sim_stage_vanguard.h"

#include <cmath>

#include <xushi2/common/math.hpp>

#include "sim_combat.h"
#include "sim_movement_geometry.h"

namespace xushi2::sim::internal {

void stage_abilities_vanguard_barrier(const TickContext& ctx){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& h=ctx.state.heroes[i]; if(!h.present||h.kind!=common::HeroKind::Vanguard) continue; if(!h.alive||!ctx.actions[i].ability_1||h.cd_ability_1!=0){h.vanguard_barrier_active=false; continue;} if(h.vanguard_barrier_hp_centi<=0) h.vanguard_barrier_hp_centi=common::kVanguardBarrierHpCenti; h.vanguard_barrier_active=true;}}

void stage_abilities_vanguard_guard_step(const TickContext& ctx){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& h=ctx.state.heroes[i]; if(!h.present||!h.alive||h.kind!=common::HeroKind::Vanguard) continue; const auto& a=ctx.actions[i]; if(ctx.aim_consumed[i]||!a.ability_2||h.cd_ability_2!=0) continue; common::Vec2 dir{std::cos(h.aim_angle),std::sin(h.aim_angle)}; common::Vec2 next{h.position.x+dir.x*common::kVanguardGuardStepDistance,h.position.y+dir.y*common::kVanguardGuardStepDistance}; // Intentionally delegate movement collision/bounds handling to shared geometry helper.
 h.position=resolve_displaced_position(h.position,next,dir,ctx.config); h.cd_ability_2=common::kVanguardGuardStepCooldownTicks;}}

void stage_vanguard_warhammer(const TickContext& ctx, DamageBuffer& buf,std::array<bool, kAgentsPerMatch>& has_damage){ for(std::uint32_t i=0;i<kAgentsPerMatch;++i){HeroState& v=ctx.state.heroes[i]; if(!v.present||!v.alive||v.kind!=common::HeroKind::Vanguard) continue; const auto& a=ctx.actions[i]; if(ctx.aim_consumed[i]||!a.primary_fire||v.vanguard_barrier_active||v.weapon.fire_cooldown_ticks>0) continue; common::Vec2 facing{std::cos(v.aim_angle),std::sin(v.aim_angle)}; float best=common::kVanguardWarhammerRange*common::kVanguardWarhammerRange; int best_slot=-1; for(std::uint32_t j=0;j<kAgentsPerMatch;++j){const HeroState& t=ctx.state.heroes[j]; if(!t.present||!t.alive||t.team==v.team) continue; common::Vec2 to{t.position.x-v.position.x,t.position.y-v.position.y}; float ds=to.x*to.x+to.y*to.y; if(ds<=1e-6F||ds>best) continue; float inv=1.0F/std::sqrt(ds); if(segment_blocked_by_cover(v.position,t.position,ctx.config)) continue; float dot=(to.x*facing.x+to.y*facing.y)*inv; if(dot<common::kVanguardWarhammerHalfAngleCos) continue; best=ds; best_slot=(int)j;} v.weapon.fire_cooldown_ticks=common::kVanguardWarhammerCooldownTicks; if(best_slot>=0){buf[i].attacker_id=v.id;buf[i].victim_slot=(std::uint32_t)best_slot;buf[i].damage_centi_hp=(std::uint32_t)common::kVanguardWarhammerDamageCentiHp;has_damage[i]=true;}} }

}  // namespace
