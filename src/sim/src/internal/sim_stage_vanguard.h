#pragma once

#include "sim_tick_pipeline.h"

namespace xushi2::sim::internal {

void stage_abilities_vanguard_barrier(const TickContext& ctx);
void stage_abilities_vanguard_guard_step(const TickContext& ctx);
void stage_vanguard_warhammer(const TickContext& ctx, DamageBuffer& buf,
                              std::array<bool, kAgentsPerMatch>& has_damage);

}  // namespace xushi2::sim::internal
