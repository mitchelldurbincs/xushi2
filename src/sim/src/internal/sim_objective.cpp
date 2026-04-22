#include "sim_objective.h"

#include <array>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>

namespace xushi2::sim::internal {

// --- Arena / objective geometry helpers. ---

static common::Vec2 arena_center(const MapBounds& map) {
    return common::Vec2{0.5F * (map.min_x + map.max_x), 0.5F * (map.min_y + map.max_y)};
}

static bool inside_objective(const common::Vec2& pos, const MapBounds& map) {
    const common::Vec2 c = arena_center(map);
    const float dx = pos.x - c.x;
    const float dy = pos.y - c.y;
    return (dx * dx + dy * dy) <= (common::kObjectiveRadius * common::kObjectiveRadius);
}

// --- Tick-pipeline step 15: objective state machine (5-case). ---

void objective_tick_update(ObjectiveState& obj,
                           const std::array<HeroState, kAgentsPerMatch>& heroes,
                           common::Tick now, const MapBounds& map) {
    // Initial lock: no state changes during the first kObjectiveLockTicks
    // ticks. The `now + 1` comparison is because state.tick is incremented
    // AFTER this function runs — callers observe tick N's state after
    // calling sim.step() N+1 times. Unlock during tick kObjectiveLockTicks-1
    // processing so "15 seconds of lock" is observable after exactly
    // kObjectiveLockTicks sim.step() calls.
    if (now + 1 < common::kObjectiveLockTicks) {
        obj.unlocked = false;
        return;
    }
    obj.unlocked = true;

    // Living heroes inside the objective circle, by team.
    bool present_a = false;
    bool present_b = false;
    for (std::uint32_t i = 0; i < kAgentsPerMatch; ++i) {
        const HeroState& h = heroes[i];
        if (!h.present || !h.alive) {
            continue;
        }
        if (!inside_objective(h.position, map)) {
            continue;
        }
        if (h.team == common::Team::A) {
            present_a = true;
        } else if (h.team == common::Team::B) {
            present_b = true;
        }
    }

    const std::uint32_t a_before = obj.team_a_score_ticks;
    const std::uint32_t b_before = obj.team_b_score_ticks;

    if (present_a && present_b) {
        // Case 1: contested — freeze. No changes.
    } else if (!present_a && !present_b) {
        // Case 2: empty — decay capture progress.
        if (obj.cap_progress_ticks > 0) {
            obj.cap_progress_ticks -= 1;
            if (obj.cap_progress_ticks == 0) {
                obj.cap_team = common::Team::Neutral;
            }
        }
    } else {
        const common::Team present_team = present_a ? common::Team::A : common::Team::B;
        const common::Team opp = (present_team == common::Team::A) ? common::Team::B : common::Team::A;
        if (obj.owner == present_team) {
            // Case 3: uncontested by owner — score.
            if (present_team == common::Team::A) {
                obj.team_a_score_ticks += 1;
            } else {
                obj.team_b_score_ticks += 1;
            }
            // Clean up any stale opposing cap progress.
            if (obj.cap_team == opp) {
                if (obj.cap_progress_ticks > 0) {
                    obj.cap_progress_ticks -= 1;
                }
                if (obj.cap_progress_ticks == 0) {
                    obj.cap_team = common::Team::Neutral;
                }
            } else {
                obj.cap_progress_ticks = 0;
                obj.cap_team = common::Team::Neutral;
            }
        } else {
            // Case 4: uncontested by non-owner — cap.
            if (obj.cap_team == present_team) {
                if (obj.cap_progress_ticks < common::kCaptureTicks) {
                    obj.cap_progress_ticks += 1;
                    if (obj.cap_progress_ticks == common::kCaptureTicks) {
                        obj.owner = present_team;
                        obj.cap_progress_ticks = 0;
                        obj.cap_team = common::Team::Neutral;
                    }
                }
            } else if (obj.cap_team == opp) {
                // Stale other-team progress — reset first.
                obj.cap_progress_ticks = 0;
                obj.cap_team = present_team;
            } else {
                obj.cap_team = present_team;
                obj.cap_progress_ticks = 0;
            }
        }
    }

    X2_INVARIANT(obj.team_a_score_ticks >= a_before, common::ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_b_score_ticks >= b_before, common::ErrorCode::CorruptState);
    X2_INVARIANT(obj.cap_progress_ticks <= common::kCaptureTicks, common::ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_a_score_ticks <= common::kWinTicks, common::ErrorCode::CorruptState);
    X2_INVARIANT(obj.team_b_score_ticks <= common::kWinTicks, common::ErrorCode::CorruptState);
}

}  // namespace xushi2::sim::internal
