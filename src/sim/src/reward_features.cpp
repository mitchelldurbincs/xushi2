#include <xushi2/sim/reward_features.h>

#include <cmath>
#include <cstdint>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/types.h>
#include <xushi2/sim/obs_utils.h>

namespace xushi2::sim {

namespace {

float clamp01(float v) noexcept {
    if (v < 0.0F) return 0.0F;
    if (v > 1.0F) return 1.0F;
    return v;
}

float team_sign_a(common::Team t) noexcept {
    if (t == common::Team::A) return 1.0F;
    if (t == common::Team::B) return -1.0F;
    return 0.0F;
}

}  // namespace

void write_reward_features(const Sim& sim,
                           float* out,
                           std::uint32_t capacity) noexcept {
    X2_REQUIRE(out != nullptr, common::ErrorCode::CorruptState);
    X2_REQUIRE(capacity >= kRewardFeatureDim, common::ErrorCode::CapacityExceeded);

    namespace rf = reward_features;
    const MatchState& s = sim.state();
    const MatchConfig& cfg = sim.config();

    out[rf::kTick] = static_cast<float>(s.tick);
    out[rf::kTeamAScoreTicks] = static_cast<float>(s.objective.team_a_score_ticks);
    out[rf::kTeamBScoreTicks] = static_cast<float>(s.objective.team_b_score_ticks);
    out[rf::kTeamAKills] = static_cast<float>(sim.team_a_kills());
    out[rf::kTeamBKills] = static_cast<float>(sim.team_b_kills());
    out[rf::kOwnerSignA] = team_sign_a(s.objective.owner);
    out[rf::kCapSignA] = team_sign_a(s.objective.cap_team);
    out[rf::kCapProgressFraction] =
        clamp01(static_cast<float>(s.objective.cap_progress_ticks) /
                static_cast<float>(cfg.objective_capture_ticks));
    out[rf::kCapProgressTicks] =
        static_cast<float>(s.objective.cap_progress_ticks);
    out[rf::kEpisodeOver] = sim.episode_over() ? 1.0F : 0.0F;
    out[rf::kScoreThresholdReached] =
        sim.score_threshold_reached() ? 1.0F : 0.0F;
    out[rf::kWinnerSign] = team_sign_a(sim.winner());

    for (std::uint32_t slot = 0;
         slot < static_cast<std::uint32_t>(kAgentsPerMatch); ++slot) {
        const HeroState& h = s.heroes[slot];
        out[rf::kKillsBySlot + slot] = static_cast<float>(h.kills);
        out[rf::kDeathsBySlot + slot] = static_cast<float>(h.deaths);
        out[rf::kDamageCentiBySlot + slot] =
            static_cast<float>(h.damage_dealt_centi_hp);
        const bool alive = h.present && h.alive;
        out[rf::kAliveBySlot + slot] = alive ? 1.0F : 0.0F;
        const bool on_point =
            alive && obs_utils::position_on_objective(h.position, cfg.map);
        out[rf::kOnPointBySlot + slot] = on_point ? 1.0F : 0.0F;
        const common::Vec2 pos_norm = obs_utils::normalize_position_to_map(
            obs_utils::mirror_position_for_team(h.position, h.team, cfg.map),
            cfg.map);
        out[rf::kDistToCenterBySlot + slot] = std::hypot(pos_norm.x, pos_norm.y);
    }
}

}  // namespace xushi2::sim
