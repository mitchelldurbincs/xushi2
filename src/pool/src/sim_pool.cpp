#include <xushi2/pool/sim_pool.h>

#include <cmath>
#include <cstring>

#include <xushi2/common/assert.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/types.h>

namespace xushi2::pool {

namespace {

constexpr std::uint32_t kAgents =
    static_cast<std::uint32_t>(sim::kAgentsPerMatch);
constexpr std::uint32_t kEntityStride = kAgents * sim::kEntityGridObsDim;
constexpr std::uint32_t kCriticStride = 2U * sim::kCriticObsDim;
// Mirrors phase4_mappo.py _AIM_DELTA_LIMIT (pi/4), applied in double like
// the Python conversion so handicap clamping is bit-identical.
constexpr double kAimDeltaLimit = 3.14159265358979323846 / 4.0;

float clip(float v, float lo, float hi) noexcept {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

// Mirror of Phase4MappoEnv._action_to_cpp_for_team: clip controls, mirror
// Team B movement back to the world frame, scale aim to +/- pi/4, threshold
// the binary controls at 0.5.
common::Action policy_action(const float* controls, bool team_b) noexcept {
    common::Action act{};
    const float move_sign = team_b ? -1.0F : 1.0F;
    act.move_x = move_sign * clip(controls[0], -1.0F, 1.0F);
    act.move_y = move_sign * clip(controls[1], -1.0F, 1.0F);
    act.aim_delta = static_cast<float>(
        static_cast<double>(clip(controls[2], -1.0F, 1.0F)) * kAimDeltaLimit);
    act.primary_fire = clip(controls[3], 0.0F, 1.0F) >= 0.5F;
    act.ability_1 = clip(controls[4], 0.0F, 1.0F) >= 0.5F;
    act.ability_2 = clip(controls[5], 0.0F, 1.0F) >= 0.5F;
    return act;
}

}  // namespace

SimPool::SimPool(std::uint32_t num_envs,
                 const sim::MatchConfig& base_cfg,
                 const sim::ObsConfig& obs_cfg) {
    X2_REQUIRE(num_envs > 0, common::ErrorCode::CorruptState);
    X2_REQUIRE(base_cfg.team_size == 3U, common::ErrorCode::CorruptState);
    envs_.reserve(num_envs);
    for (std::uint32_t i = 0; i < num_envs; ++i) {
        EnvState& env = envs_.emplace_back(obs_cfg);
        env.cfg = base_cfg;
        env.modes.fill(SlotMode::Policy);
        env.obs_enabled.fill(true);
        env.sim = std::make_unique<sim::Sim>(env.cfg);
    }
}

void SimPool::set_slot_scripted(std::uint32_t env, std::uint32_t slot,
                                const std::string& bot_name) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    X2_REQUIRE(slot < kAgents, common::ErrorCode::CorruptState);
    EnvState& e = envs_[env];
    // Recreate only when the bot actually changes, so bot-internal memory
    // (none today, but the IBot contract allows it) survives episodes the
    // way a persistent legacy env's bot would.
    if (!e.bots[slot] || e.bots[slot]->name() != bot_name) {
        e.bots[slot] = bots::make_bot_by_name(bot_name);
    }
    e.modes[slot] = SlotMode::Scripted;
}

void SimPool::set_slot_policy(std::uint32_t env, std::uint32_t slot) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    X2_REQUIRE(slot < kAgents, common::ErrorCode::CorruptState);
    envs_[env].modes[slot] = SlotMode::Policy;
}

void SimPool::set_obs_slot(std::uint32_t env, std::uint32_t slot,
                           bool enabled) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    X2_REQUIRE(slot < kAgents, common::ErrorCode::CorruptState);
    envs_[env].obs_enabled[slot] = enabled;
}

void SimPool::set_opponent_handicap(const std::string& bot,
                                    float aim_noise_radians,
                                    std::uint32_t fire_cadence_ticks) {
    X2_REQUIRE(aim_noise_radians >= 0.0F, common::ErrorCode::CorruptState);
    X2_REQUIRE(fire_cadence_ticks >= 1U, common::ErrorCode::CorruptState);
    handicap_ = Handicap{bot, aim_noise_radians, fire_cadence_ticks};
}

void SimPool::set_objective_timing_ticks(std::uint32_t unlock_ticks,
                                         std::uint32_t capture_ticks) {
    for (EnvState& env : envs_) {
        env.cfg.objective_unlock_ticks = unlock_ticks;
        env.cfg.objective_capture_ticks = capture_ticks;
        env.sim->set_objective_timing_ticks(unlock_ticks, capture_ticks);
    }
}

void SimPool::set_respawn_ticks(std::uint32_t respawn_ticks) {
    X2_REQUIRE(respawn_ticks > 0U, common::ErrorCode::CorruptState);
    for (EnvState& env : envs_) {
        env.cfg.mechanics.respawn_ticks = respawn_ticks;
    }
}

void SimPool::set_env_config(std::uint32_t env, const sim::MatchConfig& cfg) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    X2_REQUIRE(cfg.team_size == 3U, common::ErrorCode::CorruptState);
    envs_[env].cfg = cfg;
}

void SimPool::reset_env(std::uint32_t env, std::uint64_t seed) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    EnvState& e = envs_[env];
    e.cfg.seed = seed;
    e.sim = std::make_unique<sim::Sim>(e.cfg);
    e.obs_engine.reset();
}

void SimPool::write_outputs(EnvState& env, float* entity_obs,
                            float* critic_obs, float* features) {
    if (entity_obs != nullptr) {
        for (std::uint32_t slot = 0; slot < kAgents; ++slot) {
            float* row = entity_obs +
                         (static_cast<std::size_t>(slot) *
                          sim::kEntityGridObsDim);
            if (env.obs_enabled[slot]) {
                env.obs_engine.build_entity_obs(*env.sim, slot, row,
                                                sim::kEntityGridObsDim);
            } else {
                std::memset(row, 0, sizeof(float) * sim::kEntityGridObsDim);
            }
        }
    }
    if (critic_obs != nullptr) {
        sim::build_critic_obs(*env.sim, common::Team::A, critic_obs,
                              sim::kCriticObsDim);
        sim::build_critic_obs(*env.sim, common::Team::B,
                              critic_obs + sim::kCriticObsDim,
                              sim::kCriticObsDim);
    }
    if (features != nullptr) {
        sim::write_reward_features(*env.sim, features, sim::kRewardFeatureDim);
    }
}

void SimPool::env_outputs(std::uint32_t env,
                          float* entity_obs, std::uint32_t entity_capacity,
                          float* critic_obs, std::uint32_t critic_capacity,
                          float* features, std::uint32_t features_capacity) {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    X2_REQUIRE(entity_obs == nullptr || entity_capacity >= kEntityStride,
               common::ErrorCode::CapacityExceeded);
    X2_REQUIRE(critic_obs == nullptr || critic_capacity >= kCriticStride,
               common::ErrorCode::CapacityExceeded);
    X2_REQUIRE(features == nullptr || features_capacity >= sim::kRewardFeatureDim,
               common::ErrorCode::CapacityExceeded);
    write_outputs(envs_[env], entity_obs, critic_obs, features);
}

common::Action SimPool::scripted_action(EnvState& env,
                                        std::uint32_t slot) const {
    bots::IBot* bot = env.bots[slot].get();
    X2_REQUIRE(bot != nullptr, common::ErrorCode::CorruptState);
    common::Action act = bot->decide(env.sim->state(), env.sim->config(),
                                     static_cast<int>(slot));
    if (handicap_ && handicap_->bot == bot->name()) {
        const auto tick = static_cast<double>(env.sim->state().tick);
        if (handicap_->aim_noise_radians > 0.0F) {
            // Deterministic per-(tick, slot) unit noise — bit-identical to
            // Phase4MappoEnv._apply_opponent_handicap (double math, same
            // constants, same clamp).
            const double raw =
                std::sin(tick * 12.9898 + static_cast<double>(slot) * 78.233) *
                43758.5453;
            const double unit = ((raw - std::floor(raw)) * 2.0) - 1.0;
            const double noisy =
                static_cast<double>(act.aim_delta) +
                (unit * static_cast<double>(handicap_->aim_noise_radians));
            act.aim_delta = static_cast<float>(
                std::max(-kAimDeltaLimit, std::min(kAimDeltaLimit, noisy)));
        }
        if (handicap_->fire_cadence_ticks > 1U) {
            act.primary_fire =
                act.primary_fire &&
                (env.sim->state().tick % handicap_->fire_cadence_ticks == 0);
        }
    }
    return act;
}

void SimPool::step_env(std::uint32_t env_idx, const float* actions,
                       float* out_entity_obs, float* out_critic_obs,
                       float* out_features, std::uint8_t* out_terminated,
                       std::uint8_t* out_truncated) {
    EnvState& env = envs_[env_idx];
    X2_REQUIRE(!env.sim->episode_over(), common::ErrorCode::CorruptState);

    std::array<common::Action, sim::kAgentsPerMatch> acts{};
    const float* env_actions =
        actions + (static_cast<std::size_t>(env_idx) * kAgents * kActionDim);
    for (std::uint32_t slot = 0; slot < kAgents; ++slot) {
        if (env.modes[slot] == SlotMode::Scripted) {
            acts[slot] = scripted_action(env, slot);
        } else {
            acts[slot] = policy_action(
                env_actions + (static_cast<std::size_t>(slot) * kActionDim),
                /*team_b=*/slot >= kAgents / 2);
        }
    }
    env.sim->step_decision(acts);

    const bool terminated = env.sim->score_threshold_reached();
    const bool truncated = env.sim->episode_over() && !terminated;
    out_terminated[env_idx] = terminated ? 1U : 0U;
    out_truncated[env_idx] = truncated ? 1U : 0U;

    write_outputs(env,
                  out_entity_obs + (static_cast<std::size_t>(env_idx) *
                                    kEntityStride),
                  out_critic_obs + (static_cast<std::size_t>(env_idx) *
                                    kCriticStride),
                  out_features + (static_cast<std::size_t>(env_idx) *
                                  sim::kRewardFeatureDim));
}

void SimPool::step(const float* actions,
                   float* out_entity_obs,
                   float* out_critic_obs,
                   float* out_features,
                   std::uint8_t* out_terminated,
                   std::uint8_t* out_truncated) {
    X2_REQUIRE(actions != nullptr && out_entity_obs != nullptr &&
                   out_critic_obs != nullptr && out_features != nullptr &&
                   out_terminated != nullptr && out_truncated != nullptr,
               common::ErrorCode::CorruptState);
    // Serial in Phase 4. Envs are independent and write disjoint output
    // slices; Phase 5 parallelizes this loop with a thread pool.
    for (std::uint32_t i = 0; i < envs_.size(); ++i) {
        step_env(i, actions, out_entity_obs, out_critic_obs, out_features,
                 out_terminated, out_truncated);
    }
}

bool SimPool::env_episode_over(std::uint32_t env) const {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    return envs_[env].sim->episode_over();
}

std::uint64_t SimPool::env_state_hash(std::uint32_t env) const {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    return envs_[env].sim->state_hash();
}

std::uint64_t SimPool::env_obs_state_hash(std::uint32_t env) const {
    X2_REQUIRE(env < envs_.size(), common::ErrorCode::CorruptState);
    return envs_[env].obs_engine.obs_state_hash();
}

}  // namespace xushi2::pool
