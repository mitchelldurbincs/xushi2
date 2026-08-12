#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <xushi2/bots/bot.h>
#include <xushi2/common/limits.hpp>
#include <xushi2/pool/sim_pool.h>
#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/reward_features.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

// SimPool semantics: a pool env must behave exactly like a hand-driven
// Sim + ObservationEngine pair, persistent bots must match the legacy
// fresh-bot-per-call pattern, the handicap transform must match the Python
// formula, and the reward feature block must mirror direct state reads.

namespace {

using xushi2::common::Action;
using xushi2::pool::SimPool;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::kCriticObsDim;
using xushi2::sim::kEntityGridObsDim;
using xushi2::sim::kRewardFeatureDim;
using xushi2::sim::MatchConfig;
using xushi2::sim::ObsConfig;
using xushi2::sim::Sim;

constexpr std::uint32_t kActionDim = SimPool::kActionDim;
constexpr double kAimDeltaLimit = 3.14159265358979323846 / 4.0;

MatchConfig pool_test_config(int round_seconds = 30) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0x9001;
    cfg.round_length_seconds = round_seconds;
    cfg.fog_of_war_enabled = true;
    cfg.team_size = 3;
    xushi2::sim::CoverCircle cover{};
    cover.center = xushi2::common::Vec2{20.0F, 15.0F};
    cover.radius = 4.0F;
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0] = cover;
    return cfg;
}

ObsConfig pool_obs_config() {
    ObsConfig cfg{};
    cfg.fog_mode = xushi2::sim::FogMode::TeamShared;
    cfg.visible_radius = 0.65F;
    cfg.last_seen_enabled = true;
    return cfg;
}

// The same canonicalization the pool applies to Policy controls — mirrors
// Phase4MappoEnv._action_to_cpp_for_team.
Action reference_policy_action(const float* c, bool team_b) {
    auto clipf = [](float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    Action a{};
    const float sign = team_b ? -1.0F : 1.0F;
    a.move_x = sign * clipf(c[0], -1.0F, 1.0F);
    a.move_y = sign * clipf(c[1], -1.0F, 1.0F);
    a.aim_delta = static_cast<float>(
        static_cast<double>(clipf(c[2], -1.0F, 1.0F)) * kAimDeltaLimit);
    a.primary_fire = clipf(c[3], 0.0F, 1.0F) >= 0.5F;
    a.ability_1 = clipf(c[4], 0.0F, 1.0F) >= 0.5F;
    a.ability_2 = clipf(c[5], 0.0F, 1.0F) >= 0.5F;
    return a;
}

struct StepBuffers {
    explicit StepBuffers(std::uint32_t n)
        : entity(static_cast<std::size_t>(n) * kAgentsPerMatch * kEntityGridObsDim),
          critic(static_cast<std::size_t>(n) * 2 * kCriticObsDim),
          features(static_cast<std::size_t>(n) * kRewardFeatureDim),
          terminated(n),
          truncated(n) {}
    std::vector<float> entity;
    std::vector<float> critic;
    std::vector<float> features;
    std::vector<std::uint8_t> terminated;
    std::vector<std::uint8_t> truncated;
};

}  // namespace

TEST(SimPool, PoolOfOneMatchesHandDrivenSim) {
    SimPool pool(1, pool_test_config(), pool_obs_config());

    Sim ref_sim(pool_test_config());
    xushi2::sim::ObservationEngine ref_engine(pool_obs_config());

    std::vector<float> controls(kAgentsPerMatch * kActionDim, 0.0F);
    StepBuffers buf(1);
    std::vector<float> ref_entity(kAgentsPerMatch * kEntityGridObsDim, 0.0F);
    std::vector<float> ref_critic(2 * kCriticObsDim, 0.0F);
    std::vector<float> ref_features(kRewardFeatureDim, 0.0F);

    for (int step = 0; step < 30; ++step) {
        for (std::uint32_t slot = 0; slot < kAgentsPerMatch; ++slot) {
            float* c = controls.data() + (slot * kActionDim);
            c[0] = 0.3F * static_cast<float>(slot % 2 == 0 ? 1 : -1);
            c[1] = 1.0F;  // both teams approach in team frame
            c[2] = 0.15F;
            c[3] = (step % 3 == 0) ? 1.0F : 0.0F;
        }
        pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
                  buf.features.data(), buf.terminated.data(),
                  buf.truncated.data());

        std::array<Action, kAgentsPerMatch> ref_actions{};
        for (std::uint32_t slot = 0; slot < kAgentsPerMatch; ++slot) {
            ref_actions[slot] = reference_policy_action(
                controls.data() + (slot * kActionDim), slot >= 3);
        }
        ref_sim.step_decision(ref_actions);

        ASSERT_EQ(pool.env_state_hash(0), ref_sim.state_hash())
            << "sim state diverged at step " << step;

        ref_engine.build_entity_obs_all(
            ref_sim, ref_entity.data(),
            static_cast<std::uint32_t>(ref_entity.size()));
        xushi2::sim::build_critic_obs(ref_sim, xushi2::common::Team::A,
                                      ref_critic.data(), kCriticObsDim);
        xushi2::sim::build_critic_obs(ref_sim, xushi2::common::Team::B,
                                      ref_critic.data() + kCriticObsDim,
                                      kCriticObsDim);
        xushi2::sim::write_reward_features(ref_sim, ref_features.data(),
                                           kRewardFeatureDim);
        for (std::size_t i = 0; i < ref_entity.size(); ++i) {
            ASSERT_EQ(buf.entity[i], ref_entity[i]) << "entity idx " << i;
        }
        for (std::size_t i = 0; i < ref_critic.size(); ++i) {
            ASSERT_EQ(buf.critic[i], ref_critic[i]) << "critic idx " << i;
        }
        for (std::size_t i = 0; i < ref_features.size(); ++i) {
            ASSERT_EQ(buf.features[i], ref_features[i]) << "feature idx " << i;
        }
        ASSERT_EQ(pool.env_obs_state_hash(0), ref_engine.obs_state_hash());
    }
}

TEST(SimPool, TruncationReportingAndResetEnv) {
    // 2-second round: 60 ticks / action_repeat 3 = 20 decisions to expiry.
    SimPool pool(2, pool_test_config(/*round_seconds=*/2), pool_obs_config());
    std::vector<float> controls(2 * kAgentsPerMatch * kActionDim, 0.0F);
    StepBuffers buf(2);

    int steps_to_done = 0;
    for (int step = 0; step < 40; ++step) {
        pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
                  buf.features.data(), buf.terminated.data(),
                  buf.truncated.data());
        ++steps_to_done;
        if (buf.truncated[0] != 0U || buf.terminated[0] != 0U) {
            break;
        }
    }
    EXPECT_EQ(buf.truncated[0], 1U) << "timer expiry must report truncated";
    EXPECT_EQ(buf.terminated[0], 0U);
    EXPECT_EQ(buf.truncated[1], 1U) << "identical envs finish together";
    EXPECT_TRUE(pool.env_episode_over(0));
    EXPECT_EQ(steps_to_done, 20);

    pool.reset_env(0, 4242);
    pool.reset_env(1, 4242);
    EXPECT_FALSE(pool.env_episode_over(0));
    // Fresh episodes step normally and identically (same seed).
    pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
              buf.features.data(), buf.terminated.data(),
              buf.truncated.data());
    EXPECT_EQ(buf.truncated[0], 0U);
    EXPECT_EQ(pool.env_state_hash(0), pool.env_state_hash(1));
}

TEST(SimPool, PersistentBotMatchesFreshBotPerCall) {
    // Legacy path constructed a fresh bot every decision
    // (scripted_bot_action); the pool keeps one alive. For the stateless
    // shipped bots the two must produce identical episodes.
    SimPool pool(1, pool_test_config(), pool_obs_config());
    for (std::uint32_t slot = 3; slot < 6; ++slot) {
        pool.set_slot_scripted(0, slot, "weak_basic_v2");
    }

    Sim ref_sim(pool_test_config());
    std::vector<float> controls(kAgentsPerMatch * kActionDim, 0.0F);
    StepBuffers buf(1);

    for (int step = 0; step < 40; ++step) {
        for (std::uint32_t slot = 0; slot < 3; ++slot) {
            controls[(slot * kActionDim) + 1] = 1.0F;
        }
        pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
                  buf.features.data(), buf.terminated.data(),
                  buf.truncated.data());

        std::array<Action, kAgentsPerMatch> ref_actions{};
        for (std::uint32_t slot = 0; slot < 3; ++slot) {
            ref_actions[slot] = reference_policy_action(
                controls.data() + (slot * kActionDim), false);
        }
        for (std::uint32_t slot = 3; slot < 6; ++slot) {
            auto fresh = xushi2::bots::make_bot_by_name("weak_basic_v2");
            ref_actions[slot] = fresh->decide(ref_sim.state(),
                                              ref_sim.config(),
                                              static_cast<int>(slot));
        }
        ref_sim.step_decision(ref_actions);
        ASSERT_EQ(pool.env_state_hash(0), ref_sim.state_hash())
            << "diverged at step " << step;
    }
}

TEST(SimPool, HandicapMatchesLegacyFormula) {
    SimPool pool(1, pool_test_config(), pool_obs_config());
    for (std::uint32_t slot = 3; slot < 6; ++slot) {
        pool.set_slot_scripted(0, slot, "weak_basic_v2");
    }
    pool.set_opponent_handicap("weak_basic_v2", 1.5F, 60);

    Sim ref_sim(pool_test_config());
    std::vector<float> controls(kAgentsPerMatch * kActionDim, 0.0F);
    StepBuffers buf(1);

    for (int step = 0; step < 40; ++step) {
        pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
                  buf.features.data(), buf.terminated.data(),
                  buf.truncated.data());

        std::array<Action, kAgentsPerMatch> ref_actions{};
        const auto tick = static_cast<double>(ref_sim.state().tick);
        for (std::uint32_t slot = 3; slot < 6; ++slot) {
            auto fresh = xushi2::bots::make_bot_by_name("weak_basic_v2");
            Action act = fresh->decide(ref_sim.state(), ref_sim.config(),
                                       static_cast<int>(slot));
            // Python: raw = sin(tick*12.9898 + slot*78.233) * 43758.5453;
            // unit = frac(raw)*2-1; clamp(aim + unit*noise, +/- pi/4);
            // fire &&= tick % cadence == 0.
            const double raw =
                std::sin(tick * 12.9898 + static_cast<double>(slot) * 78.233) *
                43758.5453;
            const double unit = ((raw - std::floor(raw)) * 2.0) - 1.0;
            const double noisy = static_cast<double>(act.aim_delta) +
                                 (unit * 1.5);
            act.aim_delta = static_cast<float>(
                std::max(-kAimDeltaLimit, std::min(kAimDeltaLimit, noisy)));
            act.primary_fire =
                act.primary_fire && (ref_sim.state().tick % 60 == 0);
            ref_actions[slot] = act;
        }
        ref_sim.step_decision(ref_actions);
        ASSERT_EQ(pool.env_state_hash(0), ref_sim.state_hash())
            << "handicap transform diverged at step " << step;
    }
}

TEST(SimPool, RewardFeaturesMirrorSimState) {
    SimPool pool(1, pool_test_config(), pool_obs_config());
    std::vector<float> controls(kAgentsPerMatch * kActionDim, 0.0F);
    for (std::uint32_t slot = 0; slot < kAgentsPerMatch; ++slot) {
        controls[(slot * kActionDim) + 1] = 1.0F;
        controls[(slot * kActionDim) + 3] = 1.0F;
    }
    StepBuffers buf(1);
    for (int step = 0; step < 25; ++step) {
        pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
                  buf.features.data(), buf.terminated.data(),
                  buf.truncated.data());
    }

    namespace rf = xushi2::sim::reward_features;
    Sim ref_sim(pool_test_config());
    std::array<Action, kAgentsPerMatch> ref_actions{};
    for (std::uint32_t slot = 0; slot < kAgentsPerMatch; ++slot) {
        ref_actions[slot] = reference_policy_action(
            controls.data() + (slot * kActionDim), slot >= 3);
    }
    for (int step = 0; step < 25; ++step) {
        ref_sim.step_decision(ref_actions);
    }
    const auto& s = ref_sim.state();
    EXPECT_EQ(buf.features[rf::kTick], static_cast<float>(s.tick));
    EXPECT_EQ(buf.features[rf::kTeamAScoreTicks],
              static_cast<float>(s.objective.team_a_score_ticks));
    EXPECT_EQ(buf.features[rf::kCapProgressTicks],
              static_cast<float>(s.objective.cap_progress_ticks));
    for (std::uint32_t slot = 0; slot < kAgentsPerMatch; ++slot) {
        EXPECT_EQ(buf.features[rf::kKillsBySlot + slot],
                  static_cast<float>(s.heroes[slot].kills));
        EXPECT_EQ(buf.features[rf::kAliveBySlot + slot],
                  s.heroes[slot].alive ? 1.0F : 0.0F);
    }
}

TEST(SimPool, DisabledObsSlotsAreZeroed) {
    SimPool pool(1, pool_test_config(), pool_obs_config());
    for (std::uint32_t slot = 3; slot < 6; ++slot) {
        pool.set_obs_slot(0, slot, false);
    }
    std::vector<float> controls(kAgentsPerMatch * kActionDim, 0.0F);
    StepBuffers buf(1);
    pool.step(controls.data(), buf.entity.data(), buf.critic.data(),
              buf.features.data(), buf.terminated.data(),
              buf.truncated.data());

    float sum_learner = 0.0F;
    float sum_opponent = 0.0F;
    for (std::uint32_t slot = 0; slot < 3; ++slot) {
        for (std::uint32_t i = 0; i < kEntityGridObsDim; ++i) {
            sum_learner +=
                std::fabs(buf.entity[(slot * kEntityGridObsDim) + i]);
        }
    }
    for (std::uint32_t slot = 3; slot < 6; ++slot) {
        for (std::uint32_t i = 0; i < kEntityGridObsDim; ++i) {
            sum_opponent +=
                std::fabs(buf.entity[(slot * kEntityGridObsDim) + i]);
        }
    }
    EXPECT_GT(sum_learner, 0.0F);
    EXPECT_EQ(sum_opponent, 0.0F);
}
