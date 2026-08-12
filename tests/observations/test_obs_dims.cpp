#include <gtest/gtest.h>

#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/reward_features.h>

// The C++ obs dim constants must match python/xushi2/obs_manifest.py. This
// test exists so that a drift between the two surfaces is a build-time /
// CI-time failure, not a training-time mystery.

TEST(ObsDims, ActorPhase1DimIs31) {
    // See observation_spec.md §Phase 1. Must also equal the sum of widths
    // in python/xushi2/obs_manifest.py::ACTOR_PHASE1_FIELDS.
    EXPECT_EQ(xushi2::sim::kActorObsPhase1Dim, 31U);
}

TEST(ObsDims, CriticDimIs135) {
    // 3 own-team actor mirrors (3*31=93) + 3 enemy world blocks (3*12=36)
    // + 4 raw objective counters + 2 seed bits = 135. Must equal
    // python/xushi2/obs_manifest.py::CRITIC_DIM.
    EXPECT_EQ(xushi2::sim::kCriticObsDim, 135U);
}

TEST(ObsDims, CriticIsAtLeastAsWideAsActor) {
    EXPECT_GE(xushi2::sim::kCriticObsDim, xushi2::sim::kActorObsPhase1Dim);
}

TEST(ObsDims, EntityGridObsLayoutMatchesPythonAdapter) {
    // Must equal python/xushi2/multi_enemy_obs.py: ENTITY_TOKEN_DIM,
    // MULTI_ENEMY_TOKEN_COUNT, GRID_CHANNELS, GRID_SIZE, and
    // MULTI_ENEMY_ENTITY_GRID_OBS_DIM. The Python lockstep half is
    // python/tests/test_obs_manifest.py::test_entity_layout_matches_native.
    EXPECT_EQ(xushi2::sim::kEntityTokenDim, 18U);
    EXPECT_EQ(xushi2::sim::kEntityTokenCount, 5U);
    EXPECT_EQ(xushi2::sim::kEntityGridChannels, 3U);
    EXPECT_EQ(xushi2::sim::kEntityGridSize, 32U);
    EXPECT_EQ(xushi2::sim::kEntityGridObsDim, 3167U);
}

TEST(ObsDims, RewardFeatureDimIs48) {
    // Must equal python/xushi2/obs_manifest.py::REWARD_FEATURE_DIM. Python
    // half: python/tests/test_obs_manifest.py::test_reward_feature_layout_matches_native.
    EXPECT_EQ(xushi2::sim::kRewardFeatureDim, 48U);
    EXPECT_EQ(xushi2::sim::reward_features::kDistToCenterBySlot + 6U,
              xushi2::sim::kRewardFeatureDim);
}
