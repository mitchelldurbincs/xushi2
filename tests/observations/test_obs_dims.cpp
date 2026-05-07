#include <gtest/gtest.h>

#include <xushi2/sim/obs.h>

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
