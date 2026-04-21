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

TEST(ObsDims, CriticPhase1DimIs45) {
    // Actor 31 + privileged 14 (world-frame positions/velocities, raw
    // tick counters, seed bits). Must equal
    // python/xushi2/obs_manifest.py::CRITIC_PHASE1_DIM.
    EXPECT_EQ(xushi2::sim::kCriticObsPhase1Dim, 45U);
}

TEST(ObsDims, CriticIsAtLeastAsWideAsActor) {
    EXPECT_GE(xushi2::sim::kCriticObsPhase1Dim, xushi2::sim::kActorObsPhase1Dim);
}
