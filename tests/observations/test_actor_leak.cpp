#include <gtest/gtest.h>

#include <xushi2/sim/sim.h>

// Actor / critic observation leak prevention.
// This is the single highest-priority correctness test in the project:
// any leak of hidden enemy state into the actor invalidates the research
// contribution. See docs/rl_design.md §10 and docs/observation_spec.md.
//
// Phase 0: placeholder harness. Real tests to add once observation builders
// exist:
//
//   TEST: build_actor_obs is unchanged when a hidden enemy moves behind a
//         wall (enemy outside all per-agent LoS).
//   TEST: build_actor_obs changes only when an enemy becomes visible OR
//         triggers an event (fire / death / objective contest).
//   TEST: build_critic_state DOES see hidden enemies.
//   TEST: actor and critic observation builders are different functions
//         (no shared-code leak path).

TEST(ActorLeak, Placeholder) {
    SUCCEED() << "actor/critic leak tests land with the observation builder";
}
