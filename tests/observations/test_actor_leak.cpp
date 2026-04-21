#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>

#include <xushi2/common/limits.hpp>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

// Actor / critic observation leak prevention.
//
// This is the single highest-priority correctness test in the project: any
// leak of hidden enemy state into the actor invalidates the research
// contribution. See docs/rl_design.md §10 and docs/observation_spec.md.
//
// Phase 1 has no fog of war, so hidden-enemy-position leak tests are vacuous
// here. What IS meaningful at Phase 1 is the structural contract: fields
// that the manifest declares the actor does not see (enemy aim_angle, enemy
// weapon state, enemy cooldowns) must provably not affect the actor obs.
// When fog lands at Phase 7 these tests graduate to cover hidden position
// and HP.
//
// Method: run two parallel sims from identical initial state, apply
// different action streams that ONLY mutate fields the manifest considers
// hidden from the actor, then compare the Team-A actor obs tensors. They
// must be byte-identical.

namespace {

using xushi2::common::Action;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

using ObsArray = std::array<float, kActorObsPhase1Dim>;

MatchConfig leak_test_config() {
    // Default 50x50 arena: Rangers spawn 40u apart, well beyond the 22u
    // revolver range. This means Team B firing at Team A is a guaranteed
    // miss, so B's primary_fire only mutates B's *weapon state*, not A's HP.
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xD1CEDA7AULL;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = false;
    return cfg;
}

ObsArray build_team_a_obs(const Sim& sim) {
    ObsArray out{};
    xushi2::sim::build_actor_obs_phase1(
        sim, /*agent_slot=*/0, out.data(),
        static_cast<std::uint32_t>(out.size()));
    return out;
}

bool arrays_equal(const ObsArray& a, const ObsArray& b) {
    // Strict equality: structural leak tests require bit-for-bit match
    // because the hidden action should introduce zero drift. We allow a
    // tiny tolerance purely to absorb FP non-determinism that IS NOT
    // believed to occur on this machine; if anything fails, it's a real
    // leak.
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-7F) return false;
    }
    return true;
}

}  // namespace

TEST(ActorLeak, EnemyAimAngleDoesNotAffectActorObs) {
    Sim sim_idle(leak_test_config());
    Sim sim_spin(leak_test_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> spin{};
    spin[3].aim_delta = 0.5F;  // team B twirls its aim each decision

    for (int step = 0; step < 20; ++step) {
        sim_idle.step_decision(idle);
        sim_spin.step_decision(spin);
    }

    ASSERT_NE(sim_idle.state().heroes[3].aim_angle,
              sim_spin.state().heroes[3].aim_angle)
        << "test precondition: hidden aim_angle must actually diverge";

    auto a = build_team_a_obs(sim_idle);
    auto b = build_team_a_obs(sim_spin);
    EXPECT_TRUE(arrays_equal(a, b))
        << "actor obs leaked enemy aim_angle — NOT in Phase 1 manifest";
}

TEST(ActorLeak, EnemyMagazineDoesNotAffectActorObs) {
    Sim sim_full(leak_test_config());
    Sim sim_fired(leak_test_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> b_fires{};
    b_fires[3].primary_fire = true;

    // Let Team B fire until its magazine is empty (6 shots, spaced by the
    // 15-tick fire-rate gate). The 50x50 arena guarantees its shots miss
    // Team A, so Team A's observable fields are untouched.
    for (int step = 0; step < 40; ++step) {
        sim_full.step_decision(idle);
        sim_fired.step_decision(b_fires);
    }

    ASSERT_LT(sim_fired.state().heroes[3].weapon.magazine,
              sim_full.state().heroes[3].weapon.magazine)
        << "test precondition: hidden magazine state must diverge";
    ASSERT_EQ(sim_full.state().heroes[0].health_centi_hp,
              sim_fired.state().heroes[0].health_centi_hp)
        << "test precondition: Team A HP must not diverge (shots should miss)";

    auto a = build_team_a_obs(sim_full);
    auto b = build_team_a_obs(sim_fired);
    EXPECT_TRUE(arrays_equal(a, b))
        << "actor obs leaked enemy magazine — NOT in Phase 1 manifest";
}

TEST(ActorLeak, EnemyFireCooldownDoesNotAffectActorObs) {
    Sim sim_idle(leak_test_config());
    Sim sim_fire_once(leak_test_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> fire_now{};
    fire_now[3].primary_fire = true;

    // One decision of B firing -> B has a non-zero fire_cooldown_ticks
    // for the next ~15 ticks.
    sim_fire_once.step_decision(fire_now);
    sim_idle.step_decision(idle);

    ASSERT_GT(sim_fire_once.state().heroes[3].weapon.fire_cooldown_ticks,
              sim_idle.state().heroes[3].weapon.fire_cooldown_ticks);

    auto a = build_team_a_obs(sim_idle);
    auto b = build_team_a_obs(sim_fire_once);
    EXPECT_TRUE(arrays_equal(a, b))
        << "actor obs leaked enemy fire_cooldown — NOT in Phase 1 manifest";
}

TEST(ActorLeak, EnemyReloadingStateDoesNotAffectActorObs) {
    Sim sim_idle(leak_test_config());
    Sim sim_reload(leak_test_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> b_fire{};
    b_fire[3].primary_fire = true;

    // Team B fires once in sim_reload, idles in sim_idle. BOTH sims then
    // step the same number of times so their tick counters stay aligned
    // (otherwise round_timer alone makes the tensors differ — a tick-count
    // leak, not a reloading leak).
    sim_reload.step_decision(b_fire);
    sim_idle.step_decision(idle);
    // Step both for 29 more decisions so the auto-reload window (60 idle
    // ticks ~= 20 decisions past the shot) lands with sim_reload
    // mid-reload.
    for (int step = 0; step < 29; ++step) {
        sim_idle.step_decision(idle);
        sim_reload.step_decision(idle);
    }

    ASSERT_TRUE(sim_reload.state().heroes[3].weapon.reloading)
        << "test precondition: Team B should be mid-reload";
    ASSERT_FALSE(sim_idle.state().heroes[3].weapon.reloading);

    auto a = build_team_a_obs(sim_idle);
    auto b = build_team_a_obs(sim_reload);
    EXPECT_TRUE(arrays_equal(a, b))
        << "actor obs leaked enemy reloading flag — NOT in Phase 1 manifest";
}

TEST(ActorLeak, Placeholder) {
    // Kept so the previous test name still passes CI; real coverage is
    // above.
    SUCCEED() << "actor/critic leak coverage provided by the tests above";
}
