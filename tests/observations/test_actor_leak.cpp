#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>

#include <xushi2/common/limits.hpp>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_utils.h>
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

// --- Fog-enabled leak coverage ---
//
// The tests above run without fog, where hidden-enemy leaks are vacuous by
// construction. These enable fog and place cover between the teams, which is
// the configuration the whole separation contract exists for.
//
// MatchConfig::fog_of_war_enabled defaults to TRUE, so any config that
// simply omits it lands here -- these are not hypothetical futures.

namespace {

// 3v3 arena with one off-centre cover disc.
//
// Two constraints shape this setup, both learned the hard way:
//
//  - The disc must not overlap the objective (centre, radius 3). A disc that
//    covers the objective pushes heroes back out of it via
//    resolve_cover_overlap, so nobody can ever stand on the point.
//  - The disc sits at (20, 15), off the straight line from slot 1's spawn to
//    the objective, so the teammate driving the `contested` precondition has
//    an unobstructed path.
//
// It does block the diagonal from slot 0's spawn (17.5, 5) to the far-side
// slots, which is exactly what the test needs.
MatchConfig fog_test_config() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xD1CEDA7AULL;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = true;
    cfg.team_size = 3;

    xushi2::sim::CoverCircle cover{};
    cover.center = xushi2::common::Vec2{20.0F, 15.0F};
    cover.radius = 4.0F;
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0] = cover;
    return cfg;
}

}  // namespace

TEST(ActorLeak, ContestedFlagDoesNotLeakHiddenEnemyOnPoint) {
    // `contested` used to be computed by scanning every hero's position
    // directly, bypassing the fog filter that guards every other enemy field.
    // That handed the actor a free "an enemy is standing on the objective"
    // bit through cover.
    //
    // Isolating that bit takes some care, because the obvious differential
    // does not work: an enemy standing on the objective also drives the
    // objective state machine, and cap_team / cap_progress are legitimately
    // public fields in the actor manifest. Two properties make the test
    // clean:
    //
    //  - Everything happens inside the objective lock window
    //    (kObjectiveLockTicks = 450). objective_tick_update returns early
    //    while locked, so no objective state diverges between the two sims.
    //  - The varied enemy is slot 4, which is NOT slot 0's counterpart. In
    //    3v3, visible_enemy_1v1 only ever queries viewer_slot +/- 3, so slot
    //    4 never reaches slot 0's enemy_* fields by any route. `contested`
    //    was the only path.
    //
    // The `contested` precondition (own team on the point) is satisfied by
    // teammate slot 1, whose position is not part of slot 0's observation.
    Sim sim_enemy_away(fog_test_config());
    Sim sim_enemy_on_point(fog_test_config());

    std::array<Action, kAgentsPerMatch> ally_only{};
    std::array<Action, kAgentsPerMatch> ally_and_enemy{};
    // Slot 1 (Team A, spawns at x=25, y=5) walks up onto the objective in
    // BOTH sims. Slot 4 (Team B, x=25, y=45) walks down onto it in one only.
    ally_only[1].move_y = 1.0F;
    ally_and_enemy[1].move_y = 1.0F;
    ally_and_enemy[4].move_y = -1.0F;

    const auto& map = sim_enemy_away.config().map;
    int steps = 0;
    for (; steps < 60; ++steps) {
        sim_enemy_away.step_decision(ally_only);
        sim_enemy_on_point.step_decision(ally_and_enemy);
        if (xushi2::sim::obs_utils::position_on_objective(
                sim_enemy_on_point.state().heroes[4].position, map)) {
            break;
        }
    }

    // Preconditions.
    ASSERT_LT(sim_enemy_on_point.state().tick, xushi2::common::kObjectiveLockTicks)
        << "test precondition: must stay inside the objective lock window so "
           "no objective state diverges";
    ASSERT_TRUE(xushi2::sim::obs_utils::position_on_objective(
        sim_enemy_on_point.state().heroes[1].position, map))
        << "test precondition: an ALLY must be on the point so `contested` "
           "can be true at all";
    ASSERT_TRUE(xushi2::sim::obs_utils::position_on_objective(
        sim_enemy_on_point.state().heroes[4].position, map))
        << "test precondition: the hidden enemy must reach the objective";
    ASSERT_FALSE(xushi2::sim::obs_utils::position_on_objective(
        sim_enemy_away.state().heroes[4].position, map))
        << "test precondition: the enemy must NOT be on the point in the "
           "reference sim";
    ASSERT_FALSE(sim_enemy_on_point.line_of_sight(0, 4))
        << "test precondition: cover must hide slot 4 from slot 0";

    auto a = build_team_a_obs(sim_enemy_away);
    auto b = build_team_a_obs(sim_enemy_on_point);
    EXPECT_TRUE(arrays_equal(a, b))
        << "actor obs leaked a hidden enemy's presence on the objective via "
           "the `contested` field";
}
