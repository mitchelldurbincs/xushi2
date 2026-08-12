#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <xushi2/common/limits.hpp>
#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

// Counterfactual leak tests for the native entity observation builder.
// All of these run with fog ENABLED — unlike most of test_actor_leak.cpp,
// which predates native fog and is vacuous for position/HP.
//
// Method (same as test_actor_leak.cpp): two parallel sims from identical
// initial state, action streams that diverge ONLY in state hidden from the
// viewer under test, then byte-compare the viewer's entity obs.
//
// Geometry (50x50 map, team_size 3): A slots 0/1/2 at x=17.5/25/32.5, y=5;
// B slots 3/4/5 mirrored at y=45. The cover disc at (20,15) r=4 blocks
// slot 0's line to every far-side slot at spawn, but leaves slot 1's
// vertical line to slot 4 clear.

namespace {

using xushi2::common::Action;
using xushi2::sim::FogMode;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::kEntityGridObsDim;
using xushi2::sim::MatchConfig;
using xushi2::sim::ObsConfig;
using xushi2::sim::ObservationEngine;
using xushi2::sim::Sim;

using EntityObs = std::array<float, kEntityGridObsDim>;

MatchConfig fog_config() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xF06F06ULL;
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

ObsConfig per_agent_cfg(float radius = std::numeric_limits<float>::quiet_NaN(),
                        bool last_seen = false) {
    ObsConfig cfg{};
    cfg.fog_mode = FogMode::PerAgent;
    cfg.visible_radius = radius;
    cfg.last_seen_enabled = last_seen;
    return cfg;
}

EntityObs build_one(ObservationEngine& engine, const Sim& sim,
                    std::uint32_t viewer_slot) {
    EntityObs out{};
    engine.build_entity_obs(sim, viewer_slot, out.data(),
                            static_cast<std::uint32_t>(out.size()));
    return out;
}

bool obs_equal(const EntityObs& a, const EntityObs& b) {
    // Strict: hidden divergence must introduce zero drift. The tolerance
    // only absorbs FP non-determinism that is not believed to occur on one
    // machine; a failure is a real leak.
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-7F) return false;
    }
    return true;
}

}  // namespace

TEST(EntityObsLeak, HiddenEnemyMovementDoesNotAffectEntityObs) {
    // Slot 4 starts hidden from slot 0 (disc blocks the diagonal) and moves
    // straight down the x=25 line, which stays inside the disc's shadow for
    // slot 0 the whole way. Viewer slot 0's entity obs must not move.
    Sim sim_idle(fog_config());
    Sim sim_move(fog_config());
    ObservationEngine engine_idle(per_agent_cfg());
    ObservationEngine engine_move(per_agent_cfg());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> move{};
    move[4].move_y = -1.0F;

    for (int i = 0; i < 15; ++i) {
        sim_idle.step_decision(idle);
        sim_move.step_decision(move);
        ASSERT_FALSE(sim_idle.line_of_sight(0, 4));
        ASSERT_FALSE(sim_move.line_of_sight(0, 4))
            << "test precondition: slot 4 must stay hidden from slot 0 while "
               "moving (step " << i << ")";
    }
    ASSERT_NE(sim_idle.state().heroes[4].position.y,
              sim_move.state().heroes[4].position.y)
        << "test precondition: hidden position must actually diverge";

    const auto a = build_one(engine_idle, sim_idle, 0);
    const auto b = build_one(engine_move, sim_move, 0);
    EXPECT_TRUE(obs_equal(a, b))
        << "entity obs leaked a hidden enemy's position through fog";
}

TEST(EntityObsLeak, TeamSharedRevealsWhatPerAgentHides) {
    // The 7a/7b semantic test. Slot 4 spins its aim — a live token field.
    // Slot 0 has no LoS to slot 4, but teammate slot 1 does.
    //   PerAgent:  slot 0's obs must be byte-identical (nothing leaked).
    //   TeamShared: slot 0's obs MUST differ (the union channel is real) —
    //               the positive control proving the per-agent result is
    //               not vacuous.
    Sim sim_idle(fog_config());
    Sim sim_spin(fog_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> spin{};
    spin[4].aim_delta = 0.5F;

    for (int i = 0; i < 10; ++i) {
        sim_idle.step_decision(idle);
        sim_spin.step_decision(spin);
    }
    ASSERT_FALSE(sim_spin.line_of_sight(0, 4));
    ASSERT_TRUE(sim_spin.line_of_sight(1, 4));
    ASSERT_NE(sim_idle.state().heroes[4].aim_angle,
              sim_spin.state().heroes[4].aim_angle)
        << "test precondition: hidden aim must actually diverge";

    ObservationEngine pa_idle(per_agent_cfg());
    ObservationEngine pa_spin(per_agent_cfg());
    EXPECT_TRUE(obs_equal(build_one(pa_idle, sim_idle, 0),
                          build_one(pa_spin, sim_spin, 0)))
        << "per-agent fog leaked an enemy visible only to a teammate";

    ObsConfig shared{};
    shared.fog_mode = FogMode::TeamShared;
    ObservationEngine ts_idle(shared);
    ObservationEngine ts_spin(shared);
    EXPECT_FALSE(obs_equal(build_one(ts_idle, sim_idle, 0),
                           build_one(ts_spin, sim_spin, 0)))
        << "team-shared fog failed to reveal a teammate-visible enemy "
           "(positive control — the per-agent assertion above would be "
           "vacuous)";
}

TEST(EntityObsLeak, BeyondRadiusEnemyStateDoesNotAffectEntityObs) {
    // Slot 1 has clear LoS to slot 4 (vertical x=25 line), but slot 4 sits
    // at normalized distance 1.6 — beyond a 1.0 radius. Its aim/weapon
    // mutations must not reach slot 1's obs. With a 1.7 radius the same
    // divergence MUST show (positive control).
    Sim sim_idle(fog_config());
    Sim sim_busy(fog_config());

    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> busy{};
    busy[4].aim_delta = 0.4F;
    busy[4].primary_fire = true;  // 40u apart: guaranteed miss, weapon-state only

    for (int i = 0; i < 10; ++i) {
        sim_idle.step_decision(idle);
        sim_busy.step_decision(busy);
    }
    ASSERT_TRUE(sim_busy.line_of_sight(1, 4))
        << "test precondition: only the radius may hide slot 4 here";
    ASSERT_NE(sim_idle.state().heroes[4].aim_angle,
              sim_busy.state().heroes[4].aim_angle);
    ASSERT_EQ(sim_idle.state().heroes[1].health_centi_hp,
              sim_busy.state().heroes[1].health_centi_hp)
        << "test precondition: slot 1 must not take damage";

    ObservationEngine near_idle(per_agent_cfg(1.0F));
    ObservationEngine near_busy(per_agent_cfg(1.0F));
    EXPECT_TRUE(obs_equal(build_one(near_idle, sim_idle, 1),
                          build_one(near_busy, sim_busy, 1)))
        << "entity obs leaked an enemy beyond the visibility radius";

    ObservationEngine far_idle(per_agent_cfg(1.7F));
    ObservationEngine far_busy(per_agent_cfg(1.7F));
    EXPECT_FALSE(obs_equal(build_one(far_idle, sim_idle, 1),
                           build_one(far_busy, sim_busy, 1)))
        << "positive control: inside the radius the divergence must show";
}

TEST(EntityObsLeak, LastSeenIsNotACovertChannel) {
    // Slot 4 walks into the disc's shadow (identical in both sims), then
    // keeps moving in one sim only, staying hidden. The stale marker must
    // freeze at the last SEEN position: further hidden movement must not
    // update it, or last-seen memory becomes a tracker through walls.
    Sim sim_stop(fog_config());
    Sim sim_keep(fog_config());
    ObservationEngine eng_stop(
        per_agent_cfg(std::numeric_limits<float>::quiet_NaN(), true));
    ObservationEngine eng_keep(
        per_agent_cfg(std::numeric_limits<float>::quiet_NaN(), true));

    // Phase A (identical streams): walk slot 4 down-left until hidden from
    // slot 1. Build every decision so both engines fold the same sightings.
    std::array<Action, kAgentsPerMatch> hide{};
    hide[4].move_x = -1.0F;
    hide[4].move_y = -1.0F;
    int steps = 0;
    for (; steps < 80 && sim_stop.line_of_sight(1, 4); ++steps) {
        sim_stop.step_decision(hide);
        sim_keep.step_decision(hide);
        build_one(eng_stop, sim_stop, 1);
        build_one(eng_keep, sim_keep, 1);
    }
    ASSERT_FALSE(sim_stop.line_of_sight(1, 4))
        << "test precondition: slot 4 must reach the disc's shadow";
    ASSERT_FALSE(sim_keep.line_of_sight(1, 4));

    // Phase B (divergent, hidden): slot 4 idles in one sim, keeps moving
    // left in the other. Both must stay hidden from slot 1 throughout.
    std::array<Action, kAgentsPerMatch> idle{};
    std::array<Action, kAgentsPerMatch> wander{};
    wander[4].move_x = -1.0F;
    for (int i = 0; i < 12; ++i) {
        sim_stop.step_decision(idle);
        sim_keep.step_decision(wander);
        ASSERT_FALSE(sim_stop.line_of_sight(1, 4));
        ASSERT_FALSE(sim_keep.line_of_sight(1, 4))
            << "test precondition: slot 4 must stay hidden while wandering "
               "(step " << i << ")";
        build_one(eng_stop, sim_stop, 1);
        build_one(eng_keep, sim_keep, 1);
    }
    ASSERT_NE(sim_stop.state().heroes[4].position.x,
              sim_keep.state().heroes[4].position.x)
        << "test precondition: hidden position must actually diverge";

    const auto a = build_one(eng_stop, sim_stop, 1);
    const auto b = build_one(eng_keep, sim_keep, 1);
    EXPECT_TRUE(obs_equal(a, b))
        << "last-seen memory tracked a hidden enemy through cover";
    EXPECT_EQ(eng_stop.obs_state_hash(), eng_keep.obs_state_hash())
        << "last-seen memory itself diverged on hidden movement";
}
