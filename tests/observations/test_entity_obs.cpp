#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include <xushi2/common/limits.hpp>
#include <xushi2/sim/entity_obs.h>
#include <xushi2/sim/obs.h>
#include <xushi2/sim/obs_config.h>
#include <xushi2/sim/obs_utils.h>
#include <xushi2/sim/sim.h>

#include "test_config.hpp"

// ObservationEngine field/shape/lifecycle tests. The counterfactual leak
// suite lives in test_entity_obs_leak.cpp.
//
// Geometry facts used throughout (50x50 default map, team_size 3):
//   Team A spawns: slots 0/1/2 at x = 17.5/25/32.5, y = 5.
//   Team B spawns: slots 3/4/5 at the same x, y = 45.
//   Normalized (half-extent 25) distance slot1 -> slot4 is exactly 1.6;
//   slot1 -> slot3 or slot5 is sqrt(0.3^2 + 1.6^2) ~= 1.6279.
//   The fog disc at (20, 15) r=4 blocks slot0 -> slots 3/4/5 at spawn but
//   leaves slot1 -> slot4 (the vertical x=25 line) clear.

namespace {

using xushi2::common::Action;
using xushi2::sim::FogMode;
using xushi2::sim::kActorObsPhase1Dim;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::kEntityGridObsDim;
using xushi2::sim::kEntityGridSize;
using xushi2::sim::kEntityTokenDim;
using xushi2::sim::MatchConfig;
using xushi2::sim::ObsConfig;
using xushi2::sim::ObservationEngine;
using xushi2::sim::Sim;

using EntityObs = std::array<float, kEntityGridObsDim>;

// Token field offsets — mirror multi_enemy_obs.py.
constexpr std::uint32_t kHp = 6;
constexpr std::uint32_t kAlive = 7;
constexpr std::uint32_t kPos = 8;
constexpr std::uint32_t kVel = 10;
constexpr std::uint32_t kAim = 12;
constexpr std::uint32_t kAmmo = 14;
constexpr std::uint32_t kReloading = 15;
constexpr std::uint32_t kAbilityCd = 16;
constexpr std::uint32_t kAux = 17;

constexpr std::uint32_t kMaskBase = 5 * kEntityTokenDim;  // 90
constexpr std::uint32_t kGridBase = kMaskBase + 5;        // 95

const float* token(const EntityObs& obs, std::uint32_t idx) {
    return obs.data() + (idx * kEntityTokenDim);
}

float mask_bit(const EntityObs& obs, std::uint32_t idx) {
    return obs[kMaskBase + idx];
}

const float* grid_channel(const EntityObs& obs, std::uint32_t ch) {
    return obs.data() + kGridBase + (ch * kEntityGridSize * kEntityGridSize);
}

int count_grid_cells(const EntityObs& obs, std::uint32_t ch, float value) {
    const float* g = grid_channel(obs, ch);
    int n = 0;
    for (std::uint32_t i = 0; i < kEntityGridSize * kEntityGridSize; ++i) {
        if (g[i] == value) ++n;
    }
    return n;
}

MatchConfig base_config(bool fog) {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 0xE17177E5ULL;
    cfg.round_length_seconds = 30;
    cfg.fog_of_war_enabled = fog;
    cfg.team_size = 3;
    return cfg;
}

// Same disc as test_actor_leak.cpp's fog config: blocks slot 0's diagonals
// to the far side, leaves the x=25 vertical clear.
MatchConfig fog_config() {
    auto cfg = base_config(true);
    xushi2::sim::CoverCircle cover{};
    cover.center = xushi2::common::Vec2{20.0F, 15.0F};
    cover.radius = 4.0F;
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0] = cover;
    return cfg;
}

ObsConfig no_fog_obs_cfg() {
    ObsConfig cfg{};
    cfg.fog_mode = FogMode::None;
    return cfg;
}

ObsConfig per_agent_obs_cfg(float radius = std::numeric_limits<float>::quiet_NaN(),
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
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-7F) return false;
    }
    return true;
}

}  // namespace

TEST(EntityObs, SelfAndObjectiveTokensMatchActorObs) {
    Sim sim(base_config(false));
    ObservationEngine engine(no_fog_obs_cfg());

    // Wander a bit so positions/velocities are non-trivial.
    std::array<Action, kAgentsPerMatch> act{};
    act[0].move_x = 0.7F;
    act[0].move_y = 1.0F;
    act[0].aim_delta = 0.2F;
    act[0].primary_fire = true;
    for (int i = 0; i < 12; ++i) sim.step_decision(act);

    std::array<float, kActorObsPhase1Dim> actor{};
    xushi2::sim::build_actor_obs_phase1(sim, 0, actor.data(),
                                        kActorObsPhase1Dim);
    const auto obs = build_one(engine, sim, 0);
    const float* self_tok = token(obs, 0);

    // Actor obs offsets: own_hp 0, own_velocity 1-2, own_aim_unit 3-4,
    // own_position 5-6, own_ammo 7, own_reloading 8, own_combat_roll_cd 9,
    // objective_owner_onehot 17-19, cap_progress 23, self_on_point 28.
    EXPECT_FLOAT_EQ(self_tok[0], 1.0F);  // kind = hero(self)
    EXPECT_FLOAT_EQ(self_tok[3], 1.0F);  // team = own
    EXPECT_FLOAT_EQ(self_tok[kHp], actor[0]);
    EXPECT_FLOAT_EQ(self_tok[kAlive], 1.0F);
    EXPECT_FLOAT_EQ(self_tok[kPos + 0], actor[5]);
    EXPECT_FLOAT_EQ(self_tok[kPos + 1], actor[6]);
    EXPECT_FLOAT_EQ(self_tok[kVel + 0], actor[1]);
    EXPECT_FLOAT_EQ(self_tok[kVel + 1], actor[2]);
    EXPECT_FLOAT_EQ(self_tok[kAim + 0], actor[3]);
    EXPECT_FLOAT_EQ(self_tok[kAim + 1], actor[4]);
    EXPECT_FLOAT_EQ(self_tok[kAmmo], actor[7]);
    EXPECT_FLOAT_EQ(self_tok[kReloading], actor[8]);
    EXPECT_FLOAT_EQ(self_tok[kAbilityCd], actor[9]);
    EXPECT_FLOAT_EQ(self_tok[kAux], actor[28]);

    const float* obj_tok = token(obs, 4);
    EXPECT_FLOAT_EQ(obj_tok[2], 1.0F);  // kind = objective
    EXPECT_FLOAT_EQ(obj_tok[3], actor[17]);
    EXPECT_FLOAT_EQ(obj_tok[4], actor[18]);
    EXPECT_FLOAT_EQ(obj_tok[5], actor[19]);
    EXPECT_FLOAT_EQ(obj_tok[kAlive], 1.0F);
    EXPECT_FLOAT_EQ(obj_tok[kPos + 0], -actor[5]);
    EXPECT_FLOAT_EQ(obj_tok[kPos + 1], -actor[6]);
    EXPECT_FLOAT_EQ(obj_tok[kAux], actor[23]);

    EXPECT_FLOAT_EQ(mask_bit(obs, 0), 1.0F);
    EXPECT_FLOAT_EQ(mask_bit(obs, 4), 1.0F);

    // Grid: objective mark on ch0, self mark on ch1.
    EXPECT_EQ(count_grid_cells(obs, 0, 1.0F), 1);
    EXPECT_EQ(count_grid_cells(obs, 1, 1.0F), 1);
}

TEST(EntityObs, VisibleEnemyTokenMatchesHandComputedTransforms) {
    Sim sim(base_config(false));
    ObservationEngine engine(no_fog_obs_cfg());

    std::array<Action, kAgentsPerMatch> act{};
    act[4].move_x = -0.4F;
    act[4].move_y = -1.0F;
    act[4].aim_delta = 0.3F;
    for (int i = 0; i < 10; ++i) sim.step_decision(act);

    const auto& s = sim.state();
    const auto& map = sim.config().map;
    const auto& viewer = s.heroes[0];
    const auto& enemy = s.heroes[4];  // enemy index 1 for a Team A viewer

    const auto obs = build_one(engine, sim, 0);
    const float* tok = token(obs, 2);  // first enemy token is index 1

    namespace ou = xushi2::sim::obs_utils;
    const auto own_norm = ou::normalize_position_to_map(
        ou::mirror_position_for_team(viewer.position, viewer.team, map), map);
    const auto enemy_norm = ou::normalize_position_to_map(
        ou::mirror_position_for_team(enemy.position, viewer.team, map), map);

    EXPECT_FLOAT_EQ(tok[1], 1.0F);  // kind = enemy
    EXPECT_FLOAT_EQ(tok[4], 1.0F);  // team = enemy
    EXPECT_FLOAT_EQ(tok[kHp],
                    static_cast<float>(enemy.health_centi_hp) /
                        static_cast<float>(enemy.max_health_centi_hp));
    EXPECT_FLOAT_EQ(tok[kAlive], 1.0F);
    EXPECT_FLOAT_EQ(tok[kPos + 0], enemy_norm.x - own_norm.x);
    EXPECT_FLOAT_EQ(tok[kPos + 1], enemy_norm.y - own_norm.y);
    EXPECT_FLOAT_EQ(tok[kVel + 0],
                    enemy.velocity.x / ou::ranger_max_speed());
    EXPECT_FLOAT_EQ(tok[kVel + 1],
                    enemy.velocity.y / ou::ranger_max_speed());
    // World-frame aim, never mirrored (parity wart).
    EXPECT_FLOAT_EQ(tok[kAim + 0], std::sin(enemy.aim_angle));
    EXPECT_FLOAT_EQ(tok[kAim + 1], std::cos(enemy.aim_angle));
    EXPECT_FLOAT_EQ(tok[kAmmo],
                    static_cast<float>(enemy.weapon.magazine) /
                        static_cast<float>(xushi2::common::kRangerMaxMagazine));
    EXPECT_FLOAT_EQ(tok[kReloading], enemy.weapon.reloading ? 1.0F : 0.0F);
    EXPECT_FLOAT_EQ(mask_bit(obs, 2), 1.0F);
    EXPECT_GE(count_grid_cells(obs, 2, 1.0F), 1);

    // Team-B viewer sees mirrored positions/velocities but the same
    // world-frame aim.
    const auto obs_b = build_one(engine, sim, 3);
    const float* tok_b = token(obs_b, 2);  // enemy index 1 = slot 1 for Team B
    const auto& enemy_of_b = s.heroes[1];
    EXPECT_FLOAT_EQ(tok_b[kAim + 0], std::sin(enemy_of_b.aim_angle));
    EXPECT_FLOAT_EQ(tok_b[kAim + 1], std::cos(enemy_of_b.aim_angle));
    EXPECT_FLOAT_EQ(tok_b[kVel + 0],
                    -enemy_of_b.velocity.x / ou::ranger_max_speed());
    EXPECT_FLOAT_EQ(tok_b[kVel + 1],
                    -enemy_of_b.velocity.y / ou::ranger_max_speed());
}

TEST(EntityObs, RadiusBoundaryJustInsideAndOutside) {
    // At spawn, slot1 -> slot4 normalized distance is exactly 1.6; the
    // flanking enemies are at ~1.6279. Fog is off so the LoS term is
    // all-true and only the radius decides.
    Sim sim(base_config(false));

    ObservationEngine inside(per_agent_obs_cfg(1.61F));
    const auto vis_in = inside.visible_enemies(sim, 1);
    EXPECT_FALSE(vis_in[0]);  // slot 3 at 1.6279
    EXPECT_TRUE(vis_in[1]);   // slot 4 at 1.6
    EXPECT_FALSE(vis_in[2]);  // slot 5 at 1.6279

    ObservationEngine outside(per_agent_obs_cfg(1.59F));
    const auto vis_out = outside.visible_enemies(sim, 1);
    EXPECT_FALSE(vis_out[0]);
    EXPECT_FALSE(vis_out[1]);
    EXPECT_FALSE(vis_out[2]);

    // Symmetric for a Team-B viewer (mirrored frame, same distances).
    const auto vis_b = inside.visible_enemies(sim, 4);
    EXPECT_FALSE(vis_b[0]);
    EXPECT_TRUE(vis_b[1]);  // slot 1
    EXPECT_FALSE(vis_b[2]);

    // Radius covering the whole map sees every live enemy.
    ObservationEngine all(per_agent_obs_cfg(3.0F));
    const auto vis_all = all.visible_enemies(sim, 1);
    EXPECT_TRUE(vis_all[0]);
    EXPECT_TRUE(vis_all[1]);
    EXPECT_TRUE(vis_all[2]);
}

TEST(EntityObs, TeamSharedUnionRevealsThroughTeammateLoS) {
    // At spawn the disc hides every far-side slot from slot 0, but slot 1's
    // vertical line to slot 4 is clear. Per-agent: hidden. Team-shared:
    // visible through the teammate.
    Sim sim(fog_config());
    ASSERT_FALSE(sim.line_of_sight(0, 4));
    ASSERT_TRUE(sim.line_of_sight(1, 4));

    ObservationEngine per_agent(per_agent_obs_cfg());
    EXPECT_FALSE(per_agent.visible_enemies(sim, 0)[1]);

    ObsConfig shared_cfg{};
    shared_cfg.fog_mode = FogMode::TeamShared;
    ObservationEngine shared(shared_cfg);
    EXPECT_TRUE(shared.visible_enemies(sim, 0)[1]);
}

TEST(EntityObs, LastSeenLifecycle) {
    Sim sim(fog_config());
    ObservationEngine engine(
        per_agent_obs_cfg(std::numeric_limits<float>::quiet_NaN(), true));

    ASSERT_TRUE(sim.line_of_sight(1, 4));
    auto obs = build_one(engine, sim, 1);
    EXPECT_FLOAT_EQ(token(obs, 2)[kAlive], 1.0F);
    EXPECT_FLOAT_EQ(mask_bit(obs, 2), 1.0F);

    // Slot 4 walks down-left into the disc's shadow.
    std::array<Action, kAgentsPerMatch> hide{};
    hide[4].move_x = -1.0F;
    hide[4].move_y = -1.0F;
    int steps = 0;
    for (; steps < 80 && sim.line_of_sight(1, 4); ++steps) {
        sim.step_decision(hide);
        build_one(engine, sim, 1);  // keep last-seen memory current
    }
    ASSERT_FALSE(sim.line_of_sight(1, 4))
        << "test precondition: slot 4 must end up hidden from slot 1";

    obs = build_one(engine, sim, 1);
    const float* stale_tok = token(obs, 2);
    EXPECT_FLOAT_EQ(stale_tok[1], 1.0F);  // kind/team markers stay
    EXPECT_FLOAT_EQ(stale_tok[4], 1.0F);
    EXPECT_FLOAT_EQ(stale_tok[kAlive], 0.0F);
    EXPECT_FLOAT_EQ(stale_tok[kAux], 0.5F);
    EXPECT_FLOAT_EQ(mask_bit(obs, 2), 1.0F);
    EXPECT_NE(stale_tok[kPos + 1], 0.0F)
        << "stale marker must carry the remembered relative position";
    // Live fields must be blanked in the stale form.
    EXPECT_FLOAT_EQ(stale_tok[kHp], 0.0F);
    EXPECT_FLOAT_EQ(stale_tok[kVel + 0], 0.0F);
    EXPECT_FLOAT_EQ(stale_tok[kVel + 1], 0.0F);
    EXPECT_FLOAT_EQ(stale_tok[kAim + 0], 0.0F);
    EXPECT_FLOAT_EQ(stale_tok[kAim + 1], 0.0F);
    EXPECT_GE(count_grid_cells(obs, 2, 0.5F), 1);

    // Walk back out until visible again: token returns to live form.
    std::array<Action, kAgentsPerMatch> unhide{};
    unhide[4].move_x = 1.0F;
    unhide[4].move_y = 1.0F;
    for (steps = 0; steps < 80 && !sim.line_of_sight(1, 4); ++steps) {
        sim.step_decision(unhide);
        build_one(engine, sim, 1);
    }
    ASSERT_TRUE(sim.line_of_sight(1, 4));
    obs = build_one(engine, sim, 1);
    EXPECT_FLOAT_EQ(token(obs, 2)[kAlive], 1.0F);
    EXPECT_NE(token(obs, 2)[kAux], 0.5F);

    // Hide again, then reset(): memory clears, token becomes fully hidden.
    for (steps = 0; steps < 80 && sim.line_of_sight(1, 4); ++steps) {
        sim.step_decision(hide);
        build_one(engine, sim, 1);
    }
    ASSERT_FALSE(sim.line_of_sight(1, 4));
    engine.reset();
    obs = build_one(engine, sim, 1);
    EXPECT_FLOAT_EQ(mask_bit(obs, 2), 0.0F);
    EXPECT_FLOAT_EQ(token(obs, 2)[kAlive], 0.0F);
    EXPECT_FLOAT_EQ(token(obs, 2)[kPos + 1], 0.0F);
}

TEST(EntityObs, BuildIsIdempotentWithinATick) {
    Sim sim(fog_config());
    ObservationEngine engine(
        per_agent_obs_cfg(std::numeric_limits<float>::quiet_NaN(), true));

    std::array<Action, kAgentsPerMatch> act{};
    act[4].move_y = -1.0F;
    for (int i = 0; i < 5; ++i) sim.step_decision(act);

    const auto first = build_one(engine, sim, 1);
    const auto hash_first = engine.obs_state_hash();
    const auto second = build_one(engine, sim, 1);
    EXPECT_TRUE(obs_equal(first, second));
    EXPECT_EQ(hash_first, engine.obs_state_hash());
}

TEST(EntityObs, ZeroHiddenTokenMarkersStripsKindAndTeam) {
    // Radius 1.0 hides every enemy at spawn (nearest is 1.6).
    Sim sim(base_config(false));

    ObservationEngine keep(per_agent_obs_cfg(1.0F));
    const auto obs_keep = build_one(keep, sim, 1);
    EXPECT_FLOAT_EQ(token(obs_keep, 1)[1], 1.0F);
    EXPECT_FLOAT_EQ(token(obs_keep, 1)[4], 1.0F);
    EXPECT_FLOAT_EQ(mask_bit(obs_keep, 1), 0.0F);

    ObsConfig zero_cfg = per_agent_obs_cfg(1.0F);
    zero_cfg.zero_hidden_token_markers = true;
    ObservationEngine zero(zero_cfg);
    const auto obs_zero = build_one(zero, sim, 1);
    for (std::uint32_t i = 0; i < kEntityTokenDim; ++i) {
        EXPECT_FLOAT_EQ(token(obs_zero, 1)[i], 0.0F);
        EXPECT_FLOAT_EQ(token(obs_zero, 2)[i], 0.0F);
        EXPECT_FLOAT_EQ(token(obs_zero, 3)[i], 0.0F);
    }
    EXPECT_FLOAT_EQ(mask_bit(obs_zero, 1), 0.0F);
}

TEST(EntityObs, DeterministicAcrossEnginePairs) {
    Sim sim_a(fog_config());
    Sim sim_b(fog_config());
    const auto obs_cfg =
        per_agent_obs_cfg(std::numeric_limits<float>::quiet_NaN(), true);
    ObservationEngine engine_a(obs_cfg);
    ObservationEngine engine_b(obs_cfg);

    std::array<Action, kAgentsPerMatch> act{};
    act[4].move_x = -1.0F;
    act[4].move_y = -1.0F;
    act[0].move_y = 1.0F;
    act[0].aim_delta = 0.1F;

    std::array<float, kAgentsPerMatch * kEntityGridObsDim> all_a{};
    std::array<float, kAgentsPerMatch * kEntityGridObsDim> all_b{};
    for (int i = 0; i < 20; ++i) {
        sim_a.step_decision(act);
        sim_b.step_decision(act);
        engine_a.build_entity_obs_all(sim_a, all_a.data(),
                                      static_cast<std::uint32_t>(all_a.size()));
        engine_b.build_entity_obs_all(sim_b, all_b.data(),
                                      static_cast<std::uint32_t>(all_b.size()));
        ASSERT_EQ(engine_a.obs_state_hash(), engine_b.obs_state_hash());
        for (std::size_t j = 0; j < all_a.size(); ++j) {
            ASSERT_EQ(all_a[j], all_b[j]) << "index " << j << " step " << i;
        }
    }
}
