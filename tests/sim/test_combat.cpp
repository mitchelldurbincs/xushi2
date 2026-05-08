#include <gtest/gtest.h>

#include <array>

#include <test_config.hpp>
#include <xushi2/common/limits.hpp>
#include <xushi2/common/math.hpp>
#include <xushi2/sim/sim.h>

// Phase-1a combat: hitscan Revolver, damage apply, kill on HP depletion,
// fire-rate gate, simultaneous-kill tie-break.
//
// Setup: a tiny arena so the two default Rangers start within Revolver range
// and are facing each other (reset_state points team A up, team B down).

namespace {

using xushi2::common::Action;
using xushi2::sim::kAgentsPerMatch;
using xushi2::sim::MatchConfig;
using xushi2::sim::Sim;

MatchConfig close_arena_config() {
    auto cfg = xushi2::test_support::make_test_config();
    cfg.seed = 1;
    cfg.round_length_seconds = 30;
    cfg.map.max_x = 5.0F;
    cfg.map.max_y = 5.0F;
    return cfg;
}

}  // namespace

TEST(Combat, HitscanHitsEnemyAndReducesHp) {
    Sim sim(close_arena_config());
    const auto start_hp = sim.state().heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;  // only team A fires
    sim.step(actions);

    // Team A's mag decremented; Team B took damage.
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);
    EXPECT_LT(sim.state().heroes[3].health_centi_hp, start_hp);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp,
              start_hp - 7500);  // 75.0 HP from config
}

TEST(Combat, CoverCircleBlocksRevolverHitscanAndLineOfSight) {
    MatchConfig cfg = close_arena_config();
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0].center = xushi2::common::Vec2{2.5F, 2.5F};
    cfg.cover_circles[0].radius = 0.35F;
    Sim sim(cfg);
    EXPECT_FALSE(sim.line_of_sight(0, 3));
    const auto start_hp = sim.state().heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);

    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, start_hp);
}

TEST(Combat, WallSegmentBlocksRevolverHitscanAndLineOfSight) {
    MatchConfig cfg = close_arena_config();
    cfg.num_wall_segments = 1;
    cfg.wall_segments[0].a = xushi2::common::Vec2{2.5F, 2.0F};
    cfg.wall_segments[0].b = xushi2::common::Vec2{2.5F, 3.0F};
    cfg.wall_segments[0].half_width = 0.15F;
    Sim sim(cfg);
    EXPECT_FALSE(sim.line_of_sight(0, 3));
    const auto start_hp = sim.state().heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);

    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, start_hp);
}

TEST(Movement, CoverCircleRepelsHeroMovement) {
    MatchConfig cfg = close_arena_config();
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0].center = xushi2::common::Vec2{2.5F, 2.5F};
    cfg.cover_circles[0].radius = 0.5F;
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.4F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].move_y = 1.0F;
    sim.step(actions);

    const auto& hero = sim.state().heroes[0];
    const float dx = hero.position.x - cfg.cover_circles[0].center.x;
    const float dy = hero.position.y - cfg.cover_circles[0].center.y;
    EXPECT_GE(dx * dx + dy * dy,
              cfg.cover_circles[0].radius * cfg.cover_circles[0].radius - 1e-5F);
}

TEST(Movement, WallSegmentRepelsHeroMovement) {
    MatchConfig cfg = close_arena_config();
    cfg.num_wall_segments = 1;
    cfg.wall_segments[0].a = xushi2::common::Vec2{2.5F, 2.0F};
    cfg.wall_segments[0].b = xushi2::common::Vec2{2.5F, 3.0F};
    cfg.wall_segments[0].half_width = 0.25F;
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.4F, 2.5F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].move_x = 1.0F;
    sim.step(actions);

    EXPECT_LE(sim.state().heroes[0].position.x, 2.25F + 1e-5F);
}

TEST(Combat, FireRateGateSkipsConsecutiveShots) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    // Tick 0: fires, mag=5, fire_cooldown=15.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);

    // Tick 1: fire gate closed, mag unchanged.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 5);

    // Step through until cooldown expires (tick 15 total). Keep firing input.
    for (int i = 0; i < 14; ++i) {
        sim.step(actions);
    }
    // By tick 15 the gate opens and the shot lands.
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 4);
}

TEST(Combat, RangerMarkTargetUsesEnemyTargetSlotAndSetsCooldown) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[1].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[1].ability_2 = true;
    actions[1].target_slot = 1;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[1].cd_ability_2, xushi2::common::kRangerMarkTargetCooldownTicks);
    EXPECT_EQ(heroes[4].ranger_marked_ticks,
              xushi2::common::kRangerMarkTargetDurationTicks);
    EXPECT_EQ(heroes[4].ranger_marked_by, xushi2::common::Team::A);
}

TEST(Combat, RangerMarkTargetIgnoresNonEnemyTargetSlot) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[1].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[1].ability_2 = true;
    actions[1].target_slot = 2;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[1].cd_ability_2, 0U);
    EXPECT_EQ(heroes[4].ranger_marked_ticks, 0U);
    EXPECT_EQ(heroes[4].ranger_marked_by, xushi2::common::Team::Neutral);
}

TEST(Combat, RangerMarkTargetBlockedByCoverLineOfSight) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0].center = xushi2::common::Vec2{2.5F, 3.0F};
    cfg.cover_circles[0].radius = 0.35F;
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[1].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[1].ability_2 = true;
    actions[1].target_slot = 1;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[1].cd_ability_2, xushi2::common::kRangerMarkTargetCooldownTicks);
    EXPECT_EQ(heroes[4].ranger_marked_ticks, 0U);
    EXPECT_EQ(heroes[4].ranger_marked_by, xushi2::common::Team::Neutral);
}

TEST(Combat, RangerMarkTargetExpires) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[4].ranger_marked_ticks = 1;
    state.heroes[4].ranger_marked_by = xushi2::common::Team::A;

    std::array<Action, kAgentsPerMatch> actions{};
    sim.step(actions);

    const auto& target = sim.state().heroes[4];
    EXPECT_EQ(target.ranger_marked_ticks, 0U);
    EXPECT_EQ(target.ranger_marked_by, xushi2::common::Team::Neutral);
}

TEST(Combat, EmptyMagazineFireIsNoop) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;

    // Fire until mag is empty. Each shot uses 15-tick cooldown.
    for (int k = 0; k < 6; ++k) {
        sim.step(actions);
        for (int j = 0; j < 14; ++j) {
            sim.step(actions);
        }
    }
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 0);
    const auto enemy_hp_after_empty = sim.state().heroes[3].health_centi_hp;

    // Step one more tick with fire held; mag stays 0, enemy HP unchanged.
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[0].weapon.magazine, 0);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, enemy_hp_after_empty);
}

TEST(Combat, KillsCreditedAndDeathsTracked) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;

    // Two 75-HP hits kill a 150-HP Ranger. Space them by fire-rate cooldown.
    sim.step(actions);
    for (int j = 0; j < 14; ++j) {
        sim.step(actions);
    }
    sim.step(actions);  // the killing shot

    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 0);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].deaths, 1U);
    EXPECT_EQ(sim.team_a_kills(), 1U);
    EXPECT_EQ(sim.team_b_kills(), 0U);
}

TEST(Combat, DamageDealtBySlotCreditsAttackerForAppliedDamage) {
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);  // first shot: 7500 cHP applied to a 15000 cHP target

    const auto damage = sim.damage_dealt_by_slot();
    EXPECT_EQ(damage.size(), kAgentsPerMatch);
    EXPECT_EQ(damage[0], 7500U);
    for (std::size_t i = 1; i < kAgentsPerMatch; ++i) {
        EXPECT_EQ(damage[i], 0U) << "unexpected damage credited at slot " << i;
    }
}

TEST(Combat, DamageDealtClampsToRemainingHpOnKillingShot) {
    // Drive slot 3 to exactly 7500 cHP (one revolver shot of HP), then fire
    // a second shot of 7500 cHP. The killing shot must only credit 7500
    // (the remaining HP), not the full 7500 of the event — even though
    // those happen to be equal here, the *clamping logic* under
    // apply_damage_buffer is what we're checking. Drive a third shot post-
    // death and assert no further credit accrues.
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;

    sim.step(actions);  // shot 1: -7500 cHP → 7500 left
    for (int j = 0; j < 14; ++j) sim.step(actions);  // wait for cooldown
    sim.step(actions);  // shot 2: kills (7500 cHP applied)

    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 0);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    // Total credit = 7500 + 7500 = 15000 (one full Ranger HP pool).
    const auto damage = sim.damage_dealt_by_slot();
    EXPECT_EQ(damage[0], 15000U);
}

TEST(Combat, KillsBySlotAndDeathsBySlotMirrorHeroStateCounters) {
    // Same setup as KillsCreditedAndDeathsTracked: slot 0 (A Ranger) kills
    // slot 3 (B Ranger). Verify the new array-shaped accessors snapshot the
    // per-slot counters that already live on HeroState.
    Sim sim(close_arena_config());
    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);
    for (int j = 0; j < 14; ++j) {
        sim.step(actions);
    }
    sim.step(actions);  // killing shot

    const auto kills = sim.kills_by_slot();
    const auto deaths = sim.deaths_by_slot();
    EXPECT_EQ(kills.size(), kAgentsPerMatch);
    EXPECT_EQ(deaths.size(), kAgentsPerMatch);
    EXPECT_EQ(kills[0], 1U);
    EXPECT_EQ(deaths[3], 1U);
    for (std::size_t i = 1; i < kAgentsPerMatch; ++i) {
        EXPECT_EQ(kills[i], 0U) << "unexpected kill at slot " << i;
    }
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        if (i == 3) continue;
        EXPECT_EQ(deaths[i], 0U) << "unexpected death at slot " << i;
    }
}

TEST(Combat, SimultaneousKillBothDie) {
    // Drive both Rangers down to exactly 75 HP (one shot kill), then fire
    // on the same tick; both should die.
    Sim sim(close_arena_config());

    // Step 1: A fires, B down to 75 HP. A.cd=15, B.cd=0.
    std::array<Action, kAgentsPerMatch> only_a{};
    only_a[0].primary_fire = true;
    sim.step(only_a);
    EXPECT_EQ(sim.state().heroes[3].health_centi_hp, 7500);

    // Step 2: B fires this tick, A down to 75. B.cd=15, A.cd=14.
    std::array<Action, kAgentsPerMatch> only_b{};
    only_b[3].primary_fire = true;
    sim.step(only_b);
    EXPECT_EQ(sim.state().heroes[0].health_centi_hp, 7500);

    // Wait for both cooldowns to clear. A.cd=14, B.cd=15. Need 15 idle ticks.
    for (int j = 0; j < 15; ++j) {
        std::array<Action, kAgentsPerMatch> idle{};
        sim.step(idle);
    }
    EXPECT_EQ(sim.state().heroes[0].weapon.fire_cooldown_ticks, 0U);
    EXPECT_EQ(sim.state().heroes[3].weapon.fire_cooldown_ticks, 0U);

    // Both fire simultaneously. Damage accumulates, applies, both die.
    std::array<Action, kAgentsPerMatch> both{};
    both[0].primary_fire = true;
    both[3].primary_fire = true;
    sim.step(both);

    EXPECT_FALSE(sim.state().heroes[0].alive);
    EXPECT_FALSE(sim.state().heroes[3].alive);
    EXPECT_EQ(sim.state().heroes[0].kills, 1U);
    EXPECT_EQ(sim.state().heroes[3].kills, 1U);
}

TEST(Combat, VanguardBarrierAbsorbsEnemyRevolverShot) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};
    state.heroes[4].aim_angle = -0.5F * xushi2::common::kPi;
    const auto start_hp = state.heroes[0].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].ability_1 = true;
    actions[4].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[0].health_centi_hp, start_hp);
    EXPECT_TRUE(heroes[0].vanguard_barrier_active);
    EXPECT_EQ(
        heroes[0].vanguard_barrier_hp_centi,
        xushi2::common::kVanguardBarrierHpCenti -
            static_cast<std::int32_t>(cfg.mechanics.revolver_damage_centi_hp));
    EXPECT_EQ(heroes[4].weapon.magazine, 5);
}

TEST(Combat, VanguardBarrierBreaksAndArmsCooldown) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[0].vanguard_barrier_hp_centi = 1000;
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};
    state.heroes[4].aim_angle = -0.5F * xushi2::common::kPi;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].ability_1 = true;
    actions[4].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_FALSE(heroes[0].vanguard_barrier_active);
    EXPECT_EQ(heroes[0].vanguard_barrier_hp_centi, 0);
    EXPECT_EQ(heroes[0].cd_ability_1,
              xushi2::common::kVanguardBarrierBrokenCooldownTicks);
}

TEST(Combat, VanguardGuardStepDashesForwardAndSetsCooldown) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[0].aim_angle = 0.5F * xushi2::common::kPi;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].ability_2 = true;
    sim.step(actions);

    const auto& hero = sim.state().heroes[0];
    EXPECT_NEAR(hero.position.x, 2.5F, 1e-5F);
    EXPECT_NEAR(hero.position.y, 4.0F, 1e-5F);
    EXPECT_EQ(hero.cd_ability_2, xushi2::common::kVanguardGuardStepCooldownTicks);
}

TEST(Combat, VanguardGuardStepCooldownPreventsImmediateSecondDash) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 1.0F};
    state.heroes[0].aim_angle = 0.5F * xushi2::common::kPi;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].ability_2 = true;
    sim.step(actions);
    const float after_first = sim.state().heroes[0].position.y;
    sim.step(actions);

    EXPECT_NEAR(sim.state().heroes[0].position.y, after_first, 1e-5F);
    EXPECT_EQ(sim.state().heroes[0].cd_ability_2,
              xushi2::common::kVanguardGuardStepCooldownTicks - 1U);
}

TEST(Combat, VanguardWarhammerHitsNearestEnemyInCone) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[0].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[3].position = xushi2::common::Vec2{2.5F, 4.0F};
    state.heroes[4].position = xushi2::common::Vec2{2.0F, 4.5F};
    const auto target_start_hp = state.heroes[3].health_centi_hp;
    const auto other_start_hp = state.heroes[4].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[3].health_centi_hp,
              target_start_hp - xushi2::common::kVanguardWarhammerDamageCentiHp);
    EXPECT_EQ(heroes[4].health_centi_hp, other_start_hp);
    EXPECT_EQ(heroes[0].weapon.fire_cooldown_ticks,
              xushi2::common::kVanguardWarhammerCooldownTicks);
}

TEST(Combat, VanguardWarhammerSuppressedWhileBarrierActive) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[0].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[3].position = xushi2::common::Vec2{2.5F, 4.0F};
    const auto target_start_hp = state.heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].primary_fire = true;
    actions[0].ability_1 = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_TRUE(heroes[0].vanguard_barrier_active);
    EXPECT_EQ(heroes[3].health_centi_hp, target_start_hp);
    EXPECT_EQ(heroes[0].weapon.fire_cooldown_ticks, 0U);
}

TEST(Combat, MenderSidearmHitsEnemyAndSetsCooldown) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[2].mender_weapon = xushi2::common::MenderWeapon::Sidearm;
    state.heroes[3].position = xushi2::common::Vec2{2.5F, 4.0F};
    const auto target_start_hp = state.heroes[3].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[3].health_centi_hp,
              target_start_hp - xushi2::common::kMenderSidearmDamageCentiHp);
    EXPECT_EQ(heroes[2].weapon.fire_cooldown_ticks,
              xushi2::common::kMenderSidearmCooldownTicks);
}

TEST(Combat, MenderSidearmBlockedByEnemyBarrier) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[2].mender_weapon = xushi2::common::MenderWeapon::Sidearm;
    state.heroes[3].position = xushi2::common::Vec2{2.5F, 3.0F};
    state.heroes[3].vanguard_barrier_active = true;
    state.heroes[3].vanguard_barrier_hp_centi = xushi2::common::kVanguardBarrierHpCenti;
    state.heroes[4].position = xushi2::common::Vec2{2.5F, 4.0F};
    const auto target_start_hp = state.heroes[4].health_centi_hp;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].primary_fire = true;
    actions[3].ability_1 = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[4].health_centi_hp, target_start_hp);
    EXPECT_EQ(heroes[3].vanguard_barrier_hp_centi,
              xushi2::common::kVanguardBarrierHpCenti -
                  xushi2::common::kMenderSidearmDamageCentiHp);
    EXPECT_EQ(heroes[2].weapon.fire_cooldown_ticks,
              xushi2::common::kMenderSidearmCooldownTicks);
}

TEST(Combat, MenderStaffBeamLocksNearestAllyAndHeals) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 3.0F};
    state.heroes[1].position = xushi2::common::Vec2{2.5F, 4.0F};
    state.heroes[0].health_centi_hp =
        xushi2::common::kVanguardMaxHpCentiHp - 1000;
    state.heroes[1].health_centi_hp =
        xushi2::common::kRangerMaxHpCentiHp - 1000;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_EQ(heroes[2].mender_beam_locked_on, heroes[0].id);
    EXPECT_EQ(heroes[0].health_centi_hp,
              xushi2::common::kVanguardMaxHpCentiHp - 1000 +
                  xushi2::common::kMenderBeamHealCentiHpPerTick);
    EXPECT_EQ(heroes[1].health_centi_hp,
              xushi2::common::kRangerMaxHpCentiHp - 1000);
}

TEST(Combat, MenderStaffBeamHealsThroughAlliedBarrier) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 3.0F};
    state.heroes[0].vanguard_barrier_active = true;
    state.heroes[0].vanguard_barrier_hp_centi =
        xushi2::common::kVanguardBarrierHpCenti;
    state.heroes[0].health_centi_hp =
        xushi2::common::kVanguardMaxHpCentiHp - 1000;
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[0].ability_1 = true;
    actions[2].primary_fire = true;
    sim.step(actions);

    const auto& heroes = sim.state().heroes;
    EXPECT_TRUE(heroes[0].vanguard_barrier_active);
    EXPECT_EQ(heroes[0].vanguard_barrier_hp_centi,
              xushi2::common::kVanguardBarrierHpCenti);
    EXPECT_EQ(heroes[2].mender_beam_locked_on, heroes[0].id);
    EXPECT_EQ(heroes[0].health_centi_hp,
              xushi2::common::kVanguardMaxHpCentiHp - 1000 +
                  xushi2::common::kMenderBeamHealCentiHpPerTick);
}

TEST(Combat, CoverCircleBlocksMenderStaffBeam) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    cfg.num_cover_circles = 1;
    cfg.cover_circles[0].center = xushi2::common::Vec2{2.5F, 2.5F};
    cfg.cover_circles[0].radius = 0.35F;
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 3.0F};
    state.heroes[0].health_centi_hp =
        xushi2::common::kVanguardMaxHpCentiHp - 1000;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].primary_fire = true;
    sim.step(actions);

    EXPECT_EQ(sim.state().heroes[2].mender_beam_locked_on, 0U);
    EXPECT_EQ(sim.state().heroes[0].health_centi_hp,
              xushi2::common::kVanguardMaxHpCentiHp - 1000);
}

TEST(Combat, MenderStaffBeamBreaksOnReleaseAndWeaponSwap) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 2.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 3.0F};
    state.heroes[0].health_centi_hp =
        xushi2::common::kVanguardMaxHpCentiHp - 1000;

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].primary_fire = true;
    sim.step(actions);
    ASSERT_EQ(sim.state().heroes[2].mender_beam_locked_on, sim.state().heroes[0].id);

    actions = {};
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[2].mender_beam_locked_on, 0U);

    actions[2].primary_fire = true;
    sim.step(actions);
    ASSERT_EQ(sim.state().heroes[2].mender_beam_locked_on, sim.state().heroes[0].id);

    actions[2].ability_1 = true;
    sim.step(actions);
    EXPECT_EQ(sim.state().heroes[2].mender_weapon, xushi2::common::MenderWeapon::Sidearm);
    EXPECT_EQ(sim.state().heroes[2].mender_beam_locked_on, 0U);
}

TEST(Combat, MenderTetherSnapsNearAimedAllyAndSetsCooldown) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 1.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 4.0F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].ability_2 = true;
    sim.step(actions);

    const auto& mender = sim.state().heroes[2];
    EXPECT_NEAR(mender.position.x, 2.5F, 1e-5F);
    EXPECT_NEAR(mender.position.y,
                4.0F - xushi2::common::kMenderTetherStopDistance, 1e-5F);
    EXPECT_EQ(mender.cd_ability_2, xushi2::common::kMenderTetherCooldownTicks);
}

TEST(Combat, MenderTetherCooldownPreventsImmediateSecondSnap) {
    MatchConfig cfg = close_arena_config();
    cfg.team_size = 3;
    cfg.hero_kinds = {
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
        xushi2::common::HeroKind::Vanguard,
        xushi2::common::HeroKind::Ranger,
        xushi2::common::HeroKind::Mender,
    };
    Sim sim(cfg);
    auto& state = const_cast<xushi2::sim::MatchState&>(sim.state());
    state.heroes[2].position = xushi2::common::Vec2{2.5F, 1.0F};
    state.heroes[2].aim_angle = 0.5F * xushi2::common::kPi;
    state.heroes[0].position = xushi2::common::Vec2{2.5F, 4.0F};

    std::array<Action, kAgentsPerMatch> actions{};
    actions[2].ability_2 = true;
    sim.step(actions);
    const float after_first = sim.state().heroes[2].position.y;
    auto& mutable_state = const_cast<xushi2::sim::MatchState&>(sim.state());
    mutable_state.heroes[0].position = xushi2::common::Vec2{2.5F, 5.0F};
    sim.step(actions);

    EXPECT_NEAR(sim.state().heroes[2].position.y, after_first, 1e-5F);
    EXPECT_EQ(sim.state().heroes[2].cd_ability_2,
              xushi2::common::kMenderTetherCooldownTicks - 1U);
}
