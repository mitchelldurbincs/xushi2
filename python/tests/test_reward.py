"""Tests for xushi2.reward.RewardCalculator.

Covers shaped-event attribution, per-episode cumulative clipping,
terminal rewards, and the reset invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from xushi2 import xushi2_cpp as _cpp
from xushi2.reward import RewardCalculator
from xushi2.reward_components import CumulativeClipper


class _FakeSim:
    """Light-weight stand-in for a real Sim. Only the attributes used by
    RewardCalculator are exposed. Lets us test edge cases (arbitrary
    counters, arbitrary winners) without driving a full sim."""

    def __init__(self):
        self.tick = 0
        self.team_a_score_ticks = 0
        self.team_b_score_ticks = 0
        self.team_a_kills = 0
        self.team_b_kills = 0
        self.kills_by_slot = [0, 0, 0, 0, 0, 0]
        self.deaths_by_slot = [0, 0, 0, 0, 0, 0]
        self.damage_dealt_by_slot = [0, 0, 0, 0, 0, 0]
        self.on_point_by_slot = [0, 0, 0, 0, 0, 0]
        self.episode_over = False
        self.winner = _cpp.Team.Neutral


def _fresh_calc_and_sim():
    rc = RewardCalculator()
    sim = _FakeSim()
    rc.reset(sim)
    return rc, sim


# --- shaped reward -----------------------------------------------------


def test_no_deltas_yields_zero_reward():
    rc, sim = _fresh_calc_and_sim()
    a, b = rc.step(sim)
    assert a == 0.0
    assert b == 0.0


def test_team_a_kill_rewards_a_and_penalizes_b():
    rc, sim = _fresh_calc_and_sim()
    sim.team_a_kills = 1
    a, b = rc.step(sim)
    # Raw delta = +0.25 for A; -0.25 for B.
    assert a == pytest.approx(0.25)
    assert b == pytest.approx(-0.25)


def test_team_a_scoring_rewards_a_and_penalizes_b():
    rc, sim = _fresh_calc_and_sim()
    # 30 score ticks = 1 second of scoring = +0.01 to A at default rate.
    sim.team_a_score_ticks = _cpp.TICK_HZ
    a, b = rc.step(sim)
    assert a == pytest.approx(0.01)
    assert b == pytest.approx(-0.01)


def test_per_step_reward_is_zero_sum_under_cap():
    rc, sim = _fresh_calc_and_sim()
    sim.team_a_kills = 2
    sim.team_b_kills = 1
    a, b = rc.step(sim)
    assert a + b == pytest.approx(0.0, abs=1e-9)


# --- cumulative clip ----------------------------------------------------


def test_cumulative_clip_caps_team_a_at_positive_three():
    rc = RewardCalculator()  # default clip = 3.0
    sim = _FakeSim()
    rc.reset(sim)

    # Feed repeated kills: 20 kills -> raw +5 but should cap at +3.
    total_a = 0.0
    for k in range(1, 21):
        sim.team_a_kills = k
        a, _ = rc.step(sim)
        total_a += a
    assert total_a == pytest.approx(3.0)
    assert rc.cumulative_shaped_a == pytest.approx(3.0)


def test_cumulative_clip_caps_team_a_at_negative_three():
    rc = RewardCalculator()
    sim = _FakeSim()
    rc.reset(sim)

    total_a = 0.0
    for k in range(1, 21):
        sim.team_b_kills = k  # A "dies" repeatedly
        a, _ = rc.step(sim)
        total_a += a
    assert total_a == pytest.approx(-3.0)


def test_reset_zeroes_cumulative():
    rc = RewardCalculator()
    sim = _FakeSim()
    rc.reset(sim)
    for k in range(1, 21):
        sim.team_a_kills = k
        rc.step(sim)
    assert rc.cumulative_shaped_a == pytest.approx(3.0)

    # Start new episode with fresh sim. cumulative must reset.
    sim2 = _FakeSim()
    rc.reset(sim2)
    assert rc.cumulative_shaped_a == 0.0
    assert rc.cumulative_shaped_b == 0.0


# --- terminal rewards ---------------------------------------------------


def test_terminal_win_is_plus_ten_for_winner():
    rc, sim = _fresh_calc_and_sim()
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    a, b = rc.add_terminal(sim)
    assert a == 10.0
    assert b == -10.0


def test_terminal_draw_is_zero():
    rc, sim = _fresh_calc_and_sim()
    sim.episode_over = True
    sim.winner = _cpp.Team.Neutral
    a, b = rc.add_terminal(sim)
    assert a == 0.0
    assert b == 0.0


def test_terminal_before_episode_over_raises():
    rc, sim = _fresh_calc_and_sim()
    sim.episode_over = False
    with pytest.raises(RuntimeError):
        rc.add_terminal(sim)


def test_terminal_not_clipped_even_after_capped_shaping():
    rc = RewardCalculator()
    sim = _FakeSim()
    rc.reset(sim)
    # Hit the shaped cap first.
    for k in range(1, 21):
        sim.team_a_kills = k
        rc.step(sim)
    # Terminal reward is +10, not reduced by the shaping cap.
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    a, b = rc.add_terminal(sim)
    assert a == 10.0
    assert b == -10.0


# --- configuration ------------------------------------------------------


def test_zero_shaping_clip_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(shaping_clip=0.0)


def test_custom_kill_bonus_applies():
    rc = RewardCalculator(kill_bonus=1.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    a, _ = rc.step(sim)
    assert a == pytest.approx(1.0)


def test_custom_score_per_second_applies():
    rc = RewardCalculator(score_per_second=0.1)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_score_ticks = _cpp.TICK_HZ
    a, b = rc.step(sim)
    assert a == pytest.approx(0.1)
    assert b == pytest.approx(-0.1)


def test_negative_distance_shaping_coef_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(distance_shaping_coef=-0.01)


def test_negative_on_point_shaping_coef_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(on_point_shaping_coef=-0.01)


def test_default_time_penalty_is_zero_and_does_not_change_rewards():
    """Backwards compatibility: omitting time_penalty_per_second leaves
    every existing reward path unchanged."""
    rc, sim = _fresh_calc_and_sim()
    sim.team_a_kills = 1
    a, b = rc.step(sim)
    assert a == pytest.approx(0.25)
    assert b == pytest.approx(-0.25)


def test_time_penalty_charges_both_teams_per_tick_with_no_events():
    """A non-zero time_penalty_per_second subtracts the same per-tick
    amount from both teams when nothing else is happening — breaking
    zero-sum on purpose so deny-only stalemates have a negative return."""
    tps = 0.05
    rc = RewardCalculator(time_penalty_per_second=tps)
    sim = _FakeSim()
    rc.reset(sim)

    expected_per_tick = -tps / float(_cpp.TICK_HZ)
    total_a = 0.0
    total_b = 0.0
    for _ in range(_cpp.TICK_HZ):  # 1 second of no-event ticks
        a, b = rc.step(sim)
        total_a += a
        total_b += b

    assert total_a == pytest.approx(expected_per_tick * _cpp.TICK_HZ)
    assert total_b == pytest.approx(expected_per_tick * _cpp.TICK_HZ)
    # Both teams charged equally — explicitly NOT zero-sum.
    assert total_a == pytest.approx(total_b)


def test_time_penalty_stacks_with_zero_sum_shaping():
    """When events occur, time penalty adds on top of zero-sum shaping:
    raw_a = (own - enemy events) - tp; raw_b = (enemy - own events) - tp."""
    tps = 0.06
    rc = RewardCalculator(time_penalty_per_second=tps)
    sim = _FakeSim()
    rc.reset(sim)

    # Single tick, A kills 1, no scoring.
    sim.team_a_kills = 1
    a, b = rc.step(sim)
    tp_step = -tps / float(_cpp.TICK_HZ)
    assert a == pytest.approx(0.25 + tp_step)
    assert b == pytest.approx(-0.25 + tp_step)


def test_default_distance_shaping_coef_is_zero_and_no_buffer_allocated():
    rc = RewardCalculator()  # omits distance_shaping_coef entirely
    # With coef=0, the calculator should not allocate the per-team obs
    # buffers (we only pay that cost when the shaping is opted-in).
    assert rc._obs.obs_buf_a is None
    assert rc._obs.obs_buf_b is None
    assert rc._obs.pos_slice is None


def test_distance_shaping_produces_nonzero_reward_on_real_env():
    """With coef > 0, stepping a real env yields a per-decision shaping term
    even when no score/kill events occur. This is the smoke test that
    build_actor_obs wiring inside reward.step works end-to-end."""
    from xushi2.env import XushiEnv

    sim_cfg = {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 5,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }
    action = {
        "move_x": 0.0,
        "move_y": 1.0,
        "aim_delta": 0.0,
        "primary_fire": 0,
        "ability_1": 0,
        "ability_2": 0,
    }

    # Baseline run: no distance shaping.
    env_off = XushiEnv(sim_cfg, opponent_bot="noop", reward_cfg={})
    env_off.reset(seed=0xD1CEDA7A)
    _, r_off, *_ = env_off.step(action)

    # Shaped run: distance_shaping_coef > 0.
    env_on = XushiEnv(
        sim_cfg,
        opponent_bot="noop",
        reward_cfg={"distance_shaping_coef": 0.01},
    )
    env_on.reset(seed=0xD1CEDA7A)
    _, r_on, *_ = env_on.step(action)

    # A moves toward cap (upward), B is noop at its spawn. dist_A should be
    # slightly less than the spawn distance, dist_B unchanged ≈ 0.80.
    # Per-step shaping = -coef*(dist_A - dist_B) > 0. Agent's step reward
    # with shaping should be strictly greater than without.
    assert r_on > r_off
    assert r_off == pytest.approx(0.0, abs=1e-9)


# --- team_spirit mixin (per-agent path only) ---------------------------


def test_team_spirit_zero_preserves_individual_credit():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # τ=0: pure individual; only the killer gets the kill bonus.
    assert a[0] == pytest.approx(0.25)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)


def test_team_spirit_one_collapses_to_team_mean():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=1.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # τ=1: every slot receives the team mean = 0.25 / 3.
    expected = 0.25 / 3.0
    assert a[0] == pytest.approx(expected)
    assert a[1] == pytest.approx(expected)
    assert a[2] == pytest.approx(expected)
    assert a.sum() == pytest.approx(0.25)


def test_team_spirit_half_is_exact_interpolation():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.5)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # indiv = [0.25, 0, 0]; mean = 0.25/3.
    # mixed = 0.5 * indiv + 0.5 * mean.
    mean = 0.25 / 3.0
    assert a[0] == pytest.approx(0.5 * 0.25 + 0.5 * mean)
    assert a[1] == pytest.approx(0.5 * mean)
    assert a[2] == pytest.approx(0.5 * mean)
    # Sum invariant: team total still 0.25.
    assert a.sum() == pytest.approx(0.25)


def test_team_spirit_does_not_mix_terminal():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=1.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    ta, tb = rc.add_terminal(sim)
    # Terminal is uniform regardless of team_spirit (it's already a team
    # outcome by construction).
    np.testing.assert_array_equal(ta, np.full(3, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(tb, np.full(3, -10.0, dtype=np.float32))


def test_team_spirit_setter_updates_in_place():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.0)
    sim = _FakeSim()
    rc.reset(sim)
    rc.set_team_spirit(1.0)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    a, _ = rc.step(sim)
    # After setter, behavior is τ=1: uniform team mean.
    expected = 0.25 / 3.0
    assert a[0] == pytest.approx(expected)
    assert a[1] == pytest.approx(expected)
    assert a[2] == pytest.approx(expected)


def test_team_spirit_out_of_range_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(team_spirit=-0.01)
    with pytest.raises(ValueError):
        RewardCalculator(team_spirit=1.01)
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.5)
    with pytest.raises(ValueError):
        rc.set_team_spirit(2.0)


def test_team_spirit_no_op_on_scalar_path():
    """team_spirit on a scalar-mode calculator is silently a no-op:
    the kwarg is accepted (so configs don't break) but doesn't change
    the scalar reward."""
    rc = RewardCalculator(team_spirit=1.0)  # per_agent_rewards=False
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    a, b = rc.step(sim)
    # Scalar mode unchanged.
    assert isinstance(a, float)
    assert a == pytest.approx(0.25)
    assert b == pytest.approx(-0.25)


# --- damage-dealt shaping (per-agent, opt-in) --------------------------


def test_damage_dealt_default_zero_no_op():
    """Default damage_dealt_coef=0 produces no damage-related reward."""
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.damage_dealt_by_slot = [7500, 0, 0, 0, 0, 0]  # 75 HP applied by slot 0
    a, b = rc.step(sim)
    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))


def test_damage_dealt_credits_attacker_slot_per_hp():
    """coef=0.001 per HP: a 75-HP shot from slot 0 → +0.075 to slot 0."""
    rc = RewardCalculator(per_agent_rewards=True, damage_dealt_coef=0.001)
    sim = _FakeSim()
    rc.reset(sim)
    sim.damage_dealt_by_slot = [7500, 0, 0, 0, 0, 0]  # 75 HP * 100 cHP/HP
    a, b = rc.step(sim)
    assert a[0] == pytest.approx(0.075)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))


def test_damage_dealt_diffs_against_prev():
    """Subsequent shots only credit the *delta* in cumulative damage."""
    rc = RewardCalculator(per_agent_rewards=True, damage_dealt_coef=0.001)
    sim = _FakeSim()
    rc.reset(sim)
    sim.damage_dealt_by_slot = [7500, 0, 0, 0, 0, 0]
    rc.step(sim)
    # Second shot: another 75 HP applied — total cumulative now 150 HP.
    sim.damage_dealt_by_slot = [15000, 0, 0, 0, 0, 0]
    a, _ = rc.step(sim)
    # Step reward should reflect just the 75 HP delta, not the 150 total.
    assert a[0] == pytest.approx(0.075)


def test_damage_dealt_independent_per_team():
    """B's slot 3 dealing damage credits team B; doesn't affect team A."""
    rc = RewardCalculator(per_agent_rewards=True, damage_dealt_coef=0.001)
    sim = _FakeSim()
    rc.reset(sim)
    sim.damage_dealt_by_slot = [0, 0, 0, 7500, 0, 0]  # absolute slot 3 = team B local 0
    a, b = rc.step(sim)
    # Team A: no damage dealt → no per-damage reward.
    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    # Team B: slot 3 (local 0) gets the credit.
    assert b[0] == pytest.approx(0.075)
    assert b[1] == pytest.approx(0.0)
    assert b[2] == pytest.approx(0.0)


def test_damage_dealt_negative_coef_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(damage_dealt_coef=-0.01)


def test_damage_dealt_no_op_on_scalar_path():
    """damage_dealt_coef has no effect on the scalar (default) path —
    that path doesn't read damage_dealt_by_slot."""
    rc = RewardCalculator(damage_dealt_coef=0.001)  # per_agent_rewards=False
    sim = _FakeSim()
    rc.reset(sim)
    sim.damage_dealt_by_slot = [7500, 0, 0, 0, 0, 0]
    a, b = rc.step(sim)
    assert isinstance(a, float)
    assert a == pytest.approx(0.0)
    assert b == pytest.approx(0.0)


# --- per-agent rewards (opt-in flag) -----------------------------------


def test_per_agent_default_false_returns_scalars():
    """Default-false flag preserves today's scalar contract."""
    rc = RewardCalculator()
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    a, b = rc.step(sim)
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert a == pytest.approx(0.25)
    assert b == pytest.approx(-0.25)


def test_per_agent_kill_credits_only_killer_and_victim():
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, b = rc.step(sim)
    assert a.shape == (3,)
    assert b.shape == (3,)
    # Only the killer (team A slot 0) gets the kill bonus.
    assert a[0] == pytest.approx(0.25)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)
    # Only the victim (team B local slot 0 == absolute slot 3) gets the
    # death penalty.
    assert b[0] == pytest.approx(-0.25)
    assert b[1] == pytest.approx(0.0)
    assert b[2] == pytest.approx(0.0)
    # Sum invariants (kill_bonus == death_penalty default).
    assert a.sum() == pytest.approx(0.25)
    assert b.sum() == pytest.approx(-0.25)
    assert a.sum() + b.sum() == pytest.approx(0.0)


def test_per_agent_score_split_uniformly_when_no_on_point_data():
    """FakeSim has no obs path; per-agent score split falls back to uniform
    1/3 each. Enemy-score subtraction is also uniform 1/3."""
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_score_ticks = _cpp.TICK_HZ  # 1 second of A scoring
    a, b = rc.step(sim)
    np.testing.assert_allclose(a, [0.01 / 3, 0.01 / 3, 0.01 / 3], atol=1e-6)
    np.testing.assert_allclose(b, [-0.01 / 3, -0.01 / 3, -0.01 / 3], atol=1e-6)
    assert a.sum() == pytest.approx(0.01)
    assert b.sum() == pytest.approx(-0.01)


def test_per_agent_sum_matches_scalar_path_for_kill_event():
    """When kill_bonus == death_penalty (default), per-agent team sums
    equal the scalar-path scalar for the same event."""
    sim_a = _FakeSim()
    sim_a.team_a_kills = 1
    sim_a.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim_a.deaths_by_slot = [0, 0, 0, 1, 0, 0]

    rc_scalar = RewardCalculator()
    rc_scalar.reset(_FakeSim())
    a_scalar, b_scalar = rc_scalar.step(sim_a)

    rc_vec = RewardCalculator(per_agent_rewards=True)
    rc_vec.reset(_FakeSim())
    a_vec, b_vec = rc_vec.step(sim_a)

    assert a_vec.sum() == pytest.approx(a_scalar)
    assert b_vec.sum() == pytest.approx(b_scalar)


def test_per_agent_terminal_returns_uniform_arrays():
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    ta, tb = rc.add_terminal(sim)
    assert ta.shape == (3,)
    assert tb.shape == (3,)
    np.testing.assert_array_equal(ta, np.full(3, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(tb, np.full(3, -10.0, dtype=np.float32))


def test_per_agent_clip_on_team_sum_preserves_today_invariant():
    """Cumulative team-sum cap is the same ±3 as the scalar path."""
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)

    total_team_a = 0.0
    for k in range(1, 21):
        sim.team_a_kills = k
        sim.kills_by_slot = [k, 0, 0, 0, 0, 0]
        a, _ = rc.step(sim)
        total_team_a += float(a.sum())
    assert total_team_a == pytest.approx(3.0)
    assert rc.cumulative_shaped_a == pytest.approx(3.0)


def test_on_point_shaping_rewards_phase4_objective_contact():
    from envs.phase4_mappo import Phase4MappoEnv
    from xushi2.obs_manifest import actor_field_slice

    sim_cfg = {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 30,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }
    env = Phase4MappoEnv(
        sim_cfg,
        opponent_bot="noop",
        reward_cfg={"on_point_shaping_coef": 0.02},
    )
    obs, _ = env.reset(seed=0)
    pos_slice = actor_field_slice("own_position")
    total_reward = 0.0
    try:
        for _ in range(220):
            own_pos = obs[:, pos_slice]
            move = -own_pos.copy()
            norm = (move[:, :1] ** 2 + move[:, 1:2] ** 2) ** 0.5
            move = move / norm.clip(min=1e-6)
            action = np.zeros((3, 6), dtype=np.float32)
            action[:, :2] = move.astype(np.float32)
            obs, reward, term, trunc, _info = env.step(action)
            total_reward += float(reward[0])
            if term or trunc:
                break
    finally:
        env.close()

    assert total_reward > 0.0

class _LegacyScalarRewardCalculator:
    """Frozen pre-refactor scalar reward path for regression parity checks."""

    def __init__(self):
        self._score_per_second = 0.01
        self._kill_bonus = 0.25
        self._death_penalty = 0.25
        self._time_penalty_per_second = 0.05
        self._prev = _FakeSim()
        self._clip_a = 0.0
        self._clip_b = 0.0

    def reset(self, sim):
        self._prev = _FakeSim()
        self._prev.team_a_score_ticks = sim.team_a_score_ticks
        self._prev.team_b_score_ticks = sim.team_b_score_ticks
        self._prev.team_a_kills = sim.team_a_kills
        self._prev.team_b_kills = sim.team_b_kills
        self._clip_a = 0.0
        self._clip_b = 0.0

    def _apply(self, raw, team):
        old = self._clip_a if team == "a" else self._clip_b
        new = min(3.0, max(-3.0, old + raw))
        if team == "a":
            self._clip_a = new
        else:
            self._clip_b = new
        return new - old

    def step(self, sim):
        a_s = (sim.team_a_score_ticks - self._prev.team_a_score_ticks) / float(_cpp.TICK_HZ)
        b_s = (sim.team_b_score_ticks - self._prev.team_b_score_ticks) / float(_cpp.TICK_HZ)
        a_k = sim.team_a_kills - self._prev.team_a_kills
        b_k = sim.team_b_kills - self._prev.team_b_kills
        raw_a = self._score_per_second * a_s - self._score_per_second * b_s + self._kill_bonus * a_k - self._death_penalty * b_k
        raw_b = -raw_a
        tp = -self._time_penalty_per_second / float(_cpp.TICK_HZ)
        raw_a += tp
        raw_b += tp
        ra = self._apply(raw_a, "a")
        rb = self._apply(raw_b, "b")
        self._prev.team_a_score_ticks = sim.team_a_score_ticks
        self._prev.team_b_score_ticks = sim.team_b_score_ticks
        self._prev.team_a_kills = sim.team_a_kills
        self._prev.team_b_kills = sim.team_b_kills
        return ra, rb


def test_cumulative_clipper_isolated_behavior():
    clip = CumulativeClipper(1.0)
    assert clip.apply_clip(0.7, "a") == pytest.approx(0.7)
    assert clip.apply_clip(0.7, "a") == pytest.approx(0.3)
    assert clip.apply_clip(-3.0, "a") == pytest.approx(-2.0)
    assert clip.cumulative_shaped_a == pytest.approx(-1.0)


def test_regression_scalar_old_vs_new_on_synthetic_trajectory():
    sim = _FakeSim()
    new = RewardCalculator(time_penalty_per_second=0.05)
    old = _LegacyScalarRewardCalculator()
    new.reset(sim)
    old.reset(sim)

    traj = [
        (0, 0, 0, 0),
        (_cpp.TICK_HZ // 2, 0, 1, 0),
        (_cpp.TICK_HZ, _cpp.TICK_HZ // 3, 1, 1),
        (_cpp.TICK_HZ, _cpp.TICK_HZ, 2, 1),
        (_cpp.TICK_HZ * 2, _cpp.TICK_HZ, 2, 3),
    ]
    for a_ticks, b_ticks, a_k, b_k in traj:
        sim.team_a_score_ticks = a_ticks
        sim.team_b_score_ticks = b_ticks
        sim.team_a_kills = a_k
        sim.team_b_kills = b_k
        assert new.step(sim) == pytest.approx(old.step(sim))

# --- majority-on-point shaping (opt-in curriculum) ---------------------


def test_majority_on_point_rewards_team_a_when_two_vs_one_on_point():
    rc = RewardCalculator(per_agent_rewards=True, majority_on_point_coef=0.3)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 1, 0, 1, 0, 0]

    a, b = rc.step(sim)

    assert a.sum() == pytest.approx(0.3)
    assert b.sum() == pytest.approx(-0.3)
    np.testing.assert_allclose(a, [0.15, 0.15, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(b, [-0.1, -0.1, -0.1], atol=1.0e-6)
    assert rc.majority_on_point_metrics()["majority_on_point_reward_a"] == pytest.approx(0.3)


def test_majority_on_point_one_vs_one_is_zero():
    rc = RewardCalculator(per_agent_rewards=True, majority_on_point_coef=0.3)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 0, 0, 1, 0, 0]

    a, b = rc.step(sim)

    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))
    metrics = rc.majority_on_point_metrics()
    assert metrics["majority_on_point_advantage_a"] == pytest.approx(0.0)
    assert metrics["majority_on_point_advantage_b"] == pytest.approx(0.0)


def test_majority_on_point_rewards_team_b_when_one_vs_two_on_point():
    rc = RewardCalculator(per_agent_rewards=True, majority_on_point_coef=0.3)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 0, 0, 1, 1, 0]

    a, b = rc.step(sim)

    assert a.sum() == pytest.approx(-0.3)
    assert b.sum() == pytest.approx(0.3)
    np.testing.assert_allclose(a, [-0.1, -0.1, -0.1], atol=1.0e-6)
    np.testing.assert_allclose(b, [0.15, 0.15, 0.0], atol=1.0e-6)
    assert rc.majority_on_point_metrics()["majority_on_point_reward_b"] == pytest.approx(0.3)


def test_majority_on_point_setter_can_disable_term_for_real_reward_eval():
    rc = RewardCalculator(per_agent_rewards=True, majority_on_point_coef=0.3)
    sim = _FakeSim()
    rc.reset(sim)
    rc.set_majority_on_point_alpha(0.0)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 1, 0, 1, 0, 0]

    a, b = rc.step(sim)

    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))
    assert rc.majority_on_point_metrics()["majority_on_point_alpha"] == pytest.approx(0.0)


# --- uncontested-on-point shaping (opt-in hold/capture pressure) --------


def test_uncontested_on_point_rewards_team_a_when_enemy_absent():
    rc = RewardCalculator(per_agent_rewards=True, uncontested_on_point_coef=0.6)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 0, 0, 0, 0, 0]

    a, b = rc.step(sim)

    assert a.sum() == pytest.approx(0.6)
    assert b.sum() == pytest.approx(-0.6)
    np.testing.assert_allclose(a, [0.6, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(b, [-0.2, -0.2, -0.2], atol=1.0e-6)
    metrics = rc.uncontested_on_point_metrics()
    assert metrics["uncontested_on_point_reward_a"] == pytest.approx(0.6)
    assert metrics["uncontested_on_point_reward_b"] == pytest.approx(0.0)


def test_uncontested_on_point_is_zero_while_contested():
    rc = RewardCalculator(per_agent_rewards=True, uncontested_on_point_coef=0.6)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 0, 0, 1, 0, 0]

    a, b = rc.step(sim)

    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))
    metrics = rc.uncontested_on_point_metrics()
    assert metrics["uncontested_on_point_reward_a"] == pytest.approx(0.0)
    assert metrics["uncontested_on_point_reward_b"] == pytest.approx(0.0)


def test_uncontested_on_point_setter_can_disable_term_for_real_reward_eval():
    rc = RewardCalculator(per_agent_rewards=True, uncontested_on_point_coef=0.6)
    sim = _FakeSim()
    rc.reset(sim)
    rc.set_uncontested_on_point_alpha(0.0)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [1, 0, 0, 0, 0, 0]

    a, b = rc.step(sim)

    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(b, np.zeros(3, dtype=np.float32))
    assert rc.uncontested_on_point_metrics()[
        "uncontested_on_point_alpha"
    ] == pytest.approx(0.0)


def test_uncontested_on_point_negative_coef_rejected():
    with pytest.raises(ValueError):
        RewardCalculator(uncontested_on_point_coef=-0.01)


def test_uncontested_on_point_scalar_path_is_symmetric():
    rc = RewardCalculator(uncontested_on_point_coef=0.6)
    sim = _FakeSim()
    rc.reset(sim)
    sim.tick = _cpp.TICK_HZ
    sim.on_point_by_slot = [0, 0, 0, 1, 0, 0]

    a, b = rc.step(sim)

    assert a == pytest.approx(-0.6)
    assert b == pytest.approx(0.6)
