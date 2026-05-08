"""Tests for xushi2.reward.RewardCalculator.

Covers shaped-event attribution, per-episode cumulative clipping,
terminal rewards, and the reset invariant.
"""

from __future__ import annotations

import pytest
import numpy as np

from xushi2 import xushi2_cpp as _cpp
from xushi2.reward import RewardCalculator


class _FakeSim:
    """Light-weight stand-in for a real Sim. Only the attributes used by
    RewardCalculator are exposed. Lets us test edge cases (arbitrary
    counters, arbitrary winners) without driving a full sim."""

    def __init__(self):
        self.team_a_score_ticks = 0
        self.team_b_score_ticks = 0
        self.team_a_kills = 0
        self.team_b_kills = 0
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
    assert rc._obs_buf_a is None
    assert rc._obs_buf_b is None
    assert rc._pos_slice is None


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
        "move_x": 0.0, "move_y": 1.0, "aim_delta": 0.0,
        "primary_fire": 0, "ability_1": 0, "ability_2": 0,
    }

    # Baseline run: no distance shaping.
    env_off = XushiEnv(sim_cfg, opponent_bot="noop", reward_cfg={})
    env_off.reset(seed=0xD1CEDA7A)
    _, r_off, *_ = env_off.step(action)

    # Shaped run: distance_shaping_coef > 0.
    env_on = XushiEnv(
        sim_cfg, opponent_bot="noop",
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
