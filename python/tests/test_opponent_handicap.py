"""Tests for the opponent-handicap curriculum: schedule math, env
application (bot-targeted, deterministic), and validation."""

from __future__ import annotations

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from train.mappo_model import compute_opponent_handicap


def _sim_cfg() -> dict:
    return {
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


def _noop_actions(env) -> np.ndarray:
    return np.zeros((env.n_agents, env.action_dim), dtype=np.float32)


def _run_and_collect_opponent_fire(env, steps: int = 40) -> list[float]:
    env.reset(seed=3)
    fires: list[float] = []
    for _ in range(steps):
        _obs, _r, term, trunc, info = env.step(_noop_actions(env))
        opp = info.get("opponent_actions")
        if opp is not None:
            fires.extend(np.asarray(opp)[:, 3].tolist())
        if term or trunc:
            break
    return fires


# --- schedule math ------------------------------------------------------


def test_handicap_disabled_holds_initial():
    assert compute_opponent_handicap(
        update=999,
        initial_aim_noise=1.5,
        final_aim_noise=0.0,
        initial_fire_cadence=60,
        final_fire_cadence=1,
        anneal_updates=0,
    ) == (1.5, 60)


def test_handicap_midpoint_linear():
    noise, cadence = compute_opponent_handicap(
        update=150,
        initial_aim_noise=1.5,
        final_aim_noise=0.0,
        initial_fire_cadence=60,
        final_fire_cadence=1,
        anneal_updates=300,
    )
    assert noise == pytest.approx(0.75)
    assert cadence == 30 or cadence == 31


def test_handicap_reaches_full_strength_and_holds():
    assert compute_opponent_handicap(
        update=300,
        initial_aim_noise=1.5,
        final_aim_noise=0.0,
        initial_fire_cadence=60,
        final_fire_cadence=1,
        anneal_updates=300,
    ) == (0.0, 1)
    assert compute_opponent_handicap(
        update=999,
        initial_aim_noise=1.5,
        final_aim_noise=0.0,
        initial_fire_cadence=60,
        final_fire_cadence=1,
        anneal_updates=300,
    ) == (0.0, 1)


# --- env application ----------------------------------------------------


def test_fire_cadence_gates_opponent_fire():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="hold_and_shoot")
    baseline = _run_and_collect_opponent_fire(env)
    env.set_opponent_handicap("hold_and_shoot", 0.0, 1_000_000)
    gated = _run_and_collect_opponent_fire(env)
    env.close()
    assert sum(gated) < sum(baseline)
    assert sum(gated) <= len(gated) * 0.05


def test_handicap_ignores_other_bots():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="hold_and_shoot")
    baseline = _run_and_collect_opponent_fire(env)
    env.set_opponent_handicap("basic", 0.0, 1_000_000)
    unaffected = _run_and_collect_opponent_fire(env)
    env.close()
    assert sum(unaffected) == pytest.approx(sum(baseline))


def test_handicap_is_deterministic():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="hold_and_shoot")
    env.set_opponent_handicap("hold_and_shoot", 1.5, 60)
    a = _run_and_collect_opponent_fire(env)
    b = _run_and_collect_opponent_fire(env)
    env.close()
    assert a == b


def test_handicap_validation():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="hold_and_shoot")
    with pytest.raises(ValueError, match="unknown opponent_bot"):
        env.set_opponent_handicap("not_a_bot", 0.5, 10)
    with pytest.raises(ValueError, match="aim_noise_radians"):
        env.set_opponent_handicap("basic", -0.1, 10)
    with pytest.raises(ValueError, match="fire_cadence_ticks"):
        env.set_opponent_handicap("basic", 0.5, 0)
    env.close()
