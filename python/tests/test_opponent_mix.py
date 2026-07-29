"""Tests for the opponent-mix curriculum: assignment math, config validation,
the env setter's reset-time contract, wrapper delegation, and vector-env
propagation."""

from __future__ import annotations

import pytest

from envs.phase4_mappo import Phase4MappoEnv
from train.opponent_mix import opponent_mix_assignment, parse_opponent_bot_mix
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.vector_env import XushiVectorEnv


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


def _env_fn():
    return Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="weak_basic_v2")


# --- assignment math ----------------------------------------------------


def test_assignment_proportions_16_envs():
    assignment = opponent_mix_assignment({"weak_basic_v2": 0.9, "noop": 0.1}, 16)
    assert len(assignment) == 16
    assert assignment.count("noop") in (1, 2)
    assert assignment.count("weak_basic_v2") in (14, 15)


def test_assignment_is_deterministic():
    mix = {"weak_basic_v2": 0.75, "noop": 0.25}
    assert opponent_mix_assignment(mix, 12) == opponent_mix_assignment(mix, 12)


def test_assignment_interleaves_minority_bot():
    assignment = opponent_mix_assignment({"weak_basic_v2": 0.75, "noop": 0.25}, 16)
    noop_slots = [i for i, bot in enumerate(assignment) if bot == "noop"]
    assert len(noop_slots) == 4
    # Not bunched at either end: consecutive noop slots are spread out.
    gaps = [b - a for a, b in zip(noop_slots, noop_slots[1:])]
    assert all(gap >= 2 for gap in gaps)


def test_assignment_single_bot():
    assert opponent_mix_assignment({"noop": 1.0}, 3) == ["noop", "noop", "noop"]


def test_assignment_rejects_empty_mix():
    with pytest.raises(ValueError):
        opponent_mix_assignment({}, 4)


def test_assignment_rejects_nonpositive_envs():
    with pytest.raises(ValueError):
        opponent_mix_assignment({"noop": 1.0}, 0)


# --- config parsing -----------------------------------------------------


def test_parse_mix_absent_is_disabled():
    assert parse_opponent_bot_mix(None) == {}


def test_parse_mix_rejects_unknown_bot():
    with pytest.raises(ValueError, match="unknown bot"):
        parse_opponent_bot_mix({"not_a_bot": 1.0})


def test_parse_mix_rejects_nonpositive_weight():
    with pytest.raises(ValueError, match="must be > 0"):
        parse_opponent_bot_mix({"noop": 0.0})


# --- env setter: reset-time contract ------------------------------------


def test_set_opponent_bot_applies_at_next_reset():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="weak_basic_v2")
    env.reset(seed=1)
    env.set_opponent_bot("noop")
    # In-flight episode keeps its opponent.
    assert env._opponent_bot == "weak_basic_v2"
    env.reset(seed=2)
    assert env._opponent_bot == "noop"
    env.close()


def test_set_opponent_bot_rejects_unknown_bot():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="weak_basic_v2")
    with pytest.raises(ValueError, match="unknown opponent_bot"):
        env.set_opponent_bot("not_a_bot")
    env.close()


# --- vector-env propagation ---------------------------------------------


def test_sync_vector_env_assigns_per_env_bots():
    vec = XushiVectorEnv(
        [_env_fn for _ in range(4)],
        critic_obs_dim=CRITIC_DIM,
        seed_base=7,
        auto_reset=True,
    )
    try:
        vec.set_opponent_bots(["weak_basic_v2", "noop", "weak_basic_v2", "noop"])
        vec.reset(seed=7)
        bots = [env._opponent_bot for env in vec.envs]
        assert bots == ["weak_basic_v2", "noop", "weak_basic_v2", "noop"]
    finally:
        vec.close()


def test_sync_vector_env_rejects_wrong_length():
    vec = XushiVectorEnv(
        [_env_fn for _ in range(3)],
        critic_obs_dim=CRITIC_DIM,
        seed_base=7,
        auto_reset=True,
    )
    try:
        with pytest.raises(ValueError, match="one entry per env"):
            vec.set_opponent_bots(["noop"])
    finally:
        vec.close()
