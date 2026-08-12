"""terminated/truncated semantics and time-limit value bootstrapping.

Two coupled defects lived here.

Labels: the envs derived terminated/truncated from `winner`, so a round that
timed out with one team ahead was reported as *terminated* and only an exact
draw was reported as *truncated*. That inverts the common case -- reaching the
score threshold is rare, timing out is not.

Bootstrapping: GAE gated the V(s_{t+1}) bootstrap on `terminated | truncated`,
so every timeout taught the critic that the final state was worth zero. With
gamma=0.997 that is a systematic negative bias across the end of nearly every
episode.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from envs.phase4_mappo import Phase4MappoEnv
from train.common_advantage import compute_gae_core


def _sim_cfg(round_length_seconds: int = 2) -> dict:
    return {
        "seed": 5,
        "round_length_seconds": round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 120,
        },
    }


# --- the sim reports why it ended ---------------------------------------


def test_sim_distinguishes_time_limit_from_score_threshold():
    env = Phase4MappoEnv(_sim_cfg(round_length_seconds=1), opponent_bot="noop")
    try:
        env.reset(seed=1)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = truncated = False
        for _ in range(200):
            _obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        # A 1-second round with noop bots cannot reach the score threshold, so
        # it must end as a time limit.
        assert truncated is True
        assert terminated is False
        assert env._sim.round_timer_expired is True
        assert env._sim.score_threshold_reached is False
    finally:
        env.close()


def test_timeout_with_a_winner_is_truncated_not_terminated():
    """The case the old `winner`-based derivation got backwards."""
    env = Phase4MappoEnv(_sim_cfg(round_length_seconds=1), opponent_bot="noop")
    try:
        env.reset(seed=2)
        # Hand Team A a lead without reaching the win threshold, so `winner`
        # is Team A while the episode is still ending on the round timer.
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        terminated = truncated = False
        for _ in range(200):
            _obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        assert env._sim.score_threshold_reached is False
        # Regardless of who is ahead, a timer expiry is a truncation.
        assert truncated is True
        assert terminated is False
    finally:
        env.close()


# --- the vector env preserves the terminal critic observation ------------


def test_final_critic_observation_is_the_pre_reset_state():
    from xushi2.obs_manifest import CRITIC_DIM
    from xushi2.vector_env import XushiVectorEnv

    def make():
        return Phase4MappoEnv(_sim_cfg(round_length_seconds=1), opponent_bot="noop")

    vec = XushiVectorEnv([make], critic_obs_dim=CRITIC_DIM, seed_base=3)
    try:
        vec.reset(seed=3)
        actions = np.zeros((1, *vec.single_action_space.shape), dtype=np.float32)
        for _ in range(200):
            _obs, _r, term, trunc, critic_obs, infos = vec.step(actions)
            if term[0] or trunc[0]:
                final_critic = infos[0].get("final_critic_observation")
                assert final_critic is not None, "terminal critic obs was not preserved"
                assert final_critic.shape == (CRITIC_DIM,)
                # critic_obs describes the reset episode; the two must differ,
                # otherwise there was nothing to preserve and the bootstrap
                # would silently use the wrong state.
                assert not np.allclose(final_critic, critic_obs[0])
                break
        else:
            pytest.fail("episode never ended")
    finally:
        vec.close()


# --- GAE bootstraps truncation but not termination ----------------------


def _gae(*, steps: int = 2, **kwargs):
    defaults = dict(
        rewards=torch.zeros(1, steps),
        values=torch.zeros(1, steps),
        dones=torch.zeros(1, steps),
        last_value=torch.zeros(1),
        last_done=torch.zeros(1),
        gamma=0.99,
        gae_lambda=0.95,
    )
    defaults.update(kwargs)
    return compute_gae_core(**defaults)


def test_truncation_keeps_the_bootstrap_and_termination_drops_it():
    # One step, zero reward, V(s)=0, V(s') = 10.
    values = torch.zeros(1, 2)
    # V(s_T) for the episode that ends at step 0.
    next_value = torch.tensor([[10.0, 0.0]])
    dones = torch.tensor([[1.0, 0.0]])  # episode ends after step 0

    terminal_adv, _ = _gae(
        values=values,
        rewards=torch.zeros(1, 2),
        dones=dones,
        terminateds=dones,  # a real terminal
        last_terminated=torch.zeros(1),
    )
    truncated_adv, _ = _gae(
        values=values,
        rewards=torch.zeros(1, 2),
        dones=dones,
        terminateds=torch.zeros(1, 2),  # ended, but only by time limit
        last_terminated=torch.zeros(1),
        truncated_values=next_value,
    )
    # A real terminal is worth 0 going forward.
    assert terminal_adv[0, 0].item() == pytest.approx(0.0)
    # A truncation still bootstraps gamma * V(s_T).
    assert truncated_adv[0, 0].item() == pytest.approx(0.99 * 10.0)


def test_advantage_never_carries_across_an_episode_boundary():
    """Truncation keeps the bootstrap but must still cut the recursive carry.

    Otherwise the next episode's advantage leaks backwards into this one.
    """
    rewards = torch.tensor([[0.0, 100.0]])
    dones = torch.tensor([[1.0, 0.0]])
    adv, _ = _gae(
        rewards=rewards,
        values=torch.zeros(1, 2),
        dones=dones,
        terminateds=torch.zeros(1, 2),
        last_terminated=torch.zeros(1),
    )
    # Step 1's large reward must not reach step 0.
    assert adv[0, 0].item() == pytest.approx(0.0)
    assert adv[0, 1].item() == pytest.approx(100.0)


def test_omitting_terminateds_reproduces_the_conflated_behavior():
    """Back-compat: callers that pass only `dones` get the old semantics."""
    dones = torch.tensor([[1.0, 0.0]])
    conflated, _ = _gae(values=torch.zeros(1, 2), dones=dones)
    explicit, _ = _gae(values=torch.zeros(1, 2), dones=dones, terminateds=dones)
    assert torch.allclose(conflated, explicit)


def test_shape_mismatches_are_rejected():
    with pytest.raises(ValueError, match="terminateds must have shape"):
        _gae(terminateds=torch.zeros(1, 9))
    with pytest.raises(ValueError, match="last_terminated must have shape"):
        _gae(last_terminated=torch.zeros(9))
    with pytest.raises(ValueError, match="truncated_values must match values shape"):
        _gae(truncated_values=torch.zeros(1, 9))


def test_conflated_gating_is_what_the_old_code_did():
    """Pins the exact regression: gating the bootstrap on `done`.

    Under the old semantics a timeout scored the terminal state at zero; the
    fixed path bootstraps gamma * V(s_T) instead.
    """
    dones = torch.tensor([[1.0, 0.0]])
    old, _ = _gae(values=torch.zeros(1, 2), dones=dones)  # terminateds := dones
    new, _ = _gae(
        values=torch.zeros(1, 2),
        dones=dones,
        terminateds=torch.zeros(1, 2),
        truncated_values=torch.tensor([[10.0, 0.0]]),
    )
    assert old[0, 0].item() == pytest.approx(0.0)
    assert new[0, 0].item() == pytest.approx(0.99 * 10.0)
