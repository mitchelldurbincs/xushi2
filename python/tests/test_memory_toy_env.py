import numpy as np
import pytest
from envs.memory_toy import MemoryToyEnv


def test_reset_determinism_same_seed_same_target():
    env1 = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    env2 = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_allclose(obs1, obs2)


def test_cue_visible_during_window_hidden_after():
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    obs, _ = env.reset(seed=0)
    assert obs[2] == 1.0  # visible_flag at t=0
    # Target is on unit circle -> norm ~= 1
    assert abs(np.linalg.norm(obs[:2]) - 1.0) < 1e-5

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    for t in range(1, 4):  # ticks 1, 2, 3 -- still visible
        obs, _, _, _, _ = env.step(zero_action)
        assert obs[2] == 1.0

    obs, _, _, _, _ = env.step(zero_action)  # tick 4 -- now hidden
    assert obs[2] == 0.0
    np.testing.assert_allclose(obs[:2], [0.0, 0.0])


def test_terminal_reward_matches_optimal_action():
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    obs, _ = env.reset(seed=123)
    target = obs[:2].copy()  # observed at t=0
    zero = np.array([0.0, 0.0], dtype=np.float32)
    terminal_reward = None
    for t in range(64):  # steps 1..64
        # At t == episode_length - 1 (= 63), act optimally
        action = target if t == 63 else zero
        _, r, term, trunc, _ = env.step(action)
        if term or trunc:
            terminal_reward = r
            break
    assert terminal_reward is not None
    assert abs(terminal_reward) < 1e-5  # exact match -> reward 0


def test_feedforward_baseline_action_yields_expected_reward():
    # Feedforward at terminal tick sees (0,0,0). Best guess is (0,0).
    # Expected terminal reward = -||target|| = -1.
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    env.reset(seed=7)
    zero = np.array([0.0, 0.0], dtype=np.float32)
    terminal_reward = None
    for _ in range(64):
        _, r, term, trunc, _ = env.step(zero)
        if term or trunc:
            terminal_reward = r
            break
    assert abs(terminal_reward - (-1.0)) < 1e-4


def test_episode_length_exact():
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    env.reset(seed=0)
    zero = np.array([0.0, 0.0], dtype=np.float32)
    for t in range(63):
        _, _, term, trunc, _ = env.step(zero)
        assert not (term or trunc), f"premature termination at t={t}"
    _, _, term, trunc, _ = env.step(zero)
    assert term and not trunc  # terminates at tick 63 (T-1)


def test_step_before_reset_raises():
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2, dtype=np.float32))
