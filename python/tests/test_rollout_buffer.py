"""Tests for the recurrent PPO rollout buffer.

See ``docs/memory_toy.md`` §"Hidden-state management rules" and
``docs/plans/2026-04-21-memory-toy-plan.md`` Task 4 for the design.

Segment definition used throughout: a segment is a maximal contiguous run
of ticks within the rollout where ``h_init`` is constant. It starts at
rollout tick 0 (or right after a tick where ``done=True`` fired) and ends
at the tick where ``done=True`` fires (or the rollout's last tick for a
trailing partial segment).
"""

from __future__ import annotations

import torch
import pytest

from train.rollout_buffer import RolloutBuffer


NUM_ENVS = 2
ROLLOUT_LEN = 8
OBS_DIM = 3
ACTION_DIM = 2
GRU_HIDDEN = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95


def _make_buffer(
    num_envs: int = NUM_ENVS,
    rollout_len: int = ROLLOUT_LEN,
    obs_dim: int = OBS_DIM,
    action_dim: int = ACTION_DIM,
    gru_hidden: int = GRU_HIDDEN,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> RolloutBuffer:
    return RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gru_hidden=gru_hidden,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def _zeros_done() -> torch.Tensor:
    return torch.zeros(NUM_ENVS, dtype=torch.float32)


def test_buffer_fills_and_retrieves_in_order():
    """`add()` `rollout_len` ticks for `num_envs=2`, read back, assert
    values land at the correct `(env, tick)` position."""
    buf = _make_buffer()

    h_init = torch.zeros(NUM_ENVS, GRU_HIDDEN)
    for t in range(ROLLOUT_LEN):
        # obs[e, t] = [e*10 + t, e*10 + t + 0.1, e*10 + t + 0.2]
        obs = torch.tensor(
            [[e * 10 + t, e * 10 + t + 0.1, e * 10 + t + 0.2] for e in range(NUM_ENVS)],
            dtype=torch.float32,
        )
        action = torch.tensor(
            [[e * 10 + t, e * 10 + t + 0.5] for e in range(NUM_ENVS)],
            dtype=torch.float32,
        )
        logprob = torch.tensor([float(e * 10 + t) for e in range(NUM_ENVS)])
        reward = torch.tensor([float(e * 10 + t) * 0.01 for e in range(NUM_ENVS)])
        value = torch.tensor([float(e * 10 + t) * 0.02 for e in range(NUM_ENVS)])
        done = _zeros_done()
        buf.add(
            tick=t, obs=obs, action=action, logprob=logprob,
            reward=reward, value=value, done=done, h_init=h_init,
        )

    for t in range(ROLLOUT_LEN):
        for e in range(NUM_ENVS):
            assert buf.obs[e, t, 0].item() == pytest.approx(e * 10 + t)
            assert buf.obs[e, t, 1].item() == pytest.approx(e * 10 + t + 0.1)
            assert buf.obs[e, t, 2].item() == pytest.approx(e * 10 + t + 0.2)
            assert buf.action[e, t, 0].item() == pytest.approx(e * 10 + t)
            assert buf.action[e, t, 1].item() == pytest.approx(e * 10 + t + 0.5)
            assert buf.logprob[e, t].item() == pytest.approx(e * 10 + t)
            assert buf.reward[e, t].item() == pytest.approx((e * 10 + t) * 0.01)
            assert buf.value[e, t].item() == pytest.approx((e * 10 + t) * 0.02)
            assert buf.done[e, t].item() == 0.0


def test_h_init_persists_within_segment():
    """Within a segment (no `done=True`), every tick's stored `h_init`
    equals the segment's initial hidden state."""
    buf = _make_buffer()

    h_segment = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        dtype=torch.float32,
    )
    zeros_obs = torch.zeros(NUM_ENVS, OBS_DIM)
    zeros_act = torch.zeros(NUM_ENVS, ACTION_DIM)
    zeros_1d = torch.zeros(NUM_ENVS)
    for t in range(5):
        buf.add(
            tick=t, obs=zeros_obs, action=zeros_act, logprob=zeros_1d,
            reward=zeros_1d, value=zeros_1d, done=_zeros_done(),
            h_init=h_segment,
        )

    for t in range(5):
        for e in range(NUM_ENVS):
            torch.testing.assert_close(buf.h_init[e, t], h_segment[e])


def test_h_init_resets_to_zeros_after_done():
    """After a `done=True` at tick 3 and `mark_reset(env=0)`, the next
    tick for env 0 must have zero `h_init`."""
    buf = _make_buffer()

    h_pre = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float32,
    )
    zeros_obs = torch.zeros(NUM_ENVS, OBS_DIM)
    zeros_act = torch.zeros(NUM_ENVS, ACTION_DIM)
    zeros_1d = torch.zeros(NUM_ENVS)

    # Ticks 0..2 — in segment with h_pre, done=False.
    for t in range(3):
        buf.add(
            tick=t, obs=zeros_obs, action=zeros_act, logprob=zeros_1d,
            reward=zeros_1d, value=zeros_1d, done=_zeros_done(),
            h_init=h_pre,
        )

    # Tick 3 — done=True for env 0 only. The done-tick's h_init is still
    # the prior segment's h_init (the done tick IS the last tick of that
    # segment).
    done_t3 = torch.tensor([1.0, 0.0])
    buf.add(
        tick=3, obs=zeros_obs, action=zeros_act, logprob=zeros_1d,
        reward=zeros_1d, value=zeros_1d, done=done_t3,
        h_init=h_pre,
    )
    torch.testing.assert_close(buf.h_init[0, 3], h_pre[0])
    torch.testing.assert_close(buf.h_init[1, 3], h_pre[1])

    # Trainer signals: env 0 was reset between tick 3 and tick 4.
    buf.mark_reset(env_idx=0)

    # Caller would ordinarily pass zeros for env 0's h_init at tick 4.
    # The buffer must honor the reset regardless of what the caller hands
    # in (the contract is "next add after mark_reset writes zeros").
    h_caller_at_4 = torch.tensor(
        [[99.0, 99.0, 99.0, 99.0], [5.0, 6.0, 7.0, 8.0]],  # env 0 is junk
        dtype=torch.float32,
    )
    buf.add(
        tick=4, obs=zeros_obs, action=zeros_act, logprob=zeros_1d,
        reward=zeros_1d, value=zeros_1d, done=_zeros_done(),
        h_init=h_caller_at_4,
    )

    torch.testing.assert_close(
        buf.h_init[0, 4], torch.zeros(GRU_HIDDEN),
    )
    # Env 1 was never reset; its h_init at tick 4 is whatever the caller
    # passed in (the previous segment's h).
    torch.testing.assert_close(buf.h_init[1, 4], h_pre[1])


def test_gae_matches_hand_computed_values():
    """Hand-worked 4-tick 1-env example, dones=[0,0,0,0].

    rewards = [1, 0, 0, 1]
    values  = [0.5, 0.4, 0.3, 0.2]
    last_value = 0.0; last_done = 0
    gamma = 0.99, lambda = 0.95

    delta_3 = 1 + 0.99*1*0 - 0.2 = 0.8
    delta_2 = 0 + 0.99*1*0.2 - 0.3 = -0.102
    delta_1 = 0 + 0.99*1*0.3 - 0.4 = -0.103
    delta_0 = 1 + 0.99*1*0.4 - 0.5 = 0.896

    A_3 = delta_3 = 0.8
    A_2 = delta_2 + 0.99*0.95*1*A_3 = -0.102 + 0.9405*0.8 = 0.6504
    A_1 = delta_1 + 0.99*0.95*1*A_2 = -0.103 + 0.9405*0.6504 = 0.51149...
    A_0 = delta_0 + 0.99*0.95*1*A_1 = 0.896 + 0.9405*0.51149... = 1.376062...

    returns = advantages + values
    """
    buf = RolloutBuffer(
        num_envs=1, rollout_len=4, obs_dim=1, action_dim=1,
        gru_hidden=1, gamma=0.99, gae_lambda=0.95,
    )

    rewards = [1.0, 0.0, 0.0, 1.0]
    values = [0.5, 0.4, 0.3, 0.2]
    zeros_obs = torch.zeros(1, 1)
    zeros_act = torch.zeros(1, 1)
    z = torch.zeros(1)
    h_init = torch.zeros(1, 1)
    for t in range(4):
        buf.add(
            tick=t, obs=zeros_obs, action=zeros_act, logprob=z,
            reward=torch.tensor([rewards[t]]),
            value=torch.tensor([values[t]]),
            done=torch.zeros(1), h_init=h_init,
        )

    advantages, returns = buf.compute_gae(
        last_values=torch.zeros(1), last_dones=torch.zeros(1),
    )

    gamma, lam = 0.99, 0.95
    # Re-compute exactly the same way the buffer should.
    delta_3 = 1.0 + gamma * 1.0 * 0.0 - 0.2
    delta_2 = 0.0 + gamma * 1.0 * 0.2 - 0.3
    delta_1 = 0.0 + gamma * 1.0 * 0.3 - 0.4
    delta_0 = 1.0 + gamma * 1.0 * 0.4 - 0.5
    A_3 = delta_3
    A_2 = delta_2 + gamma * lam * 1.0 * A_3
    A_1 = delta_1 + gamma * lam * 1.0 * A_2
    A_0 = delta_0 + gamma * lam * 1.0 * A_1
    expected_adv = torch.tensor([[A_0, A_1, A_2, A_3]], dtype=torch.float32)
    expected_ret = expected_adv + torch.tensor([[values]], dtype=torch.float32).squeeze(0)

    torch.testing.assert_close(advantages, expected_adv, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(returns, expected_ret, atol=1e-6, rtol=1e-5)


def test_gae_respects_done_boundary():
    """4-tick 1-env, dones=[0,1,0,0].

    The (1-done_1)=0 factor must zero both the bootstrap in delta_1 and
    the recursion from A_2 into A_1. The advantage at tick 2 must be
    computed as if starting fresh from tick 2.
    """
    buf = RolloutBuffer(
        num_envs=1, rollout_len=4, obs_dim=1, action_dim=1,
        gru_hidden=1, gamma=0.99, gae_lambda=0.95,
    )

    rewards = [0.1, 0.2, 0.3, 0.4]
    values = [0.5, 0.4, 0.3, 0.2]
    dones = [0.0, 1.0, 0.0, 0.0]
    zeros_obs = torch.zeros(1, 1)
    zeros_act = torch.zeros(1, 1)
    z = torch.zeros(1)
    h_init = torch.zeros(1, 1)
    for t in range(4):
        buf.add(
            tick=t, obs=zeros_obs, action=zeros_act, logprob=z,
            reward=torch.tensor([rewards[t]]),
            value=torch.tensor([values[t]]),
            done=torch.tensor([dones[t]]),
            h_init=h_init,
        )

    advantages, returns = buf.compute_gae(
        last_values=torch.zeros(1), last_dones=torch.zeros(1),
    )

    gamma, lam = 0.99, 0.95
    # Tick 3 (last): delta_3 = r_3 + gamma*(1-done_3)*last_value - v_3
    delta_3 = 0.4 + gamma * (1.0 - 0.0) * 0.0 - 0.2
    A_3 = delta_3
    # Tick 2: delta_2 = r_2 + gamma*(1-done_2)*v_3 - v_2
    delta_2 = 0.3 + gamma * (1.0 - 0.0) * 0.2 - 0.3
    A_2 = delta_2 + gamma * lam * (1.0 - 0.0) * A_3
    # Tick 1 (done=1): delta_1 = r_1 + gamma*(1-1)*v_2 - v_1; recursion killed.
    delta_1 = 0.2 + gamma * (1.0 - 1.0) * 0.3 - 0.4
    A_1 = delta_1 + gamma * lam * (1.0 - 1.0) * A_2
    # A_1 must equal just delta_1 (no bootstrap from A_2 across boundary).
    assert A_1 == pytest.approx(delta_1)
    # Tick 0: delta_0 = r_0 + gamma*(1-done_0)*v_1 - v_0
    delta_0 = 0.1 + gamma * (1.0 - 0.0) * 0.4 - 0.5
    A_0 = delta_0 + gamma * lam * (1.0 - 0.0) * A_1

    expected_adv = torch.tensor([[A_0, A_1, A_2, A_3]], dtype=torch.float32)
    torch.testing.assert_close(
        advantages, expected_adv, atol=1e-6, rtol=1e-5,
    )

    # Explicit: advantage at tick 1 is just delta_1 (no leak from tick 2).
    assert advantages[0, 1].item() == pytest.approx(delta_1, abs=1e-6)
    # Explicit: tick 2's advantage was computed as if starting fresh.
    # (Validated above by constructing A_2 without any reference to tick 1.)


def _add_one_tick(buf, t, env_count, obs_dim, action_dim, gru_hidden,
                  done_vec, h_init):
    """Helper: fill tick `t` with deterministic placeholder data."""
    zeros_obs = torch.zeros(env_count, obs_dim)
    zeros_act = torch.zeros(env_count, action_dim)
    z = torch.zeros(env_count)
    buf.add(
        tick=t, obs=zeros_obs, action=zeros_act, logprob=z,
        reward=z, value=z, done=done_vec, h_init=h_init,
    )


def test_iter_episode_minibatches_groups_by_episode():
    """Rollout has known boundaries:
      env 0: dones at ticks 3 and 7 -> segments [0..3], [4..7]
      env 1: done at tick 5 -> segments [0..5], [6..7] (trailing partial)
    Total: 4 segments.
    """
    buf = RolloutBuffer(
        num_envs=2, rollout_len=8, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        gru_hidden=GRU_HIDDEN, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    )
    # Unique h per segment so we can check h_init later.
    # Env 0 segments: h_e0_s0 (ticks 0..3), h_e0_s1 (ticks 4..7)
    # Env 1 segments: h_e1_s0 (ticks 0..5), h_e1_s1 (ticks 6..7)
    h_e0_s0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    h_e0_s1 = torch.tensor([2.0, 0.0, 0.0, 0.0])
    h_e1_s0 = torch.tensor([3.0, 0.0, 0.0, 0.0])
    h_e1_s1 = torch.tensor([4.0, 0.0, 0.0, 0.0])

    done_schedule = {
        0: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # env 0
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # env 1
    }

    for t in range(8):
        done_vec = torch.tensor([done_schedule[0][t], done_schedule[1][t]])
        # Pick h_init for each env based on which segment tick t belongs to.
        if t <= 3:
            h_e0 = h_e0_s0
        else:
            h_e0 = h_e0_s1
        if t <= 5:
            h_e1 = h_e1_s0
        else:
            h_e1 = h_e1_s1
        h_init = torch.stack([h_e0, h_e1], dim=0)
        _add_one_tick(
            buf, t, env_count=2, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
            gru_hidden=GRU_HIDDEN, done_vec=done_vec, h_init=h_init,
        )
        # Note: we do NOT call mark_reset here because this test passes
        # each segment's distinctive h_init directly. In real trainer
        # usage mark_reset would fire and the buffer would zero the post-
        # reset h_init (see test_h_init_resets_to_zeros_after_done which
        # covers that contract explicitly).

    # First compute GAE (needed for advantages/returns in minibatches).
    buf.compute_gae(
        last_values=torch.zeros(2), last_dones=torch.zeros(2),
    )

    gen = torch.Generator().manual_seed(0)
    batches = list(buf.iter_episode_minibatches(minibatch_size=2, generator=gen))

    total_segments = sum(b["obs"].shape[0] for b in batches)
    assert total_segments == 4

    # Gather all segment h_inits across batches.
    seen_h_inits = []
    for b in batches:
        S = b["obs"].shape[0]
        for s in range(S):
            seen_h_inits.append(b["h_init"][s])

    # Expected h_init vectors for the 4 segments.
    expected = {
        tuple(h_e0_s0.tolist()),
        tuple(h_e0_s1.tolist()),
        tuple(h_e1_s0.tolist()),
        tuple(h_e1_s1.tolist()),
    }
    seen = {tuple(h.tolist()) for h in seen_h_inits}
    assert seen == expected

    # valid_mask: for each segment, mask is 1 up to its (length) and 0 after.
    # Segment lengths: env0 -> 4, 4; env1 -> 6, 2.
    # Max segment length = 6.
    expected_lengths = sorted([4, 4, 6, 2])
    actual_lengths = []
    for b in batches:
        S = b["obs"].shape[0]
        L = b["obs"].shape[1]
        vm = b["valid_mask"]
        assert vm.shape == (S, L)
        for s in range(S):
            length = int(vm[s].sum().item())
            actual_lengths.append(length)
            # Mask is a prefix of 1s followed by 0s.
            assert torch.all(vm[s, :length] == 1.0)
            assert torch.all(vm[s, length:] == 0.0)
    assert sorted(actual_lengths) == expected_lengths


def test_iter_episode_minibatches_valid_mask_padding():
    """Batch with mixed-length segments must right-pad shorter segments
    and mask pad positions with 0."""
    buf = RolloutBuffer(
        num_envs=1, rollout_len=6, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
        gru_hidden=GRU_HIDDEN, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    )
    # dones=[0,1,0,0,0,0]: segments [0..1] (len 2), [2..5] (len 4).
    h_seg0 = torch.zeros(1, GRU_HIDDEN)
    h_seg1 = torch.zeros(1, GRU_HIDDEN)
    done_schedule = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    for t in range(6):
        h_init = h_seg0 if t <= 1 else h_seg1
        # Put distinguishable obs so we can check they land in the right
        # place within each segment.
        obs = torch.tensor([[float(t), 0.0, 0.0]])
        _zeros_act = torch.zeros(1, ACTION_DIM)
        z = torch.zeros(1)
        buf.add(
            tick=t, obs=obs, action=_zeros_act, logprob=z,
            reward=z, value=z,
            done=torch.tensor([done_schedule[t]]),
            h_init=h_init,
        )
        if done_schedule[t] == 1.0:
            buf.mark_reset(env_idx=0)

    buf.compute_gae(
        last_values=torch.zeros(1), last_dones=torch.zeros(1),
    )

    gen = torch.Generator().manual_seed(42)
    batches = list(buf.iter_episode_minibatches(minibatch_size=2, generator=gen))
    # Both segments should land in the same minibatch.
    assert len(batches) == 1
    b = batches[0]
    S, L = b["obs"].shape[:2]
    assert S == 2
    assert L == 4  # max segment length

    vm = b["valid_mask"]
    # One segment has length 2, the other has length 4.
    lengths = [int(vm[s].sum().item()) for s in range(S)]
    assert sorted(lengths) == [2, 4]

    # For each segment: obs at valid ticks must be the original ticks'
    # obs; pad positions must be zeros (and masked).
    for s in range(S):
        length = int(vm[s].sum().item())
        # Pad positions are zeros and mask is 0.
        if length < L:
            assert torch.all(b["obs"][s, length:] == 0.0)
            assert torch.all(vm[s, length:] == 0.0)
        # Valid positions have mask 1.
        assert torch.all(vm[s, :length] == 1.0)
