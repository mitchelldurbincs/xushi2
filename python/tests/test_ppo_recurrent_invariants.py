"""PPO recurrent trainer invariant tests.

These defend the silent-failure modes from ``docs/rl_design.md`` §10
"Recurrent MAPPO silent failure modes":

1. Stale hidden state across PPO epochs.
2. BPTT truncation boundary ``detach()`` bugs.
3. Episode reset not zeroing hidden state.
4. Hidden state divergence between rollout sampling pass and training pass.
5. Feedforward-mode correctness (must ignore h) and loss masking.

See ``docs/plans/2026-04-21-memory-toy-plan.md`` Task 5 and Task 6 for the
design. These tests are written BEFORE the trainer exists — at the close
of Task 5 they all fail with ``ModuleNotFoundError: No module named
'train.ppo_recurrent'`` (or collection-time equivalent). Task 6 lands the
trainer and flips them green.

-------------------------------------------------------------------------
Task 6 NOTES (hooks the trainer must expose for these tests to pass):
-------------------------------------------------------------------------
The tests import ``PPOTrainer`` and ``PPOConfig`` from
``train.ppo_recurrent``. The trainer MUST expose:

* ``PPOTrainer(env_fn, config, seed=0)`` — builds a vectorized env by
  calling ``env_fn()`` ``num_envs`` times. Fully seeds model init, env
  reset, and any internal sampling RNG deterministically from ``seed``.
* ``collect_rollout() -> RolloutBuffer`` — rolls out ``rollout_len`` ticks
  across ``num_envs`` envs. Returns the buffer (with ``.obs``, ``.action``,
  ``.reward``, ``.logprob``, ``.value``, ``.done``, ``.h_init``
  populated). NOTE: ``compute_gae`` may or may not have been called yet;
  ``update`` is responsible for calling it.
* ``update(rollout) -> dict`` — runs ``num_epochs`` of PPO updates over
  minibatches from the rollout. Returns a metrics dict.
* ``trainer.model`` — the ``ActorCritic`` model.
* ``trainer.h`` — the live hidden state carried between
  ``collect_rollout`` calls, shape ``(num_envs, gru_hidden)``.

* ``trainer._training_h_init_log: list[torch.Tensor]`` — a debug hook
  used by ``test_bptt_h_init_identical_across_ppo_epochs``. The trainer
  MUST append (a clone of) every minibatch's ``h_init`` tensor to this
  list just before the minibatch forward pass, in this order:
      epoch_0_mb_0, epoch_0_mb_1, ..., epoch_1_mb_0, ...
  A clone is required so the test can compare across epochs without
  being fooled by in-place mutation. Task 6 can implement this in two
  lines at the top of the minibatch loop:
      if hasattr(self, "_training_h_init_log"):
          self._training_h_init_log.append(h_init.detach().clone())

  To guarantee PPO minibatches are deterministic across epochs AND match
  the shuffling within an epoch, the trainer should seed its minibatch
  generator once per ``update`` call (NOT per epoch). If Task 6 diverges
  on determinism guarantees, ``test_bptt_h_init_identical_across_ppo_epochs``
  may need a small tweak — but the invariant (same segment ->
  same ``h_init`` across all epochs) must still hold.

Minibatch dict field contract (used by Test 5's monkey-patch):
``update()`` iterates minibatches via ``rollout.iter_episode_minibatches(...)``
and the yielded dicts MUST use these exact keys (matching Task 4's buffer):
    "obs", "action", "old_logprob", "advantage", "return_", "old_value",
    "h_init", "valid_mask".
Test 5 also requires the tensors in each batch to be in-place mutable
during the yielded iteration — i.e., ``batch["obs"].add_(noise)`` must
affect the tensor the trainer actually uses in its forward pass. If Task 6
yields frozen dataclasses or clones tensors after yielding, Test 5's
monkey-patch strategy must be adapted.

If the API shape diverges in Task 6, lightly adjust the tests — the POINT
is the invariants, not the exact signatures.
"""

from __future__ import annotations

import copy
from typing import Callable

import pytest
import torch

from envs.memory_toy import MemoryToyEnv

# These tests are expected to fail at import time until Task 6 lands the
# module. The import is deliberately at module top so pytest surfaces a
# clean ModuleNotFoundError for every test in this file at collection.
from train.ppo_recurrent import PPOConfig, PPOTrainer  # noqa: E402
from train.ppo_recurrent import ppo_updater, rollout_collector  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# Small-but-realistic sizes so updates are fast. These match the toy task.
_OBS_DIM = 3
_ACTION_DIM = 2


def _make_config(
    *,
    num_envs: int = 4,
    rollout_len: int = 16,
    use_recurrence: bool = True,
    num_epochs: int = 4,
    minibatch_size: int = 2,
    gru_hidden: int = 8,
) -> PPOConfig:
    return PPOConfig(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=_OBS_DIM,
        action_dim=_ACTION_DIM,
        continuous_action_dim=_ACTION_DIM,
        embed_dim=8,
        gru_hidden=gru_hidden,
        head_hidden=8,
        action_log_std_init=-0.5,
        use_recurrence=use_recurrence,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
    )


def _memory_toy_env_fn(
    episode_length: int = 8, cue_visible_ticks: int = 4
) -> Callable[[], MemoryToyEnv]:
    def _thunk() -> MemoryToyEnv:
        return MemoryToyEnv(
            episode_length=episode_length, cue_visible_ticks=cue_visible_ticks
        )

    return _thunk


def _rollout_tensors_equal(r_a, r_b) -> None:
    """Assert two rollout buffers are byte-identical on all sampled fields."""
    for field in ("obs", "action", "reward", "logprob", "value", "done", "h_init"):
        ta = getattr(r_a, field)
        tb = getattr(r_b, field)
        assert torch.equal(ta, tb), f"rollout field {field!r} differs"




def test_collect_rollout_delegates_to_rollout_collector(monkeypatch):
    cfg = _make_config()
    trainer = PPOTrainer(env_fn=_memory_toy_env_fn(), config=cfg, seed=0)

    sentinel = object()

    def _fake_collect(t):
        assert t is trainer
        return sentinel

    monkeypatch.setattr(rollout_collector, "collect_rollout", _fake_collect)
    assert trainer.collect_rollout() is sentinel


def test_update_delegates_to_ppo_updater(monkeypatch):
    cfg = _make_config()
    trainer = PPOTrainer(env_fn=_memory_toy_env_fn(), config=cfg, seed=0)
    rollout = trainer._make_buffer()
    expected = {"ok": 1.0}

    def _fake_update(t, r):
        assert t is trainer
        assert r is rollout
        return expected

    monkeypatch.setattr(ppo_updater, "update_ppo", _fake_update)
    assert trainer.update(rollout) == expected

# ---------------------------------------------------------------------------
# Test 1: rollout determinism
# ---------------------------------------------------------------------------

def test_rollout_determinism_two_trainers_same_seed():
    """Two trainers with identical config, identical seed, identical
    ``env_fn`` must produce byte-identical rollouts.

    This is the base determinism check. If this fails, *everything* else
    in this file is suspect.
    """
    cfg = _make_config()
    env_fn = _memory_toy_env_fn(episode_length=8, cue_visible_ticks=4)

    trainer_a = PPOTrainer(env_fn=env_fn, config=cfg, seed=12345)
    trainer_b = PPOTrainer(env_fn=env_fn, config=cfg, seed=12345)

    rollout_a = trainer_a.collect_rollout()
    rollout_b = trainer_b.collect_rollout()

    _rollout_tensors_equal(rollout_a, rollout_b)


# ---------------------------------------------------------------------------
# Test 2: episode-reset zeroes h_init
# ---------------------------------------------------------------------------

def test_hidden_state_zero_after_env_reset():
    """For every env ``e`` and tick ``t > 0``: if ``done[e, t-1] == 1``,
    then ``h_init[e, t]`` must be a zero vector of shape
    ``(gru_hidden,)``.

    This defends failure mode 3 (episode reset not zeroing hidden state).
    With ``episode_length=8`` and ``rollout_len=32``, every env sees at
    least three episode boundaries inside the rollout, giving the buffer
    ample opportunity to leak a stale hidden state.
    """
    cfg = _make_config(num_envs=4, rollout_len=32, gru_hidden=8)
    env_fn = _memory_toy_env_fn(episode_length=8, cue_visible_ticks=4)

    trainer = PPOTrainer(env_fn=env_fn, config=cfg, seed=7)
    rollout = trainer.collect_rollout()

    num_envs = cfg.num_envs
    rollout_len = cfg.rollout_len
    zero_h = torch.zeros(cfg.gru_hidden)

    boundary_checks = 0
    for e in range(num_envs):
        for t in range(1, rollout_len):
            if rollout.done[e, t - 1].item() > 0.5:
                boundary_checks += 1
                torch.testing.assert_close(
                    rollout.h_init[e, t],
                    zero_h,
                    msg=(
                        f"h_init[{e},{t}] not zeroed after done at "
                        f"t={t - 1}: got {rollout.h_init[e, t].tolist()}"
                    ),
                )
    # Sanity: the test is vacuous if we never saw a boundary. Force at
    # least one boundary to be checked so a buggy env/rollout schedule
    # doesn't silently make this test pass.
    assert boundary_checks > 0, (
        "no episode boundaries observed inside the rollout; test is "
        "vacuous. Check episode_length vs rollout_len."
    )


# ---------------------------------------------------------------------------
# Test 3: h_init identical across PPO epochs
# ---------------------------------------------------------------------------

def test_bptt_h_init_identical_across_ppo_epochs():
    """The ``h_init`` fed to the model at training time for a given
    segment must be IDENTICAL across all PPO epochs.

    Defends failure modes 1 (stale hidden state across epochs) and 4
    (rollout vs training hidden-state divergence). If the trainer
    forgets to re-seed its minibatch generator per update (or re-runs
    the rollout's forward pass mid-update), segments will get a different
    ``h_init`` on epoch 2 vs epoch 1 and this test will catch it.

    Task 6 hook: ``trainer._training_h_init_log`` is a list onto which
    the trainer appends a clone of each minibatch's ``h_init`` in
    order (epoch 0 minibatches first, then epoch 1, ...). See module
    docstring.
    """
    num_epochs = 4
    minibatch_size = 2
    cfg = _make_config(
        num_envs=4,
        rollout_len=16,
        use_recurrence=True,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        gru_hidden=8,
    )
    env_fn = _memory_toy_env_fn(episode_length=8, cue_visible_ticks=4)

    trainer = PPOTrainer(env_fn=env_fn, config=cfg, seed=321)

    # Install the debug hook. Task 6 must check for this attribute and
    # append to it; see module docstring.
    trainer._training_h_init_log = []

    rollout = trainer.collect_rollout()
    trainer.update(rollout)

    log = trainer._training_h_init_log
    assert len(log) > 0, (
        "trainer._training_h_init_log is empty — Task 6 trainer is not "
        "appending h_init snapshots in the training loop. See module "
        "docstring for the required hook."
    )
    # The log length must be divisible by num_epochs: the trainer must
    # iterate the same number of minibatches per epoch.
    assert len(log) % num_epochs == 0, (
        f"len(_training_h_init_log) = {len(log)} is not divisible by "
        f"num_epochs={num_epochs}. Each epoch must visit the same number "
        f"of minibatches for this test's segment-comparison to be valid."
    )
    mb_per_epoch = len(log) // num_epochs

    # For each minibatch index i in [0, mb_per_epoch), the h_init tensor
    # must be byte-identical across all epochs. This implicitly also
    # asserts the minibatch SHUFFLE ORDER is identical across epochs —
    # which is also required for the invariant "same segment gets the
    # same h_init on every epoch" to hold when we only have access to
    # positional indexing.
    for mb_i in range(mb_per_epoch):
        ref = log[mb_i]  # epoch 0, minibatch mb_i
        for epoch in range(1, num_epochs):
            other = log[epoch * mb_per_epoch + mb_i]
            assert torch.equal(ref, other), (
                f"h_init for minibatch {mb_i} differs between epoch 0 "
                f"and epoch {epoch}. Stale or recomputed hidden state "
                f"leaked into the PPO update."
            )


# ---------------------------------------------------------------------------
# Test 4: feedforward mode must ignore h_init
# ---------------------------------------------------------------------------

def test_feedforward_mode_training_ignores_hidden_state():
    """With ``use_recurrence=False``, the trainer's update must produce
    byte-identical results whether the rollout's ``h_init`` tensor
    contains zeros or random garbage.

    Defends the feedforward-mode correctness invariant: a feedforward
    policy must not route any gradient through ``h_init``. If the
    trainer accidentally feeds ``h_init`` into a layer whose output
    flows into the loss, the state_dicts after update will differ.
    """
    cfg = _make_config(
        num_envs=4,
        rollout_len=16,
        use_recurrence=False,
        num_epochs=2,
        minibatch_size=2,
        gru_hidden=8,
    )
    env_fn = _memory_toy_env_fn(episode_length=8, cue_visible_ticks=4)

    trainer = PPOTrainer(env_fn=env_fn, config=cfg, seed=99)

    # Snapshot pre-update model weights so both runs start from the same
    # point.
    pre_state = copy.deepcopy(trainer.model.state_dict())

    rollout_a = trainer.collect_rollout()
    # Clone the rollout so run B starts from the same pre-update data.
    rollout_b = copy.deepcopy(rollout_a)

    # Run A: normal update with whatever h_init the rollout recorded.
    trainer.update(rollout_a)
    post_a = copy.deepcopy(trainer.model.state_dict())

    # Restore weights and any relevant RNG state that the update may have
    # advanced. Easiest: reconstruct the trainer with the same seed and
    # perform the same pre-update collect (to advance RNGs equally), then
    # mutate h_init and re-run update.
    #
    # Implementation note: rebuild instead of trying to hand-roll RNG
    # restoration. If Task 6's trainer seeds itself deterministically
    # from __init__, two fresh trainers at the same seed will yield
    # identical post_a / post_b when fed identical rollouts — and the
    # ONLY difference here is the h_init tensor content, which under
    # use_recurrence=False must not matter.
    trainer2 = PPOTrainer(env_fn=env_fn, config=cfg, seed=99)
    # Sanity-check fresh trainer starts from the same weights.
    for k, v in pre_state.items():
        assert torch.equal(v, trainer2.model.state_dict()[k]), (
            f"fresh trainer weights differ at {k}: PPOTrainer init is not "
            f"seed-deterministic, which this test relies on."
        )

    # Mutate h_init on the cloned rollout to random non-zero garbage.
    gen = torch.Generator().manual_seed(0)
    rollout_b.h_init = torch.randn(
        rollout_b.h_init.shape, generator=gen
    ).to(rollout_b.h_init.dtype)

    # Run B on the fresh trainer with the mutated rollout.
    trainer2.update(rollout_b)
    post_b = copy.deepcopy(trainer2.model.state_dict())

    # Under use_recurrence=False, the two post-update state_dicts must
    # be byte-identical.
    assert post_a.keys() == post_b.keys()
    for k in post_a:
        assert torch.equal(post_a[k], post_b[k]), (
            f"feedforward trainer weight {k!r} differs between runs with "
            f"zero vs random h_init — model or trainer is leaking h into "
            f"the loss/gradient."
        )


# ---------------------------------------------------------------------------
# Test 5: loss mask respects episode boundaries (pad contributes zero)
# ---------------------------------------------------------------------------

def test_loss_mask_respects_episode_boundaries():
    """The PPO update must weight per-sample losses by ``valid_mask``
    only — pad positions in a padded minibatch must contribute exactly
    zero to the total loss.

    We construct two rollouts that differ ONLY in the PAD positions of
    their buffers (i.e. entries that fall AFTER a ``done=True`` within
    a segment that gets shorter than ``rollout_len``, so when the buffer
    pads to ``L_max`` in a minibatch the differing values end up in the
    masked-out region). We then run ``update`` on each from the same
    initial weights and assert the post-update state_dicts are
    byte-identical.

    If the trainer forgets to mask, the pad positions will produce
    nonzero gradient contributions and the state_dicts will diverge.
    """
    # Use a rollout_len (7) that does NOT divide evenly by
    # episode_length (4) so segments have mixed lengths. With
    # rollout_len=8 the segments would all be length 4 and L_max=4
    # minibatches contain no pad positions — which makes the test
    # vacuous. The module docstring explicitly permits this kind of
    # small tweak to preserve the invariant.
    cfg = _make_config(
        num_envs=2,
        rollout_len=7,
        use_recurrence=True,
        num_epochs=1,
        minibatch_size=2,
        gru_hidden=8,
    )
    env_fn = _memory_toy_env_fn(episode_length=4, cue_visible_ticks=2)

    trainer_a = PPOTrainer(env_fn=env_fn, config=cfg, seed=55)
    trainer_b = PPOTrainer(env_fn=env_fn, config=cfg, seed=55)

    # Sanity: two fresh trainers at the same seed start identical.
    for k, v in trainer_a.model.state_dict().items():
        assert torch.equal(v, trainer_b.model.state_dict()[k])

    rollout_a = trainer_a.collect_rollout()
    rollout_b = copy.deepcopy(rollout_a)

    # Locate at least one episode boundary so that minibatches
    # padded to L_max contain a short segment with pad positions.
    # With episode_length=4 and rollout_len=8 every env sees at least
    # one mid-rollout done, so there's a short segment somewhere.
    boundary_found = False
    for e in range(cfg.num_envs):
        for t in range(cfg.rollout_len - 1):
            if rollout_b.done[e, t].item() > 0.5:
                boundary_found = True
                # Mutate positions STRICTLY AFTER this segment within
                # the env's row. If this env's next segment is the last
                # in the rollout and shorter than another env's segment,
                # those later ticks will be pad positions in the batch.
                # We mutate a broad region to maximize the chance that at
                # least one mutated position falls in a pad slot. The
                # test is valid as long as the mutated positions are
                # EITHER pad slots in the batch OR are themselves valid
                # positions — in the latter case the test would still be
                # observational (both runs see the same mutation because
                # rollout_a is unchanged; we only mutate rollout_b).
                #
                # To make the test SHARP, we only mutate obs/action/
                # advantage VALUES at positions we are confident will be
                # padded: specifically, any valid tick whose value we can
                # shift without affecting rollout_a. The cleanest way is
                # to mutate rollout_b at ALL positions and rely on the
                # valid_mask in the trainer to cancel them out — but
                # that would also perturb valid positions. Instead: we
                # zero out a single segment in rollout_b and expect the
                # trainer to still produce identical state_dicts ONLY
                # if that segment has no valid ticks in the minibatch.
                #
                # Practical approach: the rollout buffer only exposes
                # per-segment padding INSIDE ``iter_episode_minibatches``.
                # So instead of trying to surgically hit pad slots in the
                # raw rollout tensors (which don't have pad slots yet),
                # we use a second strategy: monkey-patch the buffer's
                # ``iter_episode_minibatches`` to inject pad-only noise
                # on rollout_b.
                break
        if boundary_found:
            break

    assert boundary_found, (
        "no episode boundaries inside the rollout; test is vacuous. "
        "Reduce episode_length vs rollout_len."
    )

    # --- Strategy: monkey-patch iter_episode_minibatches on rollout_b
    # to inject noise into the pad region of every minibatch it yields.
    # If the trainer masks losses correctly, the noise contributes zero
    # gradient and post_b == post_a.
    orig_iter = rollout_b.iter_episode_minibatches

    def _noised_iter(*args, **kwargs):
        gen = torch.Generator().manual_seed(777)
        for batch in orig_iter(*args, **kwargs):
            vm = batch["valid_mask"]  # (S, L)
            pad_mask = (vm < 0.5).float()  # 1 at pad positions
            # Inject random noise only at pad positions for fields that
            # feed the loss: obs, action, old_logprob, advantage,
            # return_, old_value.
            for key in ("obs", "action", "old_logprob", "advantage",
                        "return_", "old_value"):
                t = batch[key]
                if t.dim() == 3:
                    # (S, L, D) — broadcast pad_mask over the last dim.
                    noise = torch.randn(t.shape, generator=gen).to(t.dtype)
                    t.add_(noise * pad_mask.unsqueeze(-1))
                elif t.dim() == 2:
                    noise = torch.randn(t.shape, generator=gen).to(t.dtype)
                    t.add_(noise * pad_mask)
                else:
                    raise AssertionError(f"unexpected dim for {key}: {t.shape}")
            yield batch

    rollout_b.iter_episode_minibatches = _noised_iter

    # Sanity: there must be at least ONE pad slot across all minibatches.
    # Probe via a throwaway iteration on rollout_a (unnoised).
    # We need compute_gae to have been called to iterate. The trainer's
    # update() will call compute_gae on its own rollout, but for this
    # probe we call it here on a separate fresh copy.
    probe = copy.deepcopy(rollout_a)
    probe.compute_gae(
        last_values=torch.zeros(cfg.num_envs),
        last_dones=torch.zeros(cfg.num_envs),
    )
    gen_probe = torch.Generator().manual_seed(0)
    total_pad = 0
    for batch in probe.iter_episode_minibatches(
        minibatch_size=cfg.minibatch_size, generator=gen_probe
    ):
        vm = batch["valid_mask"]
        total_pad += int((vm < 0.5).sum().item())
    assert total_pad > 0, (
        "no pad positions found in any minibatch; test is vacuous. "
        "Tune episode_length/rollout_len so segments have mixed lengths."
    )

    # Run both updates.
    trainer_a.update(rollout_a)
    trainer_b.update(rollout_b)

    post_a = trainer_a.model.state_dict()
    post_b = trainer_b.model.state_dict()

    assert post_a.keys() == post_b.keys()
    for k in post_a:
        assert torch.equal(post_a[k], post_b[k]), (
            f"post-update weight {k!r} differs between unpadded-noise "
            f"rollout and pad-noised rollout — trainer is not masking "
            f"losses by valid_mask."
        )
