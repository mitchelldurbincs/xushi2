"""CleanRL-style recurrent PPO trainer for xushi2 Phase-2.

Implements the invariant contract defined by
``python/tests/test_ppo_recurrent_invariants.py``:

* Seed-deterministic rollouts (Test 1).
* Hidden-state zeroing on episode reset (Test 2).
* Identical per-segment ``h_init`` across all PPO epochs within an update
  (Test 3). Implemented by seeding the minibatch-shuffle generator once
  per update and reusing it across epochs.
* Feedforward-mode path that routes no gradient through ``h_init``
  (Test 4). The model handles this structurally; the trainer just feeds
  ``h_init`` in uniformly.
* ``valid_mask``-aware loss normalization — policy, value, and entropy
  terms all multiply by ``valid_mask`` and divide by ``valid_mask.sum()``
  (Test 5).

Design notes:
* CPU-only for Phase-2; the Phase-2 toy is tiny and the determinism tests
  assume CPU.
* Action log-probs are recomputed at training time from the stored
  squashed action by inverting tanh (atanh) with an eps-clamp.
"""

from __future__ import annotations

from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.evaluate import evaluate_policy
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.ppo_recurrent.orchestration import train_from_config
from train.ppo_recurrent.trainer import PPOTrainer

__all__ = [
    "PPOConfig",
    "PPOTrainer",
    "evaluate_policy",
    "lr_for_update",
    "train_from_config",
]
