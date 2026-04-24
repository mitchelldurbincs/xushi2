> **Status:** Completed 2026-04-23 (commit af4003c).

# Phase 2 — Memory-toy recurrent PPO: Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every task — RED → GREEN → commit. Do NOT commit per task; the user commits the whole delta at the end (see `~/.claude/projects/C--Users-mitchell-durbin-source-repos-cartridgeRepos-xushi2/memory/feedback_commits.md`).

**Goal:** Validate recurrent PPO training machinery (GRU hidden state, BPTT, rollout/training consistency) by training a recurrent policy to solve a toy "remember the cue after it disappears" task, and gating success on an automated ablation harness.

**Architecture:** Standalone Gymnasium env (`envs/memory_toy.py`, pure Python, no xushi2 dep) + CleanRL-style recurrent PPO (`train/ppo_recurrent.py`, ~500 LOC, reusable for Phase 3) + ablation eval harness (`eval/eval_memory_toy.py`). Dual-run (recurrent + feedforward baseline) in a single `train.train` invocation for apples-to-apples comparison.

**Tech Stack:** Python 3.12, PyTorch, Gymnasium 1.2.x, NumPy, PyYAML, tensorboard (all already in `python/pyproject.toml` deps; tensorboard needs adding).

**Design spec:** `docs/memory_toy.md` (companion design doc; read it first).

**Sanity baseline before starting:** `cmake --build build --config Release && ctest --test-dir build -C Release` → 87/87, and `cd python && python -m pytest tests/` → 59 passed. If either fails, stop and diagnose — don't start on a broken baseline.

---

## Task 1: Scaffold the `envs` package and register in pyproject

**Files:**
- Create: `python/envs/__init__.py`
- Create: `python/envs/memory_toy.py` (empty stub — just `class MemoryToyEnv: pass` so tests can import)
- Modify: `python/pyproject.toml` (add `"envs"` to `packages`, add `tensorboard` to deps)

**Step 1:** Create `python/envs/__init__.py` with a single line:

```python
"""Standalone Gymnasium envs (not backed by the C++ sim)."""
```

**Step 2:** Create `python/envs/memory_toy.py` with just:

```python
class MemoryToyEnv:
    """Stub — implementation in Task 2."""
```

**Step 3:** Edit `python/pyproject.toml`. In `[project]` add `"tensorboard>=2.15"` to `dependencies`. In `[tool.setuptools]` change `packages = ["xushi2", "train", "eval"]` to `packages = ["xushi2", "train", "eval", "envs"]`.

**Step 4:** Run `pip install -e . --no-deps` from `python/` to re-register the package. Verify `python -c "import envs.memory_toy"` returns no error.

**Step 5:** Install tensorboard: `pip install tensorboard>=2.15`. Verify `python -c "import torch.utils.tensorboard; print('ok')"` → `ok`.

---

## Task 2: Implement MemoryToyEnv (TDD)

**Files:**
- Modify: `python/envs/memory_toy.py`
- Create: `python/tests/test_memory_toy_env.py`

**Step 1: Write failing tests first.** Create `python/tests/test_memory_toy_env.py`:

```python
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
    # Target is on unit circle → norm ≈ 1
    assert abs(np.linalg.norm(obs[:2]) - 1.0) < 1e-5

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    for t in range(1, 4):  # ticks 1, 2, 3 — still visible
        obs, _, _, _, _ = env.step(zero_action)
        assert obs[2] == 1.0

    obs, _, _, _, _ = env.step(zero_action)  # tick 4 — now hidden
    assert obs[2] == 0.0
    np.testing.assert_allclose(obs[:2], [0.0, 0.0])


def test_terminal_reward_matches_optimal_action():
    env = MemoryToyEnv(episode_length=64, cue_visible_ticks=4)
    obs, _ = env.reset(seed=123)
    target = obs[:2].copy()  # observed at t=0
    zero = np.array([0.0, 0.0], dtype=np.float32)
    terminal_reward = None
    for t in range(63):  # steps 1..63
        # At t == episode_length - 1 (= 63), act optimally
        action = target if t == 62 else zero
        _, r, term, trunc, _ = env.step(action)
        if term or trunc:
            terminal_reward = r
            break
    assert terminal_reward is not None
    assert abs(terminal_reward) < 1e-5  # exact match → reward 0


def test_feedforward_baseline_action_yields_expected_reward():
    # Feedforward at terminal tick sees (0,0,0). Best guess is (0,0).
    # Expected terminal reward = -‖target‖ = -1.
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
```

**Step 2: Run tests, confirm they fail.**

```
cd python && python -m pytest tests/test_memory_toy_env.py -v
```

Expected: all 5 fail with `TypeError: MemoryToyEnv() takes no arguments` or `AttributeError: 'MemoryToyEnv' has no attribute 'reset'`.

**Step 3: Implement `MemoryToyEnv`.** Replace the stub in `python/envs/memory_toy.py`:

```python
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MemoryToyEnv(gym.Env):
    """Phase-2 memory sanity toy. See docs/memory_toy.md."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int = 64, cue_visible_ticks: int = 4):
        super().__init__()
        if cue_visible_ticks >= episode_length:
            raise ValueError("cue_visible_ticks must be < episode_length")
        self.T = int(episode_length)
        self.k = int(cue_visible_ticks)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )
        self._rng: np.random.Generator | None = None
        self._target: np.ndarray | None = None
        self._t: int = 0

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        theta = self._rng.uniform(0.0, 2.0 * np.pi)
        self._target = np.array(
            [np.cos(theta), np.sin(theta)], dtype=np.float32,
        )
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        assert self._target is not None, "must call reset() before step()"
        self._t += 1
        obs = self._obs()
        terminated = self._t >= self.T - 1 + 1  # after terminal tick step, done
        # Actually: we step from tick (t-1) to tick t. Terminal tick is T-1.
        # The agent acts AT tick T-1 based on obs AT tick T-1. So termination
        # fires when we've just processed action for tick T-1.
        terminated = self._t >= self.T
        reward = 0.0
        if terminated:
            a = np.asarray(action, dtype=np.float32).reshape(2)
            dist = float(np.linalg.norm(a - self._target))
            reward = max(-2.0, -dist)
        return obs, reward, bool(terminated), False, {}

    def _obs(self) -> np.ndarray:
        assert self._target is not None
        # Current tick for observation is self._t (0 at reset, incremented in step)
        if self._t < self.k:
            return np.array(
                [self._target[0], self._target[1], 1.0], dtype=np.float32,
            )
        return np.zeros(3, dtype=np.float32)
```

Note the `step` timing is subtle: `reset()` returns obs at tick 0. `step(action)` applies the action *for the current tick* and advances `_t`. The terminal tick is `T-1`, so after the `T-1`-th `step` call, `_t == T` and we terminate. The test `test_episode_length_exact` calls `step` exactly `T` times, so the 64th call terminates.

**Step 4: Run tests, confirm they pass.**

```
cd python && python -m pytest tests/test_memory_toy_env.py -v
```

Expected: 5 passed.

**Step 5:** No per-task commit (see user's feedback memory).

---

## Task 3: Implement the model module (TDD)

**Files:**
- Create: `python/train/models.py`
- Create: `python/tests/test_models.py`

**Step 1: Write failing tests.** Create `python/tests/test_models.py`:

```python
import torch
from train.models import build_model


def test_recurrent_forward_shapes():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.zeros(8, 3)
    h = model.init_hidden(batch_size=8)
    action_mean, log_std, value, h_next = model.forward(obs, h)
    assert action_mean.shape == (8, 2)
    assert log_std.shape == (2,)
    assert value.shape == (8,)
    assert h_next.shape == h.shape == (8, 64)


def test_feedforward_bypasses_gru():
    """With use_recurrence=False, forward output must not depend on h."""
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=False,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.randn(4, 3)
    h1 = torch.zeros(4, 64)
    h2 = torch.randn(4, 64)
    mean1, _, v1, _ = model.forward(obs, h1)
    mean2, _, v2, _ = model.forward(obs, h2)
    torch.testing.assert_close(mean1, mean2)
    torch.testing.assert_close(v1, v2)


def test_recurrent_uses_hidden_state():
    """With use_recurrence=True, forward output MUST depend on h."""
    torch.manual_seed(0)
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.randn(4, 3)
    h1 = torch.zeros(4, 64)
    h2 = torch.ones(4, 64)
    mean1, _, _, _ = model.forward(obs, h1)
    mean2, _, _, _ = model.forward(obs, h2)
    assert not torch.allclose(mean1, mean2)


def test_init_hidden_is_zeros():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    h = model.init_hidden(batch_size=3)
    assert torch.all(h == 0)


def test_action_sampling_tanh_squash_bounds():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=2.0)  # high std → actions could blow up
    obs = torch.zeros(1000, 3)
    h = model.init_hidden(1000)
    action, logp, _ = model.sample_action(obs, h)
    assert action.shape == (1000, 2)
    assert (action.abs() <= 1.0).all(), "tanh squash must bound actions in [-1, 1]"
```

**Step 2: Run tests, confirm they fail.**

```
cd python && python -m pytest tests/test_models.py -v
```

Expected: all 5 fail with `ModuleNotFoundError`.

**Step 3: Implement `python/train/models.py`.** Provide `build_model(...)` returning an `ActorCritic(nn.Module)` with methods `init_hidden(batch_size)`, `forward(obs, h) -> (action_mean, log_std, value, h_next)`, and `sample_action(obs, h) -> (action, logprob, h_next)`.

Key implementation points (the test suite enforces these):
- If `use_recurrence=True`: `GRUCell(embed_dim, gru_hidden)`. If False: `nn.Identity()` as the recurrent cell, with a matching-capacity MLP replacing it so param counts are comparable (one extra `Linear(embed_dim, gru_hidden) + ReLU` in the feedforward branch, applied to `embed(obs)` only).
- Actor head: `Linear(gru_hidden, head_hidden) → ReLU → Linear(head_hidden, action_dim)` → mean. `log_std` is `nn.Parameter(torch.ones(action_dim) * action_log_std_init)`.
- Critic head: same MLP shape but output dim 1. Squeeze to `(batch,)`.
- `sample_action`: sample from `Normal(mean, exp(log_std))`, then `tanh` and apply log-prob correction: `log_prob = normal.log_prob(u).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)` where `u` is the pre-tanh sample and `action = tanh(u)`.

**Step 4: Run tests, confirm they pass.**

```
cd python && python -m pytest tests/test_models.py -v
```

Expected: 5 passed.

---

## Task 4: Rollout buffer + GAE (TDD)

**Files:**
- Create: `python/train/rollout_buffer.py`
- Create: `python/tests/test_rollout_buffer.py`

The rollout buffer is the heart of the correctness problem for recurrent PPO. It must store `h_0_of_segment` per episode per env, plus per-tick `(obs, action, logprob, reward, done, value)`, and correctly identify segment boundaries.

**Step 1: Write failing tests** covering:
- Buffer fills to capacity and retrieves entries in the right order.
- Segment boundaries are identified correctly (every `done==True` starts a new segment at the next tick).
- `h_0_of_segment` is stored at the start of each segment.
- GAE computation matches a reference implementation on a hand-worked 4-tick example with known rewards, values, gamma=0.99, lambda=0.95.
- When `done[t] == True`, GAE at `t+1` starts fresh (no bootstrap from `value[t+1]` across the episode boundary into the prior episode's advantage).

**Step 2:** Run, confirm fail.

**Step 3:** Implement. The buffer stores a contiguous `(num_envs, rollout_len)` grid for each tensor, plus a parallel `(num_envs, rollout_len)` tensor of `h_init_for_segment_containing_this_tick` — every tick records *which* `h_0` its forward pass should start from. On `reset` for env `i`, we write the new `h_0 = zeros` to the buffer for that env going forward until the next reset.

GAE formula:

```
delta_t = reward_t + gamma * (1 - done_t) * value_{t+1} - value_t
A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
```

with `value_{t+1}` being the bootstrap value at rollout end. The `(1 - done_t)` factor zeros the bootstrap across episode boundaries.

**Step 4:** Run, confirm pass.

---

## Task 5: PPO invariant tests (the safety-critical ones)

**Files:**
- Create: `python/tests/test_ppo_recurrent_invariants.py`

These are the tests that defend the silent failure modes from `rl_design.md` §10. Write them BEFORE the PPO trainer.

**Tests to include:**

1. **`test_rollout_determinism`** — two full rollouts with identical seeds + identical initial model weights produce byte-identical `(actions, rewards, hidden_states)` tensors. This is the headline determinism invariant.
2. **`test_hidden_state_resets_to_zero_on_env_reset`** — after `env.reset()` for env index `i`, the trainer's stored `h_0` for env `i` is all zeros. Specifically: trigger a reset mid-rollout by forcing a terminal condition, and assert the next tick's stored `h_init` is zero.
3. **`test_bptt_boundary_h_0_identical_across_epochs`** — run one PPO update with `num_epochs=4`. On each epoch, the forward pass at training time should start from the *same* `h_0_of_segment` value (not propagated from the prior epoch's final hidden state). Assert byte-equality of the h_0 passed to the training forward on epochs 1, 2, 3, 4.
4. **`test_feedforward_mode_ignores_hidden_state`** — redundant with the model test but at the trainer level: train one step with `use_recurrence=False` using zero `h` vs random `h`; loss should be identical.
5. **`test_loss_mask_respects_episode_boundaries`** — build a minibatch straddling an episode boundary; compute PPO loss; assert loss contribution of post-boundary ticks uses the post-boundary episode's own `h_0`, not the prior episode's final `h`.

**Step 1:** Write the tests as stubs against an API you haven't written yet. Use `from train.ppo_recurrent import PPOTrainer, train_from_config` and mock/construct `PPOTrainer` directly.

**Step 2:** Run; all 5 fail with `ImportError`. Good.

Keep these tests failing while you implement Task 6. They'll drive the API shape.

---

## Task 6: Implement the PPO trainer

**Files:**
- Create: `python/train/ppo_recurrent.py`

**What this file contains** (roughly in order):
- `class PPOConfig` — dataclass matching the `ppo:` + `model:` + `env:` + `run:` sections of the YAML.
- `class PPOTrainer`:
  - `__init__(env_fn, config, logger)`: creates `num_envs` envs via `gymnasium.vector.SyncVectorEnv`, builds the model, builds the rollout buffer, sets up Adam.
  - `collect_rollout() -> Rollout`: runs `rollout_len` ticks per env, carrying hidden state correctly (reset to zero on `done`), storing every tick's `(obs, action, logp, reward, value, done, h_init)`.
  - `compute_gae(rollout) -> advantages, returns`: calls into Task 4's GAE implementation.
  - `update(rollout, advantages, returns) -> metrics`: for `num_epochs` PPO epochs, shuffles minibatches of whole episodes, re-runs the forward pass from each segment's stored `h_0`, computes clipped PPO loss + value loss + entropy bonus, backprop + clip + step.
  - `save_checkpoint(path)`, `load_checkpoint(path)`.

- `def train_from_config(config: dict) -> dict` — top-level entrypoint:
  - Runs two `PPOTrainer` instances back-to-back (recurrent then feedforward, or in two threads if easy — but single-threaded is fine at 10 min per run).
  - Each uses its own `output_dir` sub-directory (`runs/phase2_memory_toy/recurrent/` and `.../feedforward/`).
  - Prints the final summary line: `[phase2] recurrent_final={r:.3f} feedforward_final={f:.3f} gap={g:.3f}`.
  - Returns the two final eval rewards.

**Critical implementation rules (enforced by Task 5 tests):**

- In `collect_rollout`, when `done[i]` fires, the *next* tick's `h_init[i]` is zero (handled by `buffer.write_reset(i)` which the buffer stores alongside the obs).
- In `update`, the forward pass re-runs from `buffer.h_init` for each minibatch segment. It does *not* carry `h` across PPO epochs. It does *not* read `h` from the rollout buffer's stored `h_t` for non-segment-start ticks — those are stored only for diagnostics.
- The forward pass uses `detach()` exactly at `h_init`: `h = buffer.h_init[seg].detach().requires_grad_(False)`. Within the segment, `h` propagates through BPTT untouched.
- Loss masking: each minibatch is a `(num_segments, max_episode_len)` tensor; segments shorter than `max_episode_len` are right-padded and masked out of the loss via a `valid_mask` tensor.

**Step 1:** Implement the trainer. Keep functions under ~80 LOC each; the whole file will land around 500 LOC.

**Step 2:** Run the Task 5 invariant tests:

```
cd python && python -m pytest tests/test_ppo_recurrent_invariants.py -v
```

Expected: 5 passed.

**Step 3:** Also run all tests to catch regressions:

```
cd python && python -m pytest tests/ -v
```

Expected: existing 59 + new 5 + 5 + 5 = 74 passed.

---

## Task 7: Wire up the `phase: 2` branch in `train/train.py`

**Files:**
- Modify: `python/train/train.py` (lines 82-85 currently bail on `phase != 0`)
- Create: `experiments/configs/phase2_memory_toy.yaml`

**Step 1:** Create the YAML config with the exact schema from `docs/memory_toy.md` §"Config schema".

**Step 2:** Modify `train/train.py`. Replace:

```python
if phase != 0 or not assert_determinism:
    print(f"[xushi2] phase {phase} not yet supported by this entrypoint")
    return 2
```

with:

```python
if phase == 0 and assert_determinism:
    pass_a = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)
    pass_b = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)
    rc = _assert_identical(pass_a, pass_b)
    # ... (existing phase-0 logging) ...
    return rc

if phase == 2:
    from train.ppo_recurrent import train_from_config
    result = train_from_config(config)
    print(f"[phase2] recurrent_final={result['recurrent']:.3f} "
          f"feedforward_final={result['feedforward']:.3f} "
          f"gap={result['recurrent'] - result['feedforward']:.3f}")
    return 0

print(f"[xushi2] phase {phase} not yet supported")
return 2
```

Note this moves the existing phase-0 flow into a `phase == 0` branch, which is a small refactor. Read `train/train.py:58-96` carefully to preserve all existing behavior.

**Step 3:** Smoke-test the CLI:

```
cd python && python -m train.train --config ../experiments/configs/phase2_memory_toy.yaml
```

Expected: runs for ~10 minutes, prints per-update progress lines, ends with the summary line. Recurrent final should be near `0`; feedforward final should be near `-1.0`.

**Step 4:** Do *not* commit. The user reviews and commits the whole delta.

---

## Task 8: Ablation eval harness (TDD)

**Files:**
- Create: `python/eval/eval_memory_toy.py`
- Create: `python/tests/test_eval_memory_toy.py`

**Step 1: Write failing tests.**

```python
# python/tests/test_eval_memory_toy.py
import torch
import numpy as np
import pytest
from pathlib import Path
from eval.eval_memory_toy import (
    run_ablation, AblationResult, ablation_modes_differ, main
)


def test_ablation_modes_mutate_hidden_state_differently():
    """The three modes must not silently share a code path."""
    torch.manual_seed(0)
    # normal vs zero vs random should produce measurably different trajectories
    assert ablation_modes_differ(num_episodes=10, seed=0)


def test_confidence_interval_width_shrinks_with_episode_count(tmp_path):
    # Build a trivial dummy checkpoint for the harness, run at N=50 vs N=500,
    # assert CI at 500 is at least 2x tighter.
    ...  # skeleton; fill once harness exists


def test_harness_exits_nonzero_on_bound_violation(tmp_path, monkeypatch):
    # Synthesize an AblationResult that violates the normal_mean > -0.15 bound;
    # assert main() returns 1.
    ...
```

**Step 2:** Implement `eval/eval_memory_toy.py`:

- `load_checkpoint(path)` returns `(model, config)`.
- `run_ablation(model, config, mode: Literal["normal", "zero_every_tick", "random_every_tick"], num_episodes: int, seed: int) -> AblationResult` — loops `num_episodes` episodes of the env, running the model forward with `h` clobbered per `mode`. Returns mean + 95% CI of terminal reward.
- `ablation_modes_differ(...)` — utility used by the test: returns True if all three modes produce distinct mean terminal rewards.
- `main()` — argparse CLI: `--checkpoint PATH`, `--episodes N` (default 500), `--seed N`. Runs all three modes, prints a table, asserts the four conditions from `docs/memory_toy.md`, exits 0 or 1.

**Step 3: Pin the gate conditions** in `main()`:

```python
def _check_gate(normal, zero, random_) -> tuple[bool, list[str]]:
    failures = []
    if not (normal.mean > -0.15):
        failures.append(f"normal_mean={normal.mean:.3f} is not > -0.15")
    if not (-1.2 <= zero.mean <= -0.8):
        failures.append(f"zero_every_tick_mean={zero.mean:.3f} outside [-1.2, -0.8]")
    if not (-1.5 <= random_.mean <= -0.8):
        failures.append(f"random_every_tick_mean={random_.mean:.3f} outside [-1.5, -0.8]")
    if not (normal.mean - zero.mean > 0.5):
        failures.append(
            f"gap normal-zero = {normal.mean - zero.mean:.3f} is not > 0.5"
        )
    return len(failures) == 0, failures
```

**Step 4:** Run tests, confirm they pass.

**Step 5:** Run the harness against the checkpoint from Task 7:

```
python -m eval.eval_memory_toy --checkpoint runs/phase2_memory_toy/recurrent/ckpt_final.pt
```

Expected: exits 0. Prints a table something like:

```
mode                  mean     ci95      n
--------------------  -------  --------  ---
normal                -0.032   ±0.010    500
zero_every_tick       -1.005   ±0.023    500
random_every_tick     -1.110   ±0.028    500

gap (normal - zero):  0.973
PHASE 2 GATE: PASS
```

---

## Task 9: Final verification checklist

**Step 1:** Run every gate in order:

```bash
# 1. C++ tests still green
cmake --build build --config Release
ctest --test-dir build -C Release
# Expected: 100% tests passed, 0 tests failed out of 87

# 2. Python tests all pass
cd python
python -m pytest tests/
# Expected: 59 + 13 new = 72 passed (or more if Task 4 added finer-grained tests)

# 3. End-to-end training runs and gate passes
python -m train.train --config ../experiments/configs/phase2_memory_toy.yaml
# Expected: exits 0; final line shows gap near 1.0

# 4. Ablation gate passes
python -m eval.eval_memory_toy --checkpoint runs/phase2_memory_toy/recurrent/ckpt_final.pt
# Expected: exits 0; PHASE 2 GATE: PASS
```

**Step 2:** If all four pass, Phase 2 is complete. Report findings to the user (learning curves, final rewards, ablation table). Do NOT commit — the user will review `git status` and commit the whole delta themselves.

**Step 3:** If any gate fails, do NOT hack the bounds to make it pass. Instead:
- If a unit test failed: you have a correctness bug. Use `superpowers:systematic-debugging`.
- If training failed to converge: first check the invariant tests all still pass. If yes, the bug is in hyperparameters (LR, entropy coef, rollout length) or architecture — tune carefully, one variable at a time.
- If ablation gate failed but training looked fine: the silent-failure mode from `rl_design.md` §10 is real and your trainer has a bug. This is exactly what Phase 2 is designed to catch.

---

## Post-completion notes

- The trainer code (`train/ppo_recurrent.py`) is the skeleton for Phase 3. Phase 3 swaps `MemoryToyEnv` for a Gymnasium-wrapped `xushi2.env.RangerEnv`, adjusts obs/action dims in the config, and that's the delta. Do not refactor the trainer for Phase 3 in the same PR as Phase 2.
- If the user asks to also run feedforward ablation on the *trained feedforward* checkpoint: that's trivially true by construction (feedforward has no memory to ablate), so it's not a useful additional gate. Skip.
- Tensorboard logs live under `runs/phase2_memory_toy/{recurrent,feedforward}/events.out.tfevents*` — user can `tensorboard --logdir runs/phase2_memory_toy` to compare the two curves.
