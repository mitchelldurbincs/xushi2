> **Status:** Completed 2026-04-23 (commit af4003c).

# PPO Recurrent Module Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `python/train/ppo_recurrent.py` (769 lines) into a `ppo_recurrent/` package with focused submodules, preserving the public API and all behavior.

**Architecture:** Convert `ppo_recurrent.py` into a package at `python/train/ppo_recurrent/`. Split along the file's existing natural seams: config dataclass, loss math helpers, LR schedule, trainer class, evaluation helper, and training orchestration. `__init__.py` re-exports the public surface (`PPOConfig`, `PPOTrainer`, `lr_for_update`, `train_from_config`) so every existing import keeps working unchanged. The split is a pure move-and-reorganize: no behavior changes, no renames of public symbols. Private helpers used only within orchestration may drop their leading underscore on move (opt-in cleanup).

**Tech Stack:** Python 3.10+, PyTorch, Gymnasium, pytest.

---

## Ground Rules

- **No behavior changes.** Every extraction must leave the full test suite green with no diff in metrics.
- **Public API is frozen:** `PPOConfig`, `PPOTrainer`, `lr_for_update`, `train_from_config` remain importable from `train.ppo_recurrent`.
- **Tests run from `python/`** (per `pyproject.toml`: `testpaths = ["tests"]`, `pythonpath = ["."]`).
- **Run the two PPO test files as the primary regression gate** after every task: `pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v`. The full suite (`pytest`) runs once at the end.
- **No per-task commits.** The user commits the whole delta at the end.
- **Working directory:** Run all commands from `python/`.

---

## Reference: Current File Layout

Source: `python/train/ppo_recurrent.py` (769 lines). Symbols and approximate line ranges:

| Symbol | Lines | Destination |
|---|---|---|
| Module docstring + imports | 1–42 | split across modules (re-imported where needed) |
| `_ATANH_EPS`, `_LOG2` constants | 46–47 | `losses.py` |
| `PPOConfig` dataclass | 50–75 | `config.py` |
| `_tanh_squashed_logprob` | 78–100 | `losses.py` |
| `_masked_mean` | 103–110 | `losses.py` |
| `lr_for_update` | 113–168 | `lr_schedule.py` |
| `PPOTrainer` class | 171–552 | `trainer.py` |
| `evaluate_policy` | 558–590 | `evaluate.py` |
| `_make_ppo_config` | 593–619 | `orchestration.py` (rename → `make_ppo_config`) |
| `_save_checkpoint` | 622–627 | `orchestration.py` |
| `_run_variant` | 630–749 | `orchestration.py` |
| `train_from_config` | 752–768 | `orchestration.py` |

External callers of this module (must keep working):
- `python/tests/test_ppo_recurrent_invariants.py:84` — imports `PPOConfig, PPOTrainer`
- `python/tests/test_ppo_recurrent_lr_schedule.py:7` — imports `lr_for_update`
- `python/train/train.py:100` — imports `train_from_config`

---

## Task 1: Baseline the test suite

**Goal:** Capture a known-green baseline so we can detect any regression introduced by the split.

**Step 1:** From `python/`, run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: all tests pass. Note any already-failing tests and treat those as the baseline (do not fix them as part of this split).

**Step 2:** Run the full suite to capture a broader baseline.

```bash
cd python && pytest -x
```

Expected: record the pass/fail state. This is the reference we compare against at the end.

---

## Task 2: Convert module to package (no content changes)

**Goal:** Turn `ppo_recurrent.py` into `ppo_recurrent/__init__.py` verbatim. This is the smallest possible step that changes file layout — verifies Python packaging/imports still resolve before we start shuffling symbols.

**Files:**
- Move: `python/train/ppo_recurrent.py` → `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create the directory and move the file with `git mv` so history is preserved.

```bash
cd python/train && mkdir ppo_recurrent && git mv ppo_recurrent.py ppo_recurrent/__init__.py
```

**Step 2:** Update `python/pyproject.toml` to include the new subpackage. Find the `packages = [...]` line and add `"train.ppo_recurrent"`:

```toml
packages = ["xushi2", "train", "train.ppo_recurrent", "eval", "envs"]
```

Rationale: setuptools does not auto-recurse when `packages` is an explicit list. Without this, `pip install -e .` will not install the new subpackage.

**Step 3:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: identical result to Task 1's baseline. If any import fails here, the packaging move is wrong — fix before proceeding.

---

## Task 3: Extract `losses.py`

**Goal:** Move the loss math helpers (`_ATANH_EPS`, `_LOG2`, `_tanh_squashed_logprob`, `_masked_mean`) into their own module.

**Files:**
- Create: `python/train/ppo_recurrent/losses.py`
- Modify: `python/train/ppo_recurrent/__init__.py` (remove moved code, add import)

**Step 1:** Create `losses.py` with this content (copied verbatim from `__init__.py` lines 44–110, plus the imports those helpers need):

```python
"""Loss math helpers for the recurrent PPO trainer.

Kept as free functions (not methods on ``PPOTrainer``) so they can be
unit-tested in isolation and reused by evaluation/analysis code.
"""

from __future__ import annotations

import math

import torch

# Numerical guard for atanh(action) reconstruction. Pulled out as a module
# constant so tests/callers can reason about it.
_ATANH_EPS = 1e-6
_LOG2 = math.log(2.0)


def _tanh_squashed_logprob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ... (paste the exact body from __init__.py lines 78–100)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # ... (paste the exact body from __init__.py lines 103–110)
```

**IMPORTANT:** Do not reformat or rewrite the function bodies. Copy them verbatim (whitespace, comments, all).

**Step 2:** In `__init__.py`, delete the moved code (constants + both functions), and add this import near the top:

```python
from train.ppo_recurrent.losses import (
    _ATANH_EPS,
    _LOG2,
    _masked_mean,
    _tanh_squashed_logprob,
)
```

Keep the `_` prefix — these are still private to the package.

**Step 3:** The `math` import in `__init__.py` may still be needed by `lr_for_update` — leave it alone for now.

**Step 4:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: same baseline pass/fail as Task 1.

---

## Task 4: Extract `lr_schedule.py`

**Goal:** Move `lr_for_update` into its own module.

**Files:**
- Create: `python/train/ppo_recurrent/lr_schedule.py`
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create `lr_schedule.py`:

```python
"""Per-update learning-rate schedule for PPO.

Pure function — takes the current update index and hyperparameters, returns
the LR to apply. Supported schedules: constant, linear, cosine, all with
optional linear warmup.
"""

from __future__ import annotations

import math


def lr_for_update(
    update_idx: int,
    total_updates: int,
    *,
    base_lr: float,
    schedule: str = "constant",
    lr_final_ratio: float = 1.0,
    warmup_updates: int = 0,
) -> float:
    # ... (paste the exact body from __init__.py lines 113–168)
```

**Step 2:** In `__init__.py`, delete `lr_for_update`, delete the now-unused `import math` at the top of the file (verify it's not referenced anywhere else in `__init__.py` first — `_LOG2` is now imported, not recomputed), and add:

```python
from train.ppo_recurrent.lr_schedule import lr_for_update
```

**Step 3:** Run the PPO tests, especially the LR schedule test.

```bash
cd python && pytest tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: all LR schedule tests pass.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py -v
```

Expected: same baseline as Task 1.

---

## Task 5: Extract `config.py`

**Goal:** Move `PPOConfig` dataclass into its own module.

**Files:**
- Create: `python/train/ppo_recurrent/config.py`
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create `config.py`:

```python
"""Hyperparameter dataclass for the recurrent PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    # ... (paste the exact body from __init__.py lines 50–75, including docstring)
```

**Step 2:** In `__init__.py`, delete the `PPOConfig` definition and the now-unused `from dataclasses import dataclass` import (check first). Add:

```python
from train.ppo_recurrent.config import PPOConfig
```

**Step 3:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: same baseline.

---

## Task 6: Extract `trainer.py`

**Goal:** Move `PPOTrainer` (the ~380-line class) into its own module. This is the largest extraction and the one most likely to expose a missing import.

**Files:**
- Create: `python/train/ppo_recurrent/trainer.py`
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create `trainer.py`. Required top-level imports (determined by reading the current class body):

```python
"""Recurrent PPO trainer (CleanRL-style) for xushi2 Phase-2.

See the package ``__init__`` docstring for the invariant contract this
class implements.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import SyncVectorEnv

from train.models import ActorCritic, build_model
from train.rollout_buffer import RolloutBuffer
from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.losses import _masked_mean, _tanh_squashed_logprob


class PPOTrainer:
    # ... (paste the exact body from __init__.py lines 171–552)
```

Notes on imports above:
- `F` (`torch.nn.functional`) is imported at the top of the current file but **not referenced inside `PPOTrainer`** (grep confirmed — only inside other functions). Do NOT import `F` here.
- `Normal` from `torch.distributions` is imported at file top but also **not referenced inside `PPOTrainer`** (it's used by `ActorCritic` in `models.py`). Do NOT import `Normal` here.
- Before committing this task, grep the pasted class body to confirm every free name resolves: `grep -nE "\b(F|Normal|math|Path)\b" python/train/ppo_recurrent/trainer.py` — any hit means the import list above is incomplete.

**Step 2:** In `__init__.py`, delete the entire `PPOTrainer` class. Add:

```python
from train.ppo_recurrent.trainer import PPOTrainer
```

Also delete any now-unused imports in `__init__.py` (e.g., `SyncVectorEnv`, `nn`, `build_model`, `RolloutBuffer`) — only if they're genuinely unused after this move.

**Step 3:** Run the invariants test — this is the strongest regression gate for `PPOTrainer`.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py -v
```

Expected: same baseline. If anything changes, the imports or class body didn't transfer cleanly — diff the moved class against the original.

---

## Task 7: Extract `evaluate.py`

**Goal:** Move `evaluate_policy` into its own module.

**Files:**
- Create: `python/train/ppo_recurrent/evaluate.py`
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create `evaluate.py`:

```python
"""Greedy evaluation helper for trained recurrent policies."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from train.models import ActorCritic


def evaluate_policy(
    model: ActorCritic,
    env_fn: Callable[[], gym.Env],
    num_episodes: int,
    seed: int,
) -> float:
    # ... (paste the exact body from __init__.py lines 558–590, including docstring)
```

**Step 2:** In `__init__.py`, delete `evaluate_policy`. Add:

```python
from train.ppo_recurrent.evaluate import evaluate_policy
```

**Step 3:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: same baseline.

---

## Task 8: Extract `orchestration.py`

**Goal:** Move the training orchestration functions (`_make_ppo_config`, `_save_checkpoint`, `_run_variant`, `train_from_config`) into their own module. Opt-in cleanup: rename `_make_ppo_config` → `make_ppo_config` (the leading underscore was there because it was a module-private helper; now that orchestration is its own module with a clear public entry point `train_from_config`, the underscore is noise on an internal helper that doesn't need hiding). Leave `_save_checkpoint` and `_run_variant` with underscores — they remain internal implementation details of orchestration.

**Files:**
- Create: `python/train/ppo_recurrent/orchestration.py`
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Create `orchestration.py`:

```python
"""Top-level training orchestration for Phase-2 memory-toy runs.

Wraps two ``PPOTrainer`` invocations (recurrent + feedforward variants),
periodic evaluation, and best-eval checkpointing into a single
``train_from_config`` entry point.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import torch

from train.models import ActorCritic
from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.evaluate import evaluate_policy
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.ppo_recurrent.trainer import PPOTrainer


def make_ppo_config(config: dict, *, use_recurrence: bool) -> PPOConfig:
    # ... (paste body from __init__.py lines 593–619, was _make_ppo_config)


def _save_checkpoint(model: ActorCritic, path: Path, ckpt_config: dict) -> None:
    # ... (paste body from __init__.py lines 622–627)


def _run_variant(
    config: dict,
    *,
    use_recurrence: bool,
    output_dir: Path,
) -> float:
    # ... (paste body from __init__.py lines 630–749)
    # IMPORTANT: inside this function, change the single call site from
    # ``_make_ppo_config(...)`` to ``make_ppo_config(...)`` to match the rename.
    # Also: the ``import copy`` inside the function body can be removed since
    # we're importing ``copy`` at module top now.


def train_from_config(config: dict) -> dict[str, float]:
    # ... (paste body from __init__.py lines 752–768)
```

**Step 2:** In `__init__.py`, delete all four functions. Add:

```python
from train.ppo_recurrent.orchestration import (
    _run_variant,
    _save_checkpoint,
    make_ppo_config,
    train_from_config,
)
```

**Step 3:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: same baseline.

---

## Task 9: Clean up `__init__.py`

**Goal:** After Tasks 3–8, `__init__.py` contains only imports. Rewrite it as a clean public-API surface with the original module docstring preserved.

**Files:**
- Modify: `python/train/ppo_recurrent/__init__.py`

**Step 1:** Replace `__init__.py` with:

```python
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
```

Notes:
- Internal helpers (`_masked_mean`, `_tanh_squashed_logprob`, `_ATANH_EPS`, `_LOG2`, `_run_variant`, `_save_checkpoint`, `make_ppo_config`) are NOT re-exported. They live in their submodules and callers inside the package import them directly.
- If any caller outside the package imports a private name from `train.ppo_recurrent` (unlikely but check with grep below), either (a) add it to `__all__` and the re-exports here, or (b) update the caller.

**Step 2:** Verify no external caller imports a now-unexported symbol.

```bash
cd python && grep -rn "from train.ppo_recurrent import" . --include="*.py"
```

Expected set of imports (from the reference table above): `PPOConfig`, `PPOTrainer`, `lr_for_update`, `train_from_config`. Any other symbol requires adding to `__all__` or fixing the caller.

**Step 3:** Run the PPO tests.

```bash
cd python && pytest tests/test_ppo_recurrent_invariants.py tests/test_ppo_recurrent_lr_schedule.py -v
```

Expected: same baseline.

---

## Task 10: Full regression run

**Goal:** Confirm the whole test suite matches the Task 1 baseline, not just the PPO tests.

**Step 1:** Run the full suite.

```bash
cd python && pytest
```

Expected: identical pass/fail set as Task 1's full-suite baseline. Any new failure is a regression introduced by the split — bisect by reverting tasks until it reappears.

**Step 2:** Run ruff to confirm the new files pass lint (catches unused imports left over from the moves).

```bash
cd python && ruff check train/ppo_recurrent/
```

Expected: clean. Fix any flagged unused imports.

**Step 3:** Import smoke test — confirm every public symbol is reachable.

```bash
cd python && python -c "from train.ppo_recurrent import PPOConfig, PPOTrainer, lr_for_update, train_from_config, evaluate_policy; print('ok')"
```

Expected: `ok`.

---

## Task 11: Final commit

Per user preference, this is the single commit for the whole split.

**Step 1:** Stage and review.

```bash
git status
git diff --stat
```

Expected: one deletion (`python/train/ppo_recurrent.py`), several additions (`python/train/ppo_recurrent/__init__.py`, `config.py`, `losses.py`, `lr_schedule.py`, `trainer.py`, `evaluate.py`, `orchestration.py`), one modification (`python/pyproject.toml`).

**Step 2:** Commit (user drives this step per their workflow preference — do not auto-commit).

---

## Rollback

If any task's test run diverges from the Task 1 baseline and the cause isn't obvious within a few minutes, `git checkout -- python/train/ppo_recurrent*` restores the pre-split state. The split is designed to be restartable — each task is a self-contained extraction.
