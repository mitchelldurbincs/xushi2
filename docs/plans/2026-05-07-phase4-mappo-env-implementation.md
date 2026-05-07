# Phase 4 MAPPO Env — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `python/envs/phase4_mappo.py` exposing `Phase4MappoEnv` — a 3v3 MAPPO-shaped Gymnasium env with per-agent obs/action/reward and a separate caller-buffered `build_critic_obs` hook. First consumer of the prior slice's `build_critic_obs` (already in `main`).

**Architecture:** New module, fresh class, does NOT extend the 1v1 `XushiEnv`. Re-uses `_build_config`, `RewardCalculator`, and `_cpp.scripted_bot_action`. The env forces `cfg.team_size = 3` at `reset()`, builds a `(3, 31)` actor obs each tick (one per own-team slot), drives all 3 enemy slots with the same scripted bot, and exposes `build_critic_obs(out)` as a separate post-step method.

**Tech Stack:** Python 3.10+, Gymnasium, numpy, pybind11 to C++ sim.

**Reference:** All API decisions, layout rationale, and field-by-field test contracts live in `docs/plans/2026-05-07-phase4-mappo-env-design.md`. Consult the design doc when in doubt — do not re-derive.

**Worktree:** `.worktrees/phase4-mappo-env` on branch `phase4-mappo-env`, based off main's `d0bec8a`. All commands below are relative to that worktree root unless stated otherwise.

**Commit cadence:** Per user preference, do NOT commit per task. The user batches the whole delta at end-of-feature. Each task's "verify pass" step is the checkpoint.

---

## Task 0: Worktree baseline check

**Goal:** Confirm clean build + all tests pass before any change. Without this, mid-plan failures are ambiguous.

**Step 1: Configure CMake and build**

```powershell
cmake -S . -B build -DXUSHI2_BUILD_VIEWER=OFF -DXUSHI2_BUILD_TESTS=ON -DXUSHI2_BUILD_PYTHON_MODULE=ON
cmake --build build --parallel
```

**Step 2: Run C++ tests**

```powershell
ctest --test-dir build --output-on-failure
```

Expected: 94/94 PASS (per the prior critic-obs slice).

**Step 3: Run Python tests**

```powershell
cd python; python -m pytest tests -q
```

Expected: 124/124 PASS (per the prior critic-obs slice).

**If any baseline test fails:** Stop and report. Do not begin implementation.

---

## Task 1: Skeleton — class, `__init__`, spaces

**Goal:** Land enough of `Phase4MappoEnv` to construct, define spaces, and fail loudly on `step()` before `reset()`. No reset/step body yet.

**Files:**
- Create: `python/envs/phase4_mappo.py`
- Create: `python/tests/test_phase4_mappo_env.py`

**Step 1: Write failing tests**

In `python/tests/test_phase4_mappo_env.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM


def _make_sim_cfg() -> dict:
    # Match the shape used by other env tests; copy from
    # python/tests/test_phase3_ranger_env.py or test_xushi_env.py
    # for the canonical mechanics block. Adjust if the field names
    # have drifted.
    return {
        "mechanics": {
            "round_length_ticks": 1800,
            "respawn_ticks": 60,
            # ... etc, matching whatever existing tests use
        },
        "seed": 0,
    }


def _make_env(opponent_bot: str = "noop", **kwargs) -> Phase4MappoEnv:
    return Phase4MappoEnv(_make_sim_cfg(), opponent_bot=opponent_bot, **kwargs)


def test_construct_observation_space_shape_is_3_by_actor_dim():
    env = _make_env()
    assert env.observation_space.shape == (3, ACTOR_PHASE1_DIM)
    assert env.observation_space.dtype == np.float32


def test_construct_action_space_shape_is_3_by_6():
    env = _make_env()
    assert env.action_space.shape == (3, 6)
    assert env.action_space.dtype == np.float32


def test_step_before_reset_raises():
    env = _make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((3, 6), dtype=np.float32))


def test_invalid_opponent_bot_raises():
    with pytest.raises(ValueError, match="opponent_bot"):
        Phase4MappoEnv(_make_sim_cfg(), opponent_bot="not_a_real_bot")


def test_invalid_learner_team_raises():
    with pytest.raises(ValueError, match="learner_team"):
        Phase4MappoEnv(_make_sim_cfg(), opponent_bot="noop", learner_team="C")
```

NOTE: `_make_sim_cfg`'s exact field names depend on what `_build_config` accepts. Look at an existing env test (`python/tests/test_phase3_ranger_env.py` or `test_xushi_env.py`) and copy the canonical mechanics block. If the existing tests use a helper, prefer importing that.

**Step 2: Run; verify FAIL**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: ImportError on `envs.phase4_mappo`.

**Step 3: Implement skeleton**

Create `python/envs/phase4_mappo.py`:

```python
"""3v3 MAPPO-shaped Gymnasium env (Phase 4).

Per-agent (3, 31) observations and (3, 6) actions, with a separate
post-step `build_critic_obs(out)` hook that writes 135 floats into a
caller-provided buffer. Drives the C++ sim with team_size=3.

See docs/plans/2026-05-07-phase4-mappo-env-design.md for layout
rationale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from xushi2 import xushi2_cpp as _cpp
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.reward import RewardCalculator
from xushi2.runner import _build_config

__all__ = ["Phase4MappoEnv", "VALID_OPPONENT_BOTS"]

VALID_OPPONENT_BOTS: frozenset[str] = frozenset({
    "walk_to_objective", "hold_and_shoot", "basic", "noop",
})

_AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH

_AIM_DELTA_LIMIT = float(np.pi / 4.0)


class Phase4MappoEnv(gym.Env):
    metadata = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = ACTOR_PHASE1_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if opponent_bot not in VALID_OPPONENT_BOTS:
            raise ValueError(
                f"unknown opponent_bot {opponent_bot!r}; "
                f"valid: {sorted(VALID_OPPONENT_BOTS)}"
            )
        if learner_team not in ("A", "B"):
            raise ValueError(
                f"learner_team must be 'A' or 'B', got {learner_team!r}"
            )

        self._sim_cfg = dict(sim_cfg)
        self._opponent_bot = opponent_bot
        self._learner_team_str = learner_team
        self._learner_team = (
            _cpp.Team.A if learner_team == "A" else _cpp.Team.B
        )
        self._own_slots: tuple[int, int, int] = (
            (0, 1, 2) if learner_team == "A" else (3, 4, 5)
        )
        self._enemy_slots: tuple[int, int, int] = (
            (3, 4, 5) if learner_team == "A" else (0, 1, 2)
        )

        self._sim: _cpp.Sim | None = None
        self._reward_cfg = dict(reward_cfg or {})
        self._reward_calc = RewardCalculator(**self._reward_cfg)

        self._actor_obs_buf = np.zeros(
            (3, ACTOR_PHASE1_DIM), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3, ACTOR_PHASE1_DIM),
            dtype=np.float32,
        )
        low = np.tile(
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (3, 1),
        )
        high = np.tile(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (3, 1),
        )
        self.action_space = spaces.Box(
            low=low, high=high, shape=(3, 6), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("reset() is implemented in Task 2")

    def step(self, action):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        raise NotImplementedError("step() is implemented in Task 3")

    def build_critic_obs(self, out):
        raise NotImplementedError("build_critic_obs() is implemented in Task 4")

    def close(self) -> None:
        self._sim = None
```

**Step 4: Run; verify PASS**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: 5 tests PASS. (`test_step_before_reset_raises` should land its `RuntimeError` before hitting the `NotImplementedError`.)

**Step 5: Smoke the existing suite**

```powershell
cd python; python -m pytest tests -q
```

Expected: 129 PASS (124 baseline + 5 new). No regressions.

---

## Task 2: `reset` body

**Goal:** Make `reset(seed)` build the sim with `team_size=3`, populate the actor obs buffer, and return correct shapes.

**Files:**
- Modify: `python/envs/phase4_mappo.py` — `reset` body, plus `_build_actor_obs_all` and `_make_info` helpers.
- Modify: `python/tests/test_phase4_mappo_env.py` — add tests 1, 9.

**Step 1: Write failing tests**

Append to `test_phase4_mappo_env.py`:

```python
def test_reset_returns_correct_shapes():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert obs.dtype == np.float32
    assert info["tick"] == 0
    assert info["winner"] == "Neutral"
    assert info["learner_team"] == "A"


def test_team_b_learner_resets_with_correct_own_slots():
    env = _make_env(learner_team="B")
    assert env._own_slots == (3, 4, 5)
    assert env._enemy_slots == (0, 1, 2)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert info["learner_team"] == "B"
```

**Step 2: Run; verify FAIL**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py::test_reset_returns_correct_shapes tests/test_phase4_mappo_env.py::test_team_b_learner_resets_with_correct_own_slots -v
```

Expected: `NotImplementedError` from `reset`.

**Step 3: Verify `_build_config` doesn't already set `team_size`**

Read `python/xushi2/runner.py` and grep for `team_size`:

```powershell
Select-String -Path python/xushi2/runner.py -Pattern "team_size"
```

If `team_size` is set inside `_build_config` from `sim_cfg`, plan to either (a) make the env raise if `sim_cfg` carries `team_size`, or (b) overwrite it after `_build_config`. The design doc prefers (b) with a defensive raise on `__init__` if `sim_cfg` has `team_size`. Adjust below if needed.

**Step 4: Implement `reset` and helpers**

Replace the `reset`/`step`/`close` block with:

```python
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if "team_size" in self._sim_cfg:
            raise ValueError(
                "sim_cfg must not carry 'team_size'; the env owns this knob"
            )

        cfg = _build_config(self._sim_cfg, seed_override=seed)
        cfg.team_size = 3
        self._sim = _cpp.Sim(cfg)
        self._reward_calc.reset(self._sim)
        self._build_actor_obs_all()
        return self._actor_obs_buf.copy(), self._make_info()

    def _build_actor_obs_all(self) -> None:
        for i, slot in enumerate(self._own_slots):
            _cpp.build_actor_obs(self._sim, slot, self._actor_obs_buf[i])

    def _make_info(self) -> dict[str, Any]:
        s = self._sim
        winner = s.winner
        if winner == _cpp.Team.A:
            winner_str = "A"
        elif winner == _cpp.Team.B:
            winner_str = "B"
        else:
            winner_str = "Neutral"
        return {
            "tick": int(s.tick),
            "state_hash": int(s.state_hash),
            "team_a_score": float(s.team_a_score),
            "team_b_score": float(s.team_b_score),
            "team_a_kills": int(s.team_a_kills),
            "team_b_kills": int(s.team_b_kills),
            "winner": winner_str,
            "learner_team": self._learner_team_str,
        }
```

**Step 5: Run; verify PASS**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: 7 PASS.

---

## Task 3: `step` body

**Goal:** Make `step(action)` advance the sim, broadcast team reward to `(3,)`, and return correct shapes.

**Files:**
- Modify: `python/envs/phase4_mappo.py` — `step` body, `_action_to_cpp`.
- Modify: `python/tests/test_phase4_mappo_env.py` — add tests 2, 3, 12.

**Step 1: Write failing tests**

```python
def test_step_returns_correct_shapes_and_finite_values():
    env = _make_env(opponent_bot="noop")
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(
        np.zeros((3, 6), dtype=np.float32)
    )
    assert obs.shape == (3, ACTOR_PHASE1_DIM)
    assert reward.shape == (3,)
    assert reward.dtype == np.float32
    assert np.all(np.isfinite(reward))
    assert reward[0] == reward[1] == reward[2]
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)


@pytest.mark.parametrize("bad_shape", [(6,), (3,), (3, 5), (4, 6)])
def test_action_shape_validation_raises(bad_shape):
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(bad_shape, dtype=np.float32))


def test_reward_broadcast_is_team_reward_across_full_episode():
    env = _make_env(opponent_bot="noop")
    env.reset(seed=42)
    cumulative_per_agent = np.zeros(3, dtype=np.float32)
    for _ in range(2000):  # safely past round_length
        action = np.zeros((3, 6), dtype=np.float32)
        _, reward, term, trunc, _ = env.step(action)
        assert reward[0] == reward[1] == reward[2], "reward not broadcast"
        cumulative_per_agent += reward
        if term or trunc:
            break
    # All three agents accumulated the same total.
    assert cumulative_per_agent[0] == cumulative_per_agent[1]
    assert cumulative_per_agent[1] == cumulative_per_agent[2]
```

**Step 2: Run; verify FAIL**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: NotImplementedError from `step`.

**Step 3: Implement**

Replace the `step` body with:

```python
    def step(self, action: np.ndarray):
        if self._sim is None:
            raise RuntimeError("reset() must be called before step()")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3, 6):
            raise ValueError(
                f"action shape must be (3, 6), got {action.shape}"
            )

        actions = [_cpp.Action() for _ in range(_AGENTS_PER_MATCH)]
        for slot, a in zip(self._own_slots, action):
            actions[slot] = self._action_to_cpp(a)
        for enemy_slot in self._enemy_slots:
            actions[enemy_slot] = _cpp.scripted_bot_action(
                self._sim, enemy_slot, self._opponent_bot
            )

        self._sim.step_decision(actions)

        r_a, r_b = self._reward_calc.step(self._sim)
        team_reward = r_a if self._learner_team_str == "A" else r_b

        terminated = bool(self._sim.episode_over) and (
            self._sim.winner != _cpp.Team.Neutral
        )
        truncated = bool(self._sim.episode_over) and (
            self._sim.winner == _cpp.Team.Neutral
        )
        if terminated or truncated:
            ta, tb = self._reward_calc.add_terminal(self._sim)
            team_reward += ta if self._learner_team_str == "A" else tb

        reward = np.full(3, team_reward, dtype=np.float32)
        self._build_actor_obs_all()
        info = self._make_info()
        info["reward_team_a"] = float(r_a)
        info["reward_team_b"] = float(r_b)
        return self._actor_obs_buf.copy(), reward, terminated, truncated, info

    @staticmethod
    def _action_to_cpp(a: np.ndarray) -> "_cpp.Action":
        a = np.asarray(a, dtype=np.float32).reshape(6)
        a[:3] = np.clip(a[:3], -1.0, 1.0)
        a[3:] = np.clip(a[3:], 0.0, 1.0)
        act = _cpp.Action()
        act.move_x = float(a[0])
        act.move_y = float(a[1])
        act.aim_delta = float(a[2] * _AIM_DELTA_LIMIT)
        act.primary_fire = bool(a[3] >= 0.5)
        act.ability_1 = bool(a[4] >= 0.5)
        act.ability_2 = bool(a[5] >= 0.5)
        return act
```

**Step 4: Run; verify PASS**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: 11 PASS.

**Step 5: Sanity smoke on `RewardCalculator` 3v3 generalization**

If `test_reward_broadcast_is_team_reward_across_full_episode` passes with non-trivial reward (i.e. the episode actually terminated and produced a reward delta), `RewardCalculator` is generalizing correctly. If reward is exactly zero throughout AND no termination fires, that's a smoke-test failure to investigate (reward calc may be summing wrong over the 6-hero state).

---

## Task 4: `build_critic_obs`

**Goal:** Implement the caller-buffered critic-obs hook with shape/dtype validation. Verify it matches a direct `_cpp.build_critic_obs` call on the same sim.

**Files:**
- Modify: `python/envs/phase4_mappo.py` — `build_critic_obs` body.
- Modify: `python/tests/test_phase4_mappo_env.py` — add tests 5, 6, 7, 11.

**Step 1: Write failing tests**

```python
def test_build_critic_obs_writes_135_finite_floats_with_actor_prefix():
    env = _make_env()
    env.reset(seed=0)
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    env.build_critic_obs(out)
    assert np.all(np.isfinite(out))
    # First 31 floats must equal the actor obs of the first own slot.
    actor = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(env._sim, env._own_slots[0], actor)
    np.testing.assert_array_equal(out[:ACTOR_PHASE1_DIM], actor)


def test_build_critic_obs_before_reset_raises():
    env = _make_env()
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    with pytest.raises(RuntimeError, match="reset"):
        env.build_critic_obs(out)


@pytest.mark.parametrize("bad", [
    np.zeros(CRITIC_DIM - 1, dtype=np.float32),
    np.zeros((CRITIC_DIM, 1), dtype=np.float32),
    np.zeros(CRITIC_DIM, dtype=np.float64),
])
def test_build_critic_obs_buffer_validation(bad):
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.build_critic_obs(bad)


def test_critic_obs_team_b_uses_team_b_actor_prefix():
    env = _make_env(learner_team="B")
    env.reset(seed=0)
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    env.build_critic_obs(out)
    actor_b0 = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(env._sim, 3, actor_b0)
    np.testing.assert_array_equal(out[:ACTOR_PHASE1_DIM], actor_b0)
```

**Step 2: Run; verify FAIL**

Expected: NotImplementedError.

**Step 3: Implement**

Replace `build_critic_obs`:

```python
    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._sim is None:
            raise RuntimeError(
                "reset() must be called before build_critic_obs()"
            )
        if not isinstance(out, np.ndarray):
            raise ValueError("out must be an np.ndarray")
        if out.shape != (CRITIC_DIM,) or out.dtype != np.float32:
            raise ValueError(
                f"out must be float32 ndarray of shape ({CRITIC_DIM},), "
                f"got {out.shape} {out.dtype}"
            )
        _cpp.build_critic_obs(self._sim, self._learner_team, out)
```

**Step 4: Run; verify PASS**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: 16 PASS (5+2+3+1 new = 11 new + 5 from Task 1).

---

## Task 5: Determinism + smoke

**Goal:** Verify the gym env's seed plumbing produces bit-identical state-hash trajectories, and that 1000 random-action ticks don't crash.

**Files:**
- Modify: `python/tests/test_phase4_mappo_env.py` — add tests 8, 10.

**Step 1: Write failing tests**

```python
def test_determinism_two_envs_same_seed_same_state_hash():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(100, 3, 6)).astype(np.float32)
    actions[..., 3:] = (actions[..., 3:] > 0).astype(np.float32)

    env_a = _make_env()
    env_b = _make_env()
    _, info_a = env_a.reset(seed=42)
    _, info_b = env_b.reset(seed=42)
    assert info_a["state_hash"] == info_b["state_hash"]

    for action in actions:
        _, _, term_a, trunc_a, info_a = env_a.step(action)
        _, _, term_b, trunc_b, info_b = env_b.step(action)
        assert info_a["state_hash"] == info_b["state_hash"], (
            f"state_hash divergence at tick {info_a['tick']}"
        )
        if term_a or trunc_a:
            assert term_b == term_a and trunc_b == trunc_a
            break


def test_idle_1000_ticks_no_crash():
    env = _make_env(opponent_bot="basic")
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        action = rng.uniform(-1, 1, size=(3, 6)).astype(np.float32)
        action[:, 3:] = (action[:, 3:] > 0).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=rng.integers(0, 2**31 - 1))
```

**Step 2: Run**

```powershell
cd python; python -m pytest tests/test_phase4_mappo_env.py::test_determinism_two_envs_same_seed_same_state_hash tests/test_phase4_mappo_env.py::test_idle_1000_ticks_no_crash -v
```

Expected: PASS (no implementation needed if Tasks 1–4 are correct). If determinism fails, investigate the seed plumbing in `_build_config(sim_cfg, seed_override=seed)`. If 1000-tick smoke fails with an `X2_ENSURE` abort, that's a sim-side regression to triage.

---

## Task 6: Final verification

**Step 1: Full Python suite**

```powershell
cd python; python -m pytest tests -q
```

Expected: ≥ 137 PASS (124 baseline + ~13 new tests). No regressions.

**Step 2: Full C++ suite (sanity)**

```powershell
ctest --test-dir build --output-on-failure
```

Expected: 94/94 PASS. (No C++ files were touched; this is a paranoia check.)

**Step 3: Spot-check the public surface**

```powershell
cd python; python -c "from envs.phase4_mappo import Phase4MappoEnv, VALID_OPPONENT_BOTS; print(sorted(VALID_OPPONENT_BOTS))"
```

Expected: `['basic', 'hold_and_shoot', 'noop', 'walk_to_objective']`.

---

## Final hand-off

When Task 6 reports green, all the work is in the worktree on branch `phase4-mappo-env`. Per user preference, do NOT auto-commit. Stop here and ask the user how they want to integrate.

**Followups to flag for the user (not this slice):**
- `XushiVectorEnv` (audit item 4) — wraps N `Phase4MappoEnv` instances, returns `(actor_obs, critic_obs, …)` per env.
- MAPPO rollout buffer (item 5).
- Models split + MAPPOTrainer (items 6–7).
