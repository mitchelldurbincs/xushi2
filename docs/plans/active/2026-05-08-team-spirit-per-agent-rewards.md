# Team-Spirit + Per-Agent Rewards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire OpenAI Five-style per-agent shaped rewards and the `team_spirit` credit-assignment lever (with a 0.3 → 1.0 linear ramp over the first 30% of training) into the Phase 4 MAPPO path, so that `team_spirit` is a real lever and not a no-op.

**Architecture:** Three layers of change:
1. **C++ sim:** the per-slot lifetime kill/death counters already exist on `HeroState.kills` / `HeroState.deaths` (sim.h:90-91), and `process_deaths` already increments them (sim_combat.cpp:393, 405). All we need is two new public accessors on `Sim` that return `std::array<std::uint32_t, kAgentsPerMatch>` snapshots gathered from `state_.heroes[i].kills` / `.deaths`, plus pybind11 bindings.
2. **`RewardCalculator`:** opt-in `per_agent_rewards=True` flag. When true, `step()` returns `(np.ndarray(3,), np.ndarray(3,))`; when false (default), it returns `(float, float)` exactly as today. The per-agent formula credits kill bonus to the killer's slot and death penalty to the victim's slot directly — no "enemy mirror" subtraction (it would double-count the same event). Score-tick reward is split among own-team slots by their on-point share; enemy score is split team-uniformly across the 3 own slots. `sum_i r_i_team_A` equals today's team scalar exactly when `kill_bonus == death_penalty` (the default 0.25 == 0.25); call this out as the zero-sum precondition.
3. **Trainer + env:** `Phase4MappoEnv` constructs `RewardCalculator(..., per_agent_rewards=True)` and passes the per-agent vector through. XushiEnv and Phase11CurrentSelfplayMappoEnv are untouched in this plan — the default-false flag keeps them on the scalar API. MAPPO config gets `team_spirit_initial`, `team_spirit_final`, `team_spirit_ramp_fraction`; trainer applies the interpolation `r_i ← (1-τ)·r_i + τ·mean(r_team)` with τ recomputed per update. Both `XushiVectorEnv` (sync) and `XushiAsyncVectorEnv` (multiprocess) gain a `set_team_spirit(value)` method so the trainer doesn't reach into `.envs` directly.

**Why OAI Five-style and not the rl_design.md exact text:** rl_design §5 prescribes 0.3→0.9; OAI Five's *Dota 2 with Large Scale Deep RL* paper used 0.3→1.0. The user has chosen to follow OAI Five — we use 1.0 as the ramp target and update `docs/rl_design.md` §5 to match.

**Tech Stack:** C++20, pybind11, numpy, PyTorch, pytest, GoogleTest, CMake.

**Scope guardrails:**
- Phase 4 only (Ranger 3v3, fixed map, no fog). Phase 5+ envs inherit naturally because they all carry forward `Phase4MappoEnv`'s reward path.
- Terminal reward (`±10` win/loss) stays team-uniform across the 3 agents — terminal is by definition a team outcome and `team_spirit` applies to shaped rewards only (rl_design §5).
- Per-episode `±shaping_clip` stays a per-*team* cumulative cap, applied to the team-mean of per-agent shaped rewards (preserves the existing magnitude invariant; terminal continues to dominate).

**Pre-flight:** This plan touches C++/bindings/Python. The user is on `goalTest` and commits the whole delta at the end (skip per-task commits). If you want filesystem isolation, branch a worktree off `goalTest` first via `superpowers:using-git-worktrees`; otherwise work in-place on `goalTest`.

---

## Task 1: Per-slot kill/death accessors on `Sim`

**Why:** `HeroState` already has lifetime `kills` / `deaths` per slot (sim.h:90-91), and `process_deaths` already increments them (sim_combat.cpp:393, 405). All we need is two `Sim` accessors that surface this data as fixed-size arrays so the Python `RewardCalculator` can diff them. **No new `MatchState` fields, no `process_deaths` edits, no `sim_spawn_reset` edits** — that data already exists and is already maintained.

**Files:**
- Modify: `src/sim/include/xushi2/sim/sim.h` — add public `kills_by_slot()` / `deaths_by_slot()` accessor declarations to `Sim`, alongside `team_a_kills()` / `team_b_kills()`.
- Modify: `src/sim/src/sim.cpp` — implement the accessors by copying from `state_.heroes[i].kills` / `.deaths`.
- Test: `tests/sim/test_combat.cpp` — contract test asserting that after slot 0 kills slot 3, `sim.kills_by_slot()[0] == 1` and `sim.deaths_by_slot()[3] == 1`.

**Step 1: Read the existing `team_a_kills` accessor to mirror its style**

```
grep -n "team_a_kills" src/sim/include/xushi2/sim/sim.h src/sim/src/sim.cpp
```

Read 5 lines around each match. The new accessors should follow the same `const noexcept` signature pattern.

**Step 2: Find a combat test that already drives a kill in `tests/sim/test_combat.cpp`**

```
grep -n "TEST(.*Kill\|kills += 1\|deaths += 1\|.kills ==\|.deaths ==" tests/sim/test_combat.cpp
```

Pick the closest existing test that already positions slot 0 to kill slot 3 (or any pair) and uses the existing scaffolding. Mirror its setup — do NOT write a new helper from scratch if one exists.

**Step 3: Write the failing C++ test**

Add to `tests/sim/test_combat.cpp` (use whichever pre-positioning helper the existing combat tests use):

```cpp
TEST(SimAccessors, KillsAndDeathsBySlotMirrorHeroStateCounters) {
    // Reuse whatever existing test scaffolding fires a fatal revolver shot
    // from slot 0 at slot 3. Confirm that after the kill, the new accessors
    // surface the per-slot counters that already live on HeroState.
    Sim sim = /* existing 3v3 setup helper, low HP on slot 3 */;
    /* existing fire_at(slot=0, target=3) helper + step_decision */;

    const auto kills = sim.kills_by_slot();
    const auto deaths = sim.deaths_by_slot();
    EXPECT_EQ(kills[0], 1u);
    EXPECT_EQ(deaths[3], 1u);
    for (int s : {1, 2, 4, 5}) {
        EXPECT_EQ(kills[s], 0u);
    }
    for (int s : {0, 1, 2, 4, 5}) {
        EXPECT_EQ(deaths[s], 0u);
    }
}
```

**Step 4: Run, verify compile failure**

```
cmake --build build --target test_combat -j
```

Expected: compile error — `kills_by_slot` not declared.

**Step 5: Add the accessor declarations**

In `src/sim/include/xushi2/sim/sim.h`, alongside `team_a_kills()`:

```cpp
std::array<std::uint32_t, kAgentsPerMatch> kills_by_slot() const noexcept;
std::array<std::uint32_t, kAgentsPerMatch> deaths_by_slot() const noexcept;
```

**Step 6: Implement the accessors**

In `src/sim/src/sim.cpp`, mirroring the existing `team_a_kills` impl:

```cpp
std::array<std::uint32_t, kAgentsPerMatch> Sim::kills_by_slot() const noexcept {
    std::array<std::uint32_t, kAgentsPerMatch> out{};
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        out[i] = state_.heroes[i].kills;
    }
    return out;
}

std::array<std::uint32_t, kAgentsPerMatch> Sim::deaths_by_slot() const noexcept {
    std::array<std::uint32_t, kAgentsPerMatch> out{};
    for (std::size_t i = 0; i < kAgentsPerMatch; ++i) {
        out[i] = state_.heroes[i].deaths;
    }
    return out;
}
```

These are pure read-side helpers — `HeroState.kills` / `.deaths` are already incremented by `process_deaths` and already zeroed when `MatchState` is constructed (the `HeroState` struct default-initializes them to 0 and reset rebuilds heroes from scratch — confirm by reading `sim_spawn_reset.cpp` if uncertain).

**Step 7: Run the test, verify it passes**

```
cmake --build build --target test_combat -j
ctest --test-dir build -R "SimAccessors.KillsAndDeathsBySlot" --output-on-failure
```

Expected: 1 passed.

**Step 8: Run the full combat test suite for regressions**

```
ctest --test-dir build -R "Combat" --output-on-failure
```

Expected: all combat tests still pass (read-only accessors can't regress behavior).

---

## Task 2: Expose per-slot counters via pybind11

**Why:** Python's `RewardCalculator` needs to read `sim.kills_by_slot` / `sim.deaths_by_slot` like it currently reads `sim.team_a_kills`.

**Files:**
- Modify: `src/python_bindings/module.cpp` (around line 212–213) — add bindings.
- Test: `python/tests/test_bindings_obs.py` (or a new `test_bindings_kills.py` if `test_bindings_obs.py` is observation-scoped — check first).

**Step 1: Find where `team_a_kills` is bound**

Already located: `src/python_bindings/module.cpp:212`. Read the surrounding context (~10 lines) so the new bindings match style.

**Step 2: Write the failing pytest**

`Phase1MechanicsConfig` rejects sentinel values (sim.h:107-119), so the test config MUST include a `mechanics` block. Look at how the existing `test_phase4_mappo_env.py` or another already-passing test builds a 3v3 sim config, and copy that mechanics block.

```python
def test_sim_exposes_kills_and_deaths_by_slot():
    # Use whatever helper / fixture already exists in this test file for
    # building a valid 3v3 MatchConfig. Required mechanics fields:
    #   revolver_damage_centi_hp, revolver_fire_cooldown_ticks,
    #   revolver_hitbox_radius, respawn_ticks
    sim = _make_3v3_sim(seed=0)  # or equivalent existing fixture
    kills = sim.kills_by_slot
    deaths = sim.deaths_by_slot
    assert hasattr(kills, "__len__")
    assert len(kills) == _cpp.AGENTS_PER_MATCH
    assert len(deaths) == _cpp.AGENTS_PER_MATCH
    assert all(k == 0 for k in kills)
    assert all(d == 0 for d in deaths)
```

If the chosen test file does not already have a 3v3 sim helper, prefer adding the test to a file that does (e.g. `test_phase4_mappo_env.py`'s setup) rather than reinventing the config builder.

**Step 3: Run the test, verify it fails**

```
cd python && python -m pytest tests/test_bindings_obs.py::test_sim_exposes_kills_and_deaths_by_slot -v
```

Expected: `AttributeError: ... has no attribute 'kills_by_slot'`.

**Step 4: Add the bindings**

In `src/python_bindings/module.cpp`, alongside the `team_a_kills` / `team_b_kills` bindings:

```cpp
.def_property_readonly("kills_by_slot",
    [](const xushi2::sim::Sim& s) {
        const auto a = s.kills_by_slot();
        return std::vector<std::uint32_t>(a.begin(), a.end());
    })
.def_property_readonly("deaths_by_slot",
    [](const xushi2::sim::Sim& s) {
        const auto a = s.deaths_by_slot();
        return std::vector<std::uint32_t>(a.begin(), a.end());
    })
```

(Returning `std::vector` rather than `std::array` so pybind11 produces a Python `list`, matching what numpy/list-comprehension callers expect. Confirm by checking what pattern other array-returning bindings in this file use; if they return `py::array_t<uint32_t>`, follow that instead.)

**Step 5: Rebuild the extension**

```
cmake --build build --target xushi2_cpp -j
```

**Step 6: Run the test, verify it passes**

```
cd python && python -m pytest tests/test_bindings_obs.py::test_sim_exposes_kills_and_deaths_by_slot -v
```

Expected: 1 passed.

---

## Task 3: Per-agent shaped rewards in `RewardCalculator` (opt-in flag, no team_spirit yet)

**Why:** Before introducing the `team_spirit` mixin, we need per-agent reward streams that *are different from each other* — otherwise the lever has nothing to interpolate between. This task adds a `per_agent_rewards: bool = False` flag to `RewardCalculator`. Default-false preserves the today scalar API exactly so XushiEnv (`python/xushi2/env.py:166`) and `Phase11CurrentSelfplayMappoEnv` (`python/envs/phase11_current_selfplay_mappo.py:210`) remain untouched. Only Phase 4 opts in (Task 4).

**Per-agent formula** (team A, agent i ∈ {0, 1, 2}):
```
r_i = +kill_bonus    * kills_delta_by_slot[i]                        # own kill credit
      - death_penalty * deaths_delta_by_slot[i]                       # own death penalty
      + score_per_sec * own_score_seconds * on_point_share_i          # own score share
      - score_per_sec * enemy_score_seconds / 3                       # enemy score (team-uniform)
```

Where `on_point_share_i` is i's share of the team's on-point presence this tick (0 if no own-team agents are on point; otherwise `on_point_i / sum_j on_point_j`). On-point read via `_cpp.build_actor_obs(sim, slot, buf)[on_point_slice]`, like the existing on-point shaping path. **No "enemy mirror" subtraction for kills/deaths** — earlier draft of the plan added `−kill_bonus * enemy_kills_delta / 3 + death_penalty * enemy_deaths_delta / 3`, which double-counts the same kill event (it's already credited as the killer's own kill on the killing team and as the victim's own death on the dying team). Dropping it preserves today's per-event magnitude exactly.

**Sum-invariant:** When `kill_bonus == death_penalty` (the default 0.25 == 0.25), `sum_i r_i_team_A` exactly equals today's team scalar `raw_a` (and `sum_i r_i_team_B = -sum_i r_i_team_A`). Verify this in tests. If a future change breaks `kill_bonus == death_penalty`, the per-agent path can still be used but the cross-team zero-sum no longer holds — leave a comment noting this precondition.

**Cross-check** (slot 0 of A kills slot 3 of B, no scoring this tick):
- Today's scalar path: `raw_a = +0.25`, `raw_b = -0.25`. Sum = 0. ✓
- New per-agent path:
  - Team A: `r_0 = +0.25, r_1 = 0, r_2 = 0` → sum = +0.25. ✓
  - Team B: `r_0 = 0, r_1 = 0, r_2 = -0.25` (slot 3 in absolute = slot 0 in team B's local indexing) → sum = -0.25. ✓
  - Total across both teams: 0. ✓

**Files:**
- Modify: `python/xushi2/reward.py` — add `per_agent_rewards: bool = False` to `__init__`. When True, `step()` returns `(np.ndarray(3,), np.ndarray(3,))`; when False (default) returns `(float, float)` exactly as today. Read `kills_by_slot` / `deaths_by_slot` only on the per-agent path.
- Modify: `python/tests/test_reward.py` — add new per-agent attribution tests under `per_agent_rewards=True`. **Existing scalar-API tests stay as-is** (they construct `RewardCalculator()` with defaults, which keeps the scalar path).
- Phase 4 env wiring is pushed to Task 4.

**Step 1: Extend `_FakeSim` with per-slot counters**

In `python/tests/test_reward.py`:

```python
class _FakeSim:
    def __init__(self):
        self.team_a_score_ticks = 0
        self.team_b_score_ticks = 0
        self.team_a_kills = 0
        self.team_b_kills = 0
        self.kills_by_slot = [0, 0, 0, 0, 0, 0]
        self.deaths_by_slot = [0, 0, 0, 0, 0, 0]
        self.episode_over = False
        self.winner = _cpp.Team.Neutral
```

**Step 2: Write a failing per-agent kill-attribution test**

Note `b` is indexed in absolute slot order (3, 4, 5) but the per-agent vector is team-local (length 3, indices 0..2). Decide and document which convention the API uses; below assumes `b[0]` is the team-B agent at absolute slot 3.

```python
def test_per_agent_kill_credits_only_killer_and_victim():
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    # Slot 0 (team A) kills slot 3 (team B). team_a_kills/team_b_kills also
    # ticked since they reflect totals.
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, b = rc.step(sim)
    assert a.shape == (3,)
    assert b.shape == (3,)
    # Team A: only slot 0 (the killer) gets the kill_bonus.
    assert a[0] == pytest.approx(0.25)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)
    # Team B: only the local-slot-0 agent (absolute slot 3, the victim) gets
    # the death_penalty.
    assert b[0] == pytest.approx(-0.25)
    assert b[1] == pytest.approx(0.0)
    assert b[2] == pytest.approx(0.0)
    # Sum invariants (kill_bonus == death_penalty default).
    assert a.sum() == pytest.approx(0.25)
    assert b.sum() == pytest.approx(-0.25)
    assert a.sum() + b.sum() == pytest.approx(0.0)
```

Run: `cd python && python -m pytest tests/test_reward.py::test_per_agent_kill_credits_only_killer_and_victim -v`
Expected: FAIL — `per_agent_rewards` kwarg doesn't exist.

**Step 3: Implement the per-agent path under the opt-in flag**

In `python/xushi2/reward.py`:

- Extend `_EventCounters` with two `np.ndarray` fields (length 6, dtype int64) for `a_kills_by_slot_total` and `a_deaths_by_slot_total` — read from `sim.kills_by_slot` / `sim.deaths_by_slot` only when `per_agent_rewards=True`. **Do not break the scalar path**: scalar callers never pass these so they stay at `np.zeros(6, dtype=np.int64)`.
- Add `per_agent_rewards: bool = False` to `__init__`. Store as `self._per_agent`. Validate via type/range as needed.
- In `__init__`, if `self._per_agent`, allocate the slot-array fields on `self._prev`. Otherwise leave the existing scalar `_EventCounters` shape alone.
- `_read_counters(sim)`: gain a branch — when `self._per_agent`, also populate `kills_by_slot` / `deaths_by_slot` arrays. Otherwise unchanged.
- `step(sim)`:
  - When `not self._per_agent`: existing code path, unchanged. Returns `(float, float)`.
  - When `self._per_agent`: returns `(np.ndarray(3,), np.ndarray(3,))`. Compute per-agent shaped using:
    ```python
    a_kills_delta_slot = now.a_kills_by_slot - self._prev.a_kills_by_slot   # shape (6,)
    a_deaths_delta_slot = now.a_deaths_by_slot - self._prev.a_deaths_by_slot

    raw_a = np.zeros(3, dtype=np.float32)
    raw_b = np.zeros(3, dtype=np.float32)

    # Own kill credit: each team's slots 0..2 (A) and 3..5 (B).
    raw_a += self._kill_bonus * a_kills_delta_slot[0:3]
    raw_b += self._kill_bonus * a_kills_delta_slot[3:6]
    # Own death penalty.
    raw_a -= self._death_penalty * a_deaths_delta_slot[0:3]
    raw_b -= self._death_penalty * a_deaths_delta_slot[3:6]

    # Score: own split by on-point share; enemy share team-uniform 1/3.
    a_score_seconds = (now.a_score_ticks - self._prev.a_score_ticks) / float(TICK_HZ)
    b_score_seconds = (now.b_score_ticks - self._prev.b_score_ticks) / float(TICK_HZ)
    raw_a += self._score_per_second * a_score_seconds * self._on_point_share(sim, (0, 1, 2))
    raw_b += self._score_per_second * b_score_seconds * self._on_point_share(sim, (3, 4, 5))
    raw_a -= self._score_per_second * b_score_seconds / 3.0
    raw_b -= self._score_per_second * a_score_seconds / 3.0
    ```
    Where `_on_point_share(sim, slots)` returns a length-3 array summing to 1.0 (or all-equal 1/3 fallback when no team member is on point). It must use the existing obs-buf path when those buffers are allocated; if obs bufs are not allocated (e.g. unit-test `_FakeSim`), default to even split — add a hasattr/try guard like the existing `_team_on_point_fraction`.
  - Distance shaping (`_distance_shaping_coef > 0`), on-point shaping (`_on_point_shaping_coef > 0`), and time penalty: keep computing the per-team scalar exactly as today, then add to `raw_a` / `raw_b` *uniformly across the 3 slots* (broadcast). These are diagnostic / probe knobs and don't decompose per agent.
  - **Clip on team sum, not mean.** Today the clip caps the team scalar cumulatively at `±shaping_clip`. For the per-agent path, the equivalent is: cap `cum_a += r_a.sum()` to `±shaping_clip`, then if the cap binds this step, scale `r_a *= clipped_step / unclipped_step`. Same for B. This preserves today's invariant exactly (when `kill_bonus == death_penalty`, today's scalar reward = today's `raw_a.sum()`).
    ```python
    team_step_a = float(raw_a.sum())
    clipped_step_a = self._apply_clip(team_step_a, "a")  # scalar like today
    if abs(team_step_a) > 1e-12 and clipped_step_a != team_step_a:
        raw_a *= (clipped_step_a / team_step_a)
    elif clipped_step_a == 0.0 and team_step_a == 0.0:
        pass
    # else: no scaling needed (clipped == raw)
    ```
- `add_terminal(sim)`:
  - When `not self._per_agent`: returns `(float, float)` as today.
  - When `self._per_agent`: returns `(np.full(3, ta, dtype=np.float32), np.full(3, tb, dtype=np.float32))`. Terminal stays uniform regardless of team_spirit (Task 5).

**Step 4: Add additional per-agent tests**

```python
def test_per_agent_score_split_equally_when_no_on_point_data():
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_score_ticks = _cpp.TICK_HZ  # 1 second of A scoring
    a, b = rc.step(sim)
    # Own A: +0.01 split equally → +0.00333 each. Enemy A on B: -0.01/3 each.
    np.testing.assert_allclose(a, [0.01/3, 0.01/3, 0.01/3], atol=1e-6)
    np.testing.assert_allclose(b, [-0.01/3, -0.01/3, -0.01/3], atol=1e-6)
    assert a.sum() == pytest.approx(0.01)
    assert b.sum() == pytest.approx(-0.01)


def test_per_agent_sum_invariant_matches_scalar_path_for_kill_only():
    """When kill_bonus == death_penalty (default), per-agent sums must
    equal the scalar-path totals event-for-event."""
    sim = _FakeSim()
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]

    rc_scalar = RewardCalculator()
    rc_scalar.reset(_FakeSim())
    a_scalar, b_scalar = rc_scalar.step(sim)

    rc_vec = RewardCalculator(per_agent_rewards=True)
    rc_vec.reset(_FakeSim())
    a_vec, b_vec = rc_vec.step(sim)

    assert a_vec.sum() == pytest.approx(a_scalar)
    assert b_vec.sum() == pytest.approx(b_scalar)


def test_per_agent_terminal_is_uniform():
    rc = RewardCalculator(per_agent_rewards=True)
    sim = _FakeSim()
    rc.reset(sim)
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    ta, tb = rc.add_terminal(sim)
    np.testing.assert_array_equal(ta, np.full(3, 10.0, dtype=np.float32))
    np.testing.assert_array_equal(tb, np.full(3, -10.0, dtype=np.float32))
```

**Step 5: Verify _FakeSim still works without `kills_by_slot` for scalar path**

The existing `_FakeSim` in `test_reward.py` does not set `kills_by_slot`. The scalar path must not access this attribute. Add a guard or simply only read it when `self._per_agent`. Confirm by running the existing reward tests unchanged.

**Step 6: Run the full reward test suite**

```
cd python && python -m pytest tests/test_reward.py -v
```

Expected: all existing tests pass (untouched), plus the new per-agent tests pass.

---

## Task 4: Wire per-agent rewards through `Phase4MappoEnv`

**Why:** The env's `step()` currently does `np.full(3, team_reward)` to broadcast a scalar. Phase4MappoEnv opts in to `per_agent_rewards=True` and drops the broadcast. **Phase 1 (`xushi2/env.py`) and Phase 11 (`envs/phase11_current_selfplay_mappo.py`) keep their existing scalar contracts** — they construct `RewardCalculator()` without the flag and remain unchanged.

**Files:**
- Modify: `python/envs/phase4_mappo.py` — `RewardCalculator(...)` construction (add `per_agent_rewards=True`) and `step()` reward construction.
- Test: `python/tests/test_phase4_mappo_env.py` (or wherever the env-level test lives — find with `grep -rn "Phase4MappoEnv" python/tests/`).

**Step 1: Find existing env test**

```
grep -rn "Phase4MappoEnv\|phase4_mappo" python/tests/
```

Locate the file that already tests `Phase4MappoEnv.step()`. Read one of its tests for the existing pattern.

**Step 2: Smoke test asserting reward.shape and per-slot variance is possible**

```python
def test_phase4_step_reward_shape_is_per_agent():
    env = _make_phase4_env_for_test()  # use whatever helper / fixture already exists
    env.reset(seed=0)
    actions = np.zeros((3, _ACTION_DIM), dtype=np.float32)
    obs, reward, term, trunc, info = env.step(actions)
    assert reward.shape == (3,)
    assert reward.dtype == np.float32
    # Per-team scalars in info preserved for downstream logging.
    assert isinstance(info["reward_team_a"], float)
    assert isinstance(info["reward_team_b"], float)
```

The unit-level kill-attribution is already covered by `test_reward.py` (Task 3). This test only confirms the env propagates per-agent values and preserves the scalar info keys.

**Step 3: Run the failing test**

```
cd python && python -m pytest tests/test_phase4_mappo_env.py::test_phase4_step_reward_shape_is_per_agent -v
```

Expected: PASS already on shape (since old code broadcasts to length 3) but the info-key assertion may fail depending on existing structure. Tighten to the failure mode that actually exists.

**Step 4: Implement the change**

In `python/envs/phase4_mappo.py`:

1. Construction: `self._reward_calc = RewardCalculator(per_agent_rewards=True, **self._reward_cfg)` (placement of the kwarg depends on whether `_reward_cfg` already contains it; safer to pop or assert-not-present).
2. In `step()`, replace:

```python
r_a, r_b = self._reward_calc.step(self._sim)
team_reward = r_a if self._learner_team_str == "A" else r_b

terminated = ...
truncated = ...
if terminated or truncated:
    ta, tb = self._reward_calc.add_terminal(self._sim)
    team_reward += ta if self._learner_team_str == "A" else tb

reward = np.full(3, team_reward, dtype=np.float32)
```

with:

```python
r_a, r_b = self._reward_calc.step(self._sim)  # (3,), (3,)
own_reward = r_a if self._learner_team_str == "A" else r_b

terminated = ...
truncated = ...
if terminated or truncated:
    ta, tb = self._reward_calc.add_terminal(self._sim)  # (3,), (3,)
    own_reward = own_reward + (ta if self._learner_team_str == "A" else tb)

reward = np.asarray(own_reward, dtype=np.float32)
```

3. Keep `info["reward_team_a"] / reward_team_b` as **scalars** (sum across agents) to preserve callers:

```python
info["reward_team_a"] = float(r_a.sum())
info["reward_team_b"] = float(r_b.sum())
```

Cross-reference what existing callers do with these keys (`grep -rn "reward_team_a" python/`) before flipping them to arrays — keeping scalars is the lowest-blast-radius choice.

**Step 5: Run the env test**

```
cd python && python -m pytest tests/test_phase4_mappo_env.py -v
```

Expected: PASS.

**Step 6: Run reward + phase4 tests together**

```
cd python && python -m pytest tests/test_reward.py tests/test_phase4_mappo_env.py -v
```

Expected: all pass.

**Step 7: Run the full Python suite to detect downstream breakage**

```
cd python && python -m pytest tests -q
```

Possible regressions:
- MAPPO trainer tests (`test_mappo_loss_mask.py`, `test_mappo_matrix_eval.py`) — reward values flow into advantage computation; non-uniform per-agent rewards may change a numerical assertion. Update if the test was asserting *identical-broadcast* (an artifact of the old impl); leave alone if it asserts a real spec invariant.
- Snapshot/replay tests — replay metadata may serialize reward summaries.
- Phase 11 env tests — should be unaffected since Phase 11 stays on the scalar `RewardCalculator()` path.

For each regression: read the failing test, decide whether it was asserting *identical-broadcast* or *real spec invariant*. Update accordingly. Note each fix in the task summary; do not blanket-update tests to silence them.

---

## Task 5: `team_spirit` mixin (constant τ, no ramp yet)

**Why:** With per-agent rewards in place, the team_spirit interpolation becomes a real lever. Wire the math first as a constant-τ knob; the ramp comes in Task 6.

**Files:**
- Modify: `python/xushi2/reward.py` — add `team_spirit` parameter to `RewardCalculator.__init__`, apply mixin in `step()` to shaped rewards (NOT terminal).
- Modify: `python/tests/test_reward.py` — add tests for τ=0, τ=0.5, τ=1.

**Note:** team_spirit only applies on the per-agent path. With `per_agent_rewards=False`, the kwarg is silently a no-op (it's a per-agent mixin; nothing to mix on a scalar).

**Step 1: Write failing tests**

Cross-check expected values against the corrected per-agent formula (no enemy mirror term):
- One-A-kills-one-B event: indiv = `[0.25, 0.0, 0.0]`, mean = `0.25/3 ≈ 0.0833`.
- τ=0: `r = [0.25, 0, 0]`, sum 0.25.
- τ=1: `r = [0.0833, 0.0833, 0.0833]`, sum 0.25.
- τ=0.5: `r = [0.5*0.25+0.5*0.0833, 0.5*0+0.5*0.0833, 0.5*0+0.5*0.0833] = [0.1667, 0.0417, 0.0417]`, sum 0.25.

```python
def test_team_spirit_zero_preserves_individual_rewards():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # τ=0: pure individual; slot 0 has the entire kill bonus.
    assert a[0] == pytest.approx(0.25)
    assert a[1] == pytest.approx(0.0)
    assert a[2] == pytest.approx(0.0)

def test_team_spirit_one_collapses_to_team_mean():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=1.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # τ=1: all slots receive team mean = 0.25 / 3.
    assert a[0] == pytest.approx(0.25 / 3.0)
    assert a[1] == pytest.approx(0.25 / 3.0)
    assert a[2] == pytest.approx(0.25 / 3.0)
    assert a.sum() == pytest.approx(0.25)  # invariant preserved

def test_team_spirit_half_is_exact_interpolation():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.5)
    sim = _FakeSim()
    rc.reset(sim)
    sim.team_a_kills = 1
    sim.kills_by_slot = [1, 0, 0, 0, 0, 0]
    sim.deaths_by_slot = [0, 0, 0, 1, 0, 0]
    a, _ = rc.step(sim)
    # τ=0.5: r_i = 0.5*indiv_i + 0.5*mean
    # indiv = [0.25, 0, 0], mean = 0.25/3 ≈ 0.0833
    # mixed = [0.5*0.25 + 0.5*0.0833, 0.5*0 + 0.5*0.0833, ...]
    #       = [0.1667, 0.0417, 0.0417]
    expected_mean = 0.25 / 3.0
    assert a[0] == pytest.approx(0.5 * 0.25 + 0.5 * expected_mean)
    assert a[1] == pytest.approx(0.5 * expected_mean)
    assert a[2] == pytest.approx(0.5 * expected_mean)
    assert a.sum() == pytest.approx(0.25)  # invariant preserved

def test_team_spirit_does_not_mix_terminal():
    rc = RewardCalculator(per_agent_rewards=True, team_spirit=0.0)
    sim = _FakeSim()
    rc.reset(sim)
    sim.episode_over = True
    sim.winner = _cpp.Team.A
    ta, tb = rc.add_terminal(sim)
    # Terminal is uniformly broadcast regardless of team_spirit.
    assert (ta == 10.0).all()
    assert (tb == -10.0).all()
```

**Step 2: Run tests, verify they fail**

```
cd python && python -m pytest tests/test_reward.py -k team_spirit -v
```

Expected: 4 failures (parameter doesn't exist).

**Step 3: Implement**

In `RewardCalculator.__init__`, add:

```python
team_spirit: float = 0.0,
```

with a validator: `if not 0.0 <= team_spirit <= 1.0: raise ValueError(...)`. Store as `self._team_spirit`.

Add a setter for ramp use in Task 6:

```python
def set_team_spirit(self, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"team_spirit must be in [0, 1], got {value}")
    self._team_spirit = float(value)
```

In `step()` on the per-agent branch, after computing `raw_a`, `raw_b` (each shape `(3,)`), and *before* the cumulative-team-sum clip, apply:

```python
if self._team_spirit > 0.0:
    mean_a = raw_a.mean()
    raw_a = (1.0 - self._team_spirit) * raw_a + self._team_spirit * mean_a
    mean_b = raw_b.mean()
    raw_b = (1.0 - self._team_spirit) * raw_b + self._team_spirit * mean_b
```

Note: this transform preserves `raw_a.sum()` exactly (mean is invariant under affine combination with itself), so the team-sum clip from Task 3 sees the same value before and after the mixin. Sum invariant preserved.

When `per_agent_rewards=False`, `team_spirit` has no effect (no per-agent vector to mix). Make sure the validator still runs and stores the value, but the scalar branch doesn't reference it — we want construction with a non-default `team_spirit` on the scalar path to be a clean no-op rather than an error, since Phase 11 might inherit a config that has both flags set.

`add_terminal` is unchanged — terminal already returns uniform values regardless of team_spirit.

**Step 4: Run tests**

```
cd python && python -m pytest tests/test_reward.py -k team_spirit -v
```

Expected: 4 passed.

---

## Task 6: `team_spirit` ramp scheduling in MAPPO trainer + vector wrappers

**Why:** OAI Five ramps τ from 0.3 → 1.0 over training. The trainer computes τ each update and needs to push it to every env in the vector — both `XushiVectorEnv` (sync, in-process) and `XushiAsyncVectorEnv` (multi-process, IPC). Direct `for env in vec_env.envs` only works for sync; the async wrapper has no `.envs` accessor.

**Files:**
- Modify: `python/train/mappo.py` — `MappoConfig` adds 3 fields; training loop computes and pushes τ per update.
- Modify: `python/train/phases.py` (or wherever MappoConfig is built from yaml) — config parsing for the 3 new fields.
- Modify: `python/envs/phase4_mappo.py` — add `set_team_spirit(value)` method on the env.
- Modify: `python/xushi2/vector_env.py` — add `set_team_spirit(value)` method to BOTH `XushiVectorEnv` (iterates `self._envs`) and `XushiAsyncVectorEnv` (dispatches via existing IPC pattern).
- Test: `python/tests/test_mappo_team_spirit_ramp.py` (new).

**Step 1: Write failing test for the ramp schedule**

```python
# python/tests/test_mappo_team_spirit_ramp.py
import pytest
from train.mappo import compute_team_spirit  # to be added

def test_team_spirit_at_start_is_initial():
    assert compute_team_spirit(update=0, total=1000, initial=0.3, final=1.0,
                               ramp_fraction=0.3) == pytest.approx(0.3)

def test_team_spirit_at_ramp_end_is_final():
    assert compute_team_spirit(update=300, total=1000, initial=0.3, final=1.0,
                               ramp_fraction=0.3) == pytest.approx(1.0)

def test_team_spirit_after_ramp_end_holds_at_final():
    assert compute_team_spirit(update=999, total=1000, initial=0.3, final=1.0,
                               ramp_fraction=0.3) == pytest.approx(1.0)

def test_team_spirit_midpoint_linear():
    # At update=150 of 1000 with ramp=30%, ramp progress = 150/300 = 0.5
    # τ = 0.3 + 0.5 * (1.0 - 0.3) = 0.65
    assert compute_team_spirit(update=150, total=1000, initial=0.3, final=1.0,
                               ramp_fraction=0.3) == pytest.approx(0.65)

def test_team_spirit_ramp_fraction_zero_jumps_to_final():
    # No ramp at all → start at final.
    assert compute_team_spirit(update=0, total=1000, initial=0.3, final=1.0,
                               ramp_fraction=0.0) == pytest.approx(1.0)
```

**Step 2: Run, verify fails**

```
cd python && python -m pytest tests/test_mappo_team_spirit_ramp.py -v
```

Expected: ImportError on `compute_team_spirit`.

**Step 3: Implement `compute_team_spirit` in `python/train/mappo.py`**

```python
def compute_team_spirit(*, update: int, total: int, initial: float,
                        final: float, ramp_fraction: float) -> float:
    """Linear ramp from `initial` at update 0 to `final` at
    `ramp_fraction * total`, then held at `final`."""
    if ramp_fraction <= 0.0:
        return final
    ramp_end_update = max(1, int(ramp_fraction * total))
    if update >= ramp_end_update:
        return final
    progress = update / ramp_end_update
    return initial + progress * (final - initial)
```

**Step 4: Run, verify passes**

```
cd python && python -m pytest tests/test_mappo_team_spirit_ramp.py -v
```

Expected: 5 passed.

**Step 5: Add config fields to `MappoConfig`**

In `python/train/mappo.py`, add to the dataclass:

```python
team_spirit_initial: float = 0.0   # default OFF for back-compat
team_spirit_final: float = 0.0
team_spirit_ramp_fraction: float = 0.3
```

**Step 6: Add `set_team_spirit` proxy on `Phase4MappoEnv`**

In `python/envs/phase4_mappo.py`:

```python
def set_team_spirit(self, value: float) -> None:
    self._reward_calc.set_team_spirit(value)
```

**Step 7: Add `set_team_spirit` on both vector wrappers**

In `python/xushi2/vector_env.py`:

- `XushiVectorEnv.set_team_spirit(value)`: iterate `self._envs` (or whatever the in-proc env list is named — verify by reading the class) and call `env.set_team_spirit(value)` on each.
- `XushiAsyncVectorEnv.set_team_spirit(value)`: dispatch to each subprocess worker. Read the existing IPC pattern in this class (look for how `seed` or `reset` push commands to workers) and mirror it. If the existing pattern uses a Pipe/Queue command enum, add a `SET_TEAM_SPIRIT` command variant.

If the async wrapper does not yet have a generic command-passing mechanism (i.e. it only proxies `step` and `reset` directly), add a minimal one rather than reaching across processes. Document the choice inline. **Confirm with a unit test that the async wrapper actually pushes the value through** — easy to fail silently otherwise.

```python
# python/tests/test_vector_env.py (append)
def test_xushi_vector_env_set_team_spirit_propagates_sync():
    env = XushiVectorEnv([_make_phase4_env, _make_phase4_env], critic_obs_dim=...)
    env.set_team_spirit(0.7)
    # Reach into env._envs to assert each got the value.
    for sub in env._envs:
        assert sub._reward_calc._team_spirit == pytest.approx(0.7)

def test_xushi_async_vector_env_set_team_spirit_propagates():
    env = XushiAsyncVectorEnv([_make_phase4_env, _make_phase4_env], critic_obs_dim=...)
    env.set_team_spirit(0.7)
    # Async: round-trip through workers via a getter command, OR use a
    # follow-up step that depends on team_spirit and assert behavior changes.
    # If a `get_team_spirit` accessor doesn't exist, it's reasonable to add one
    # alongside set_team_spirit so the test has a probe.
```

**Step 8: Apply τ in the training loop**

Find the training-loop start of each update in `mappo.py`. Before rollout collection, compute and apply τ:

```python
tau = compute_team_spirit(
    update=update_idx,
    total=cfg.run.total_updates,
    initial=cfg.mappo.team_spirit_initial,
    final=cfg.mappo.team_spirit_final,
    ramp_fraction=cfg.mappo.team_spirit_ramp_fraction,
)
vec_env.set_team_spirit(tau)
metrics["team_spirit"] = tau   # log it
```

Note the trainer no longer touches `vec_env.envs` directly; the wrapper is the API boundary.

**Step 9: Add config-parsing for the 3 fields**

Find where `MappoConfig` is constructed from yaml (likely `python/train/phases.py` or similar). Pass through `team_spirit_initial`, `team_spirit_final`, `team_spirit_ramp_fraction` from the `ppo:` config block.

**Step 10: Add an integration smoke test**

```python
# python/tests/test_mappo_team_spirit_ramp.py (append)
def test_phase4_smoke_with_team_spirit_ramp_runs():
    # Run the phase4_mappo_smoke config with team_spirit_initial=0.3,
    # team_spirit_final=1.0, total_updates=2. Assert no crash and
    # metrics["team_spirit"] is logged.
    ...
```

(Pattern this on the existing `test_phase_registry.py::test_phase4_*` smoke tests — copy structure.)

**Step 11: Run all new tests + full reward + mappo path**

```
cd python && python -m pytest tests/test_reward.py tests/test_mappo_team_spirit_ramp.py tests/test_vector_env.py tests/test_phase_registry.py -v
```

Expected: all pass.

---

## Task 7: End-to-end smoke + config update for `phase4_mappo_basic.yaml`

**Why:** Update the actual training config so `team_spirit` is enabled for any future Phase 4 run. Also functions as the integration smoke for the whole stack.

**Files:**
- Modify: `experiments/configs/phase4_mappo_basic.yaml` — add the 3 team_spirit fields under `ppo:`.
- Modify: `experiments/configs/phase4_mappo_smoke.yaml` — same, with τ=1.0 (collapses to team mean → exercises the path without changing the smoke's expected behavior since smoke is 2 updates).

**Step 1: Update `phase4_mappo_basic.yaml`**

Under `ppo:`, add:

```yaml
team_spirit_initial: 0.3
team_spirit_final: 1.0
team_spirit_ramp_fraction: 0.3
```

(Per the user's "follow OpenAI Five" choice: 0.3→1.0, ramp over first 30% of training.)

**Step 2: Run a 2-update smoke to verify nothing crashes**

```
cd python && python -m train.train --config ../experiments/configs/phase4_mappo_smoke.yaml
```

Expected: smoke completes, `metrics["team_spirit"]` is in the logged output, run dir contains `ckpt_final.pt`.

**Step 3: Spot-check that team_spirit appears in metrics**

Read `python/runs/phase4_mappo_smoke/mappo/log.jsonl` (or whatever log file exists) and confirm `team_spirit` key is present and its value matches what `compute_team_spirit` returns for the 2-update run.

---

## Task 8: Update `docs/rl_design.md` to reflect actual implementation

**Why:** rl_design §5 currently says ramp 0.3 → 0.9. We chose 0.3 → 1.0 (OAI Five). Spec-and-code drift bites later.

**Files:**
- Modify: `docs/rl_design.md` §5 — change "ramp linearly to `0.9`" to "ramp linearly to `1.0` (OpenAI Five-style; collapse to team mean once team coordination is the dominant signal)".

**Step 1: Make the edit**

Find the line `start Phase 4 at team_spirit = 0.3 and ramp linearly to 0.9` and update.

**Step 2: Add a sentence on per-agent attribution**

After the `team_spirit` paragraph, add:

> Per-agent individual rewards are derived from per-slot kill/death attribution (sim exposes `kills_by_slot` / `deaths_by_slot`). Score-tick rewards are split among on-point teammates by their per-tick on-point share. The team-level zero-sum invariant is preserved by subtracting a 1/Nteam share of enemy-team mirror events from each agent.

---

## Final verification

```
cd python && python -m pytest tests -q
ctest --test-dir build --output-on-failure
cmake --build build-viewer --target xushi2_viewer --parallel
```

Expected:
- Python suite passes (226+ tests after additions).
- C++ suite passes (123+ tests after the per-slot counter addition).
- Viewer compiles (no viewer changes in this plan, but the build-graph touches `MatchState` which the viewer may include transitively).

---

## Out of scope (explicit non-goals)

- Per-component reward weighting beyond what already exists (rl_design §5 has fixed weights; not retuning them in this plan).
- Per-agent terminal rewards (terminal stays uniform; rl_design §5 mandates this).
- Phase 5+ env updates beyond what flows naturally from `Phase4MappoEnv` inheritance — if a Phase 5+ test breaks, fix only the broken assertion, do not redesign the inheriting envs in this plan.
- The Phase 4 numerical gate definition (separate plan: "define Phase 4 acceptance gate").
- Behavioral metrics (objective contest time, ally-deaths-while-isolated, cooldown-waste; separate plan).
- Scaled training run (separate plan; needs all three of: this plan, the gate definition, and behavioral metrics first).
