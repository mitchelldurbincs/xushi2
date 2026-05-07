# Phase 4 — 3v3 MAPPO Env: Design

**Date:** 2026-05-07
**Status:** Design approved. Ready for implementation.
**Slice:** Audit item 3 from `2026-04-24-phase4-prep.md`.

## Goals & Scope

Add a 3v3 MAPPO-shaped Gymnasium env (`Phase4MappoEnv`) that drives the
C++ sim with `MatchConfig.team_size = 3` and exposes per-agent
observations + actions and a separate centralized-critic obs hook. This
is the first consumer of the new `build_critic_obs` (landed in the
prior slice) and the first env shape the future MAPPO trainer will see.

### In scope

1. New module `python/envs/phase4_mappo.py` and class
   `Phase4MappoEnv`. Fresh code — does NOT extend the 1v1-shaped
   `XushiEnv`. The audit explicitly recommends a fresh module.
2. Reuses `xushi2.runner._build_config`, `xushi2.reward.RewardCalculator`,
   and `xushi2_cpp.scripted_bot_action`. Does not reuse `XushiEnv`.
3. `reset(seed)` → `(actor_obs (3, 31) float32, info)`.
4. `step(action: (3, 6) float32)` → `(actor_obs (3, 31), reward (3,),
   terminated bool, truncated bool, info dict)`.
5. `build_critic_obs(out: ndarray) -> None` — caller-provided buffer of
   shape `(135,)` float32. Must be called after `reset()` or `step()`
   for the matching tick. Raises `RuntimeError` if `_sim is None`.
6. Opponent: single `opponent_bot: str` drives all 3 enemy slots
   independently each tick (each slot calls `scripted_bot_action`
   independently — they're not coordinated).
7. `learner_team` configurable to `"A"` or `"B"`.

### Out of scope (deferred Phase-4 slices)

- `XushiVectorEnv` — audit item 4. Builds on this.
- MAPPO rollout buffer / trainer / models split — items 5–7.
- Self-play opponent pool — item 10.
- Per-agent individualized rewards — kept as `(3,)` broadcast of the
  team scalar. Specializing later is a tensor-level change, not an API
  change.
- Any change to existing `XushiEnv` or `Phase3RangerEnv` — they remain
  the Phase 1–3 path, untouched.

### Success criteria

- `Phase4MappoEnv(sim_cfg, opponent_bot=…)` constructs and asserts the
  underlying sim has `team_size == 3`.
- 1000 random-action ticks complete without `X2_ENSURE` aborts.
- Two envs seeded identically produce bit-equal `info["state_hash"]`
  trajectories.
- `build_critic_obs` writes 135 finite floats; matches a direct
  `_cpp.build_critic_obs(self._sim, learner_team, …)` call on the same
  sim state.

## API surface

```python
class Phase4MappoEnv(gym.Env):
    metadata = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int   # = ACTOR_PHASE1_DIM (31)
    critic_obs_dim: int  # = CRITIC_DIM (135)
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,                # one of VALID_OPPONENT_BOTS
        learner_team: str = "A",          # "A" or "B"
        reward_cfg: dict[str, Any] | None = None,
    ) -> None: ...

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(3, ACTOR_PHASE1_DIM),      # (3, 31)
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=np.tile([-1, -1, -1, 0, 0, 0], (3, 1)).astype(np.float32),
        high=np.tile([ 1,  1,  1, 1, 1, 1], (3, 1)).astype(np.float32),
        shape=(3, 6),
        dtype=np.float32,
    )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(
        self, action: np.ndarray,         # (3, 6)
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]: ...

    def build_critic_obs(self, out: np.ndarray) -> None: ...

    def close(self) -> None: ...
```

### Decision rationale

**Why a separate `build_critic_obs` method instead of bundling the
critic obs into `step()`'s return tuple or `info` dict?**

- Cleanly separates the actor (decentralized, gym contract) and the
  critic (centralized-training only) surfaces.
- Plays nicely with generic gym tooling: `gym.Wrapper`, `RecordVideo`,
  eval harnesses don't need to know critic obs exists.
- Lazy: pure inference (eval, demos, replay) doesn't pay critic
  compute. Only the trainer pays.
- `info` dict is the wrong place for a 135-float ndarray; that
  serializes badly under `AsyncVectorEnv` (audit E3 explicitly flagged
  this).
- The vector env (next slice) will combine the two outputs at its
  layer with no extra cost.

**Why caller-provided buffer instead of return-allocating?**

- Mirrors the C++ binding's signature.
- Lets the vector env pre-allocate one `(N, 135)` buffer and slice into
  it, zero per-step allocations.
- Documents the post-step ordering invariant explicitly (the caller
  owns the buffer's lifetime, the env doesn't store it).

**Ordering invariant.** `build_critic_obs` must be called after
`reset()` or after the matching `step()` and before the next `step()`.
Otherwise the buffer reflects stale sim state. The env asserts
`self._sim is not None` but cannot enforce the within-tick ordering;
this is documented in the docstring and on the vector env contract
later.

## Internals

### Cached state

Three slot tuples set at `__init__`:
- `self._own_slots: tuple[int, int, int]` — `(0, 1, 2)` for
  `learner_team == "A"`, `(3, 4, 5)` for `"B"`.
- `self._enemy_slots: tuple[int, int, int]` — the complement.
- `self._learner_team: _cpp.Team` — `Team.A` or `Team.B`.

Reusable buffers:
- `self._actor_obs_buf: np.ndarray` of shape `(3, 31)` float32 —
  C-contiguous; per-row slicing is zero-copy for the C++ binding.
- `self._sim: _cpp.Sim | None` — set by `reset`, `None` between
  construction and first `reset`.

Reward calc:
- `self._reward_calc: RewardCalculator` — constructed at `__init__`,
  reset to the new sim each `reset()`.

### `reset`

```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    cfg = _build_config(self._sim_cfg, seed_override=seed)
    cfg.team_size = 3                       # force; refuse if caller set
    self._sim = _cpp.Sim(cfg)
    self._reward_calc.reset(self._sim)
    self._build_actor_obs_all()
    return self._actor_obs_buf.copy(), self._make_info()
```

`cfg.team_size = 3` is set after `_build_config`. If `_build_config`
already populates `team_size` from `sim_cfg`, the env should raise on
`__init__` to make the conflict loud. The Phase-1/2/3 callers don't
pass `team_size` so this constraint is non-breaking.

### `step`

```python
def step(self, action: np.ndarray):
    if self._sim is None:
        raise RuntimeError("reset() must be called before step()")
    if action.shape != (3, 6):
        raise ValueError(f"action shape must be (3, 6), got {action.shape}")

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
```

`_action_to_cpp(a: ndarray(6,)) -> _cpp.Action` mirrors
`Phase3RangerEnv._action_to_dict`: clip `[-1,1]` for `[move_x, move_y,
aim_delta_unit]`, scale `aim_delta` by `pi/4`, threshold `[primary_fire,
ability_1, ability_2]` at `>= 0.5` for the bool fields.

### `_build_actor_obs_all`

```python
def _build_actor_obs_all(self) -> None:
    for i, slot in enumerate(self._own_slots):
        _cpp.build_actor_obs(self._sim, slot, self._actor_obs_buf[i])
```

`self._actor_obs_buf` is C-contiguous (default `np.zeros`); per-row
slicing produces zero-copy views the C++ binding can write into.

### `build_critic_obs`

```python
def build_critic_obs(self, out: np.ndarray) -> None:
    if self._sim is None:
        raise RuntimeError("reset() must be called before build_critic_obs()")
    if out.shape != (CRITIC_DIM,) or out.dtype != np.float32:
        raise ValueError(
            f"out must be float32 ndarray of shape ({CRITIC_DIM},), "
            f"got {out.shape} {out.dtype}"
        )
    _cpp.build_critic_obs(self._sim, self._learner_team, out)
```

## Test plan

All tests in new file `python/tests/test_phase4_mappo_env.py`.

1. **`test_construct_and_reset_returns_correct_shapes`** — `obs.shape ==
   (3, 31)`, `obs.dtype == float32`. `info["tick"] == 0`,
   `info["winner"] == "Neutral"`.
2. **`test_step_returns_correct_shapes_and_finite_values`** — step with
   `np.zeros((3, 6))`. Assert returned shapes, finite values, broadcast
   invariant `reward[0] == reward[1] == reward[2]`.
3. **`test_action_shape_validation_raises`** — `(6,)`, `(3,)`, `(3, 5)`
   each raise `ValueError`.
4. **`test_step_before_reset_raises`** — fresh env's `step` raises
   `RuntimeError`.
5. **`test_build_critic_obs_writes_135_finite_floats`** — pre-allocated
   `out`, call after `reset(seed=0)`, assert all 135 entries finite,
   first 31 match `_cpp.build_actor_obs(sim, own_slots[0], …)`.
6. **`test_build_critic_obs_before_reset_raises`** — `RuntimeError`.
7. **`test_build_critic_obs_buffer_validation`** — wrong dtype
   (float64) or wrong shape ((134,), (135, 1)) raises `ValueError`.
8. **`test_determinism_two_envs_same_seed_same_state_hash`** — two
   envs, same seed, same fixed sequence of 100 random actions, identical
   `info["state_hash"]` at every tick.
9. **`test_team_b_learner_works_symmetrically`** —
   `learner_team="B"`, `own_slots == (3,4,5)`, idle step, shapes
   correct.
10. **`test_idle_1000_ticks_no_crash`** — 1000 random-action ticks via
    gym API, no exceptions. Auto-resets at terminal.
11. **`test_critic_obs_team_b_perspective_uses_team_b`** — for
    `learner_team="B"`, critic obs `[0:31]` matches
    `_cpp.build_actor_obs(sim, 3, …)`, not slot 0.
12. **`test_reward_broadcast_is_team_reward`** — drive a deterministic
    episode (e.g. `noop` opponent), after termination assert
    `reward[0] == reward[1] == reward[2]` and cumulative agent-0 reward
    equals reported per-team episode reward.

## Risks / things to verify during implementation

1. **`RewardCalculator` 3v3 generalization.** It iterates over hero
   state for kill/score aggregation. Should generalize from 1 to 3
   heroes per team naturally, but worth a smoke test as part of
   developing test 12. If broken, that's a sim-side fix outside this
   slice — flag and add a follow-up rather than expanding scope.
2. **`_build_config` and `team_size`.** Verify it does not silently
   override or set `team_size` from `sim_cfg`. If `sim_cfg` happens to
   carry `team_size`, raise — the env owns this knob.
3. **State-hash determinism under 3v3 with mixed scripted-bot
   opponents.** The C++ side is deterministic given identical inputs;
   the Python side just needs to feed identical action sequences. Test
   8 covers this.
4. **Phase-1/2/3 paths unaffected.** No changes to `XushiEnv` /
   `Phase3RangerEnv`. The 1v1 path's `team_size=1` default is
   bit-identical to before this slice (verified by the existing
   determinism / golden tests, which still pass).

## File changes

- New: `python/envs/phase4_mappo.py`
- New: `python/tests/test_phase4_mappo_env.py`
- No modifications to existing files.

## Implementation order

1. Skeleton: class, `__init__`, spaces, no `reset`/`step` body. Tests
   1 + 4 fail → implement → pass.
2. `reset`: build cfg, force `team_size=3`, construct sim, build actor
   obses. Tests 1, 2, 9 pass.
3. `step`: action assembly, sim step, reward broadcast, return. Tests
   2, 3, 12 pass.
4. `build_critic_obs`: validate buffer, delegate to C++. Tests 5, 6, 7,
   11 pass.
5. Determinism + smoke: tests 8, 10 pass.
6. Verify `RewardCalculator` works at 3v3 (smoke during test 12).

## Order of Phase-4 work after this slice

This design lands audit item 3. Remaining order:

4. Custom `XushiVectorEnv` (sync-only) — wraps N `Phase4MappoEnv`
   instances; combines `(actor_obs, critic_obs, …)` per env.
5. MAPPO rollout buffer with agent + team axes.
6. Split `models.py` into `Actor` (decentralized, recurrent) +
   `Critic` (centralized).
7. `MAPPOTrainer`.
8. Wire into orchestration; per-agent greedy eval.
9. Smoke-train Phase-4 vs scripted bots.
10. Self-play opponent pool.
11. (Later) async vector env once rollout dominates wall clock.
