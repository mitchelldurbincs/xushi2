# Plan — Phase 1 Environment

**Status:** proposed
**Target branch:** `claude/review-and-plan-GRZ9e`
**Date:** 2026-04-21

## Context

The repo today is the Phase 0 skeleton described in `docs/rl_design.md` §6:
two Rangers on a 50×50 arena, movement + aim + cooldown decrement, plus
`state_hash()` and a determinism test. Everything downstream is stubbed —
`src/bots/src/bot.cpp` returns no-op actions, `python/train/train.py` prints
"Phase 0 not implemented", there is no observation builder, no env wrapper,
and the `tests/observations/test_actor_leak.cpp` file referenced in the docs
does not exist.

The curriculum's stated next step is **Phase 1** — feedforward PPO, flat
observation, 1v1 Ranger on a fixed map (`docs/rl_design.md` §6,
`docs/observation_spec.md` §"Phase 1"). Phase 1 splits naturally into two
units of work:

1. **Phase 1 environment** — gameplay, observations, scripted bot, env wrapper,
   tests. Can be validated end-to-end with random-action rollouts, no PPO.
2. **Phase 1 trainer** — CleanRL-style feedforward PPO against that env.

**This plan covers unit (1) only.** The PPO trainer lands as a follow-up so
that training failures can be binary-searched against a known-good env.

Why first: every later phase depends on working Ranger combat, an objective
scoring loop, and the actor/critic obs split. Those are the large net-new
pieces; once they exist, Phase 2+ can reuse the same env with additional
wrappers.

## Design decisions (confirmed)

- **Scope:** Phase 1 environment only. PPO trainer deferred.
- **Default `action_repeat`:** 3 ticks (10 Hz policy rate). Both 2 and 3
  remain supported by `Sim` and covered by tests.
- **Fog of war off** (as spec'd for Phase 1). No LoS raycasting yet — that
  lands at Phase 7. The actor obs is computed from ground truth for now;
  the *separation* of actor and critic builders is what Phase 1
  establishes.
- **1v1 Ranger.** Slots 0 (Team A damage) and 3 (Team B damage) are the
  only `present` heroes. The other four `HeroState` slots stay zeroed with
  `present = false`.
- **Map:** the same 50×50 arena already in `MapBounds`. No interior walls.
  Control point is a fixed circle at map center.

## Work breakdown

Each step is a reviewable commit. Order matters: earlier steps are
dependencies of later ones.

### Step 1 — Ranger combat constants and reload state machine

**Files:**
- `src/common/include/xushi2/common/limits.hpp` — add Ranger/objective
  constants (see below)
- `src/sim/include/xushi2/sim/sim.h` — add small nested structs if the
  reload / respawn state grows beyond loose fields (per coding_philosophy
  §"Make invalid states hard to represent")
- `src/sim/src/sim.cpp` — implement tick-pipeline steps 7, 10, 11, 12, 13

**Constants to add** (values from `docs/game_design.md` §3, §6):
```
kRangerMaxHp              = 150.0
kRangerRevolverDamage     = 70.0    // 2-shot kill, matches Cassidy archetype
kRangerRevolverRangeU     = 22.0
kRangerRevolverFireRateHz = 2.0     // ticks between shots = TICK_HZ / fire_rate
kRangerAutoReloadTicks    = ...     // from game_design reload state machine
kCombatRollDashU          = 3.0
kCombatRollCdTicks        = 5 * TICK_HZ
kRespawnTicks             = 7 * TICK_HZ   // middle of 6-8s window
kCaptureTicks             = 8 * TICK_HZ
kWinTicks                 = 100 * TICK_HZ
kObjectiveUnlockTicks     = 15 * TICK_HZ
kObjectiveRadiusU         = 3.0
```
Exact values for fire-rate / damage / auto-reload should be cross-checked
against `game_design.md` §6 when implementing.

**Sim changes:**
- Add `ticks_since_last_shot`, `auto_reload_progress_ticks` to `HeroState`
  (or wrap them in a `RangerReloadState` struct per coding_philosophy §3).
- Extend `state_hash()` manifest to include the new fields, and update
  `docs/determinism_rules.md` §"state_hash() manifest".
- Wire Combat Roll impulse: dash 3u along `normalize(move_vec)` if nonzero,
  else along aim; refill magazine; set cd.
- Wire Revolver fire: hitscan vs the other Ranger; if within `kRangerRevolverRangeU`
  and not obstructed (no walls in Phase 1), apply damage; decrement magazine.

### Step 2 — Death, respawn, and spawn wave

**Files:** `src/sim/src/sim.cpp`, `src/sim/include/xushi2/sim/sim.h`

- On health reaching 0 (pipeline step 13): set `alive = false`, set
  `respawn_tick = state.tick + kRespawnTicks`.
- On `state.tick == respawn_tick` (pipeline step 14): respawn at
  team-fixed spawn point, reset hp/magazine/cooldowns, `alive = true`.
- Add `X2_INVARIANT(hero.health >= 0)` and `respawn_tick` monotonicity
  check.

### Step 3 — Objective state machine

**Files:** `src/sim/src/sim.cpp`

Implement the 5 cases from `docs/game_design.md` §3 exactly (integer-tick
arithmetic, no floats). Pipeline step 15. Occupancy test is "alive hero
within `kObjectiveRadiusU` of map center." Extract as a small helper:
```
struct ObjectiveOccupancy { uint8_t a_count; uint8_t b_count; };
ObjectiveOccupancy count_occupancy(const MatchState&, const MapBounds&);
```

Update `Sim::episode_over()` to match the full spec: terminate on
`team_X_score_ticks >= kWinTicks` OR `tick >= round_length * TICK_HZ`.
(The existing implementation is close; verify against `game_design.md` §3
win condition.)

### Step 4 — Reward computation

**Files:**
- `src/sim/include/xushi2/sim/sim.h` — add `StepRewards` struct:
  ```
  struct StepRewards {
      std::array<float, kAgentsPerMatch> per_agent;
      bool terminal;
  };
  ```
  Returned from a new `Sim::last_rewards()` accessor, or emitted from
  `step_decision` directly.
- `src/sim/src/sim.cpp` — pipeline step 16. Compute per-team shaped reward
  per `docs/rl_design.md` §5:
  - Terminal: ±10 on episode end (win/loss/draw)
  - +0.01 per own score point gained this decision
  - −0.01 per enemy score point gained
  - +0.25 per enemy kill credited to team
  - −0.25 per ally death
  - Clip cumulative non-terminal shaping to ±3.0 per team per episode
  - Broadcast team reward to all `present` agents on that team

Track cumulative shaping in `MatchState` so the clip is stateful across
decisions.

### Step 5 — Actor and critic observation builders (Tier 0)

**Files (new):**
- `src/common/include/xushi2/common/obs_utils.hpp` — pure utilities safe
  for actor side: `team_frame_pos`, `team_frame_vec`, `mirror_angle`,
  `normalize01`, enum one-hot.
- `src/sim/include/xushi2/sim/obs.h` — declares:
  ```
  constexpr std::uint32_t kPhase1ActorObsDim  = 28;  // verify against manifest
  constexpr std::uint32_t kPhase1CriticObsDim = N;   // compute from spec

  Result<void> build_actor_obs_phase1(
      const MatchState&, const MatchConfig&, std::uint32_t agent_index,
      std::span<float> out);

  Result<void> build_critic_obs_phase1(
      const MatchState&, const MatchConfig&, common::Team team,
      std::span<float> out);
  ```
- `src/sim/src/obs.cpp` — implementations. **Hard rule:**
  `build_actor_obs_phase1` must not take `MatchState` as a whole — it
  takes the specific hero slot it needs and a lightweight
  `PublicMatchView` derived from the objective state and own-team allies
  only. Enemy slot data is passed through a `MaybeVisibleEnemy` view that
  zeroes out hidden fields. This is the contract that
  `test_actor_leak.cpp` (Step 8) will enforce.

Field order matches `docs/observation_spec.md` §"Phase 1" exactly. Record
the field order in a new Python manifest `python/xushi2/obs_manifest.py`
(list of `(name, dim)` tuples).

### Step 6 — Python bindings extension

**Files:**
- `src/python_bindings/src/*.cpp` (wherever the current bindings live —
  verify path; may need creation).
- Expose `Sim::step_decision` returning rewards and actor obs for both
  agents, or: add new methods `observe_actor(agent_index) -> np.ndarray`
  and `observe_critic(team) -> np.ndarray`.
- Expose constants: `PHASE1_ACTOR_OBS_DIM`, `PHASE1_CRITIC_OBS_DIM`,
  `ACTION_REPEAT_DEFAULT`, all `k*` limits from `limits.hpp`.
- Obs arrays are views into preallocated `Sim`-owned buffers (zero-copy,
  per `coding_philosophy.md` §6). Python must never mutate them.

### Step 7 — Gymnasium-style env wrapper (Python)

**Files (new):**
- `python/xushi2/env.py` — single-env wrapper:
  ```
  class Xushi2Env:
      observation_space: gym.spaces.Box         # actor side
      critic_observation_space: gym.spaces.Box
      action_space: gym.spaces.Dict             # move_xy, aim_delta, buttons
      def reset(seed: int | None = None) -> (obs, info)
      def step(actions: dict[int, Action]) -> (obs, rewards, terminated, truncated, info)
      def critic_obs(team) -> np.ndarray
  ```
- `python/xushi2/vec_env.py` — stub for Phase 1 (not needed for env
  validation); leave as TODO for Phase 1 trainer.

`info` returns per-decision metrics: kills, deaths, objective progress,
shaping clip residual.

### Step 8 — Real scripted bot

**Files:**
- `src/bots/include/xushi2/bots/bot.h` (exists) — keep interface.
- `src/bots/src/bot.cpp` — replace the `NoopBot` in
  `make_basic_bot()` with a deterministic "walk-to-objective + shoot
  when LoS" bot:
  1. Compute 2D vector toward objective center (team-frame).
  2. If enemy within Revolver range and (no walls yet) ≈ LoS: aim toward
     enemy, press `primary_fire`.
  3. If magazine empty and enemy close: press Combat Roll.
  4. If not near objective: move toward it.
  Pure function of `MatchState`, no hidden RNG draws (or draw from a
  seeded per-bot RNG exposed in `MatchState`).

Keep `make_walk_to_objective_bot()` / `make_hold_and_shoot_bot()` as two
variants for the eval suite (`docs/rl_design.md` §11).

### Step 9 — Tests

**New / updated tests:**
- `tests/sim/test_combat.cpp`
  - Two Rangers, fire one shot, assert damage + magazine decrement.
  - Empty magazine → fire is no-op.
  - Combat Roll impulse → refills mag to 6, sets cooldown.
  - Auto-reload completes after the specified ticks.
- `tests/sim/test_respawn.cpp`
  - Reduce HP to 0 via direct damage, assert `alive = false`,
    `respawn_tick` set, then step past it and assert respawn.
- `tests/sim/test_objective.cpp`
  - All 5 cases from `game_design.md` §3 (contested, empty, single-team,
    capture completion, decay). Capture latency at exact-tick is asserted.
  - WIN_TICKS triggers `episode_over()`.
- `tests/sim/test_determinism.cpp` (update)
  - Extend the Phase-0 fixture to the full Phase-1 kit; two runs with
    same seed + same action trace = identical hashes.
- `tests/observations/test_actor_leak.cpp` (new)
  - Manipulate hidden enemy fields (position, cooldown, ammo, reload
    state) and assert actor obs bytes are unchanged. Must fail if
    `build_actor_obs_phase1` starts reading hidden state.
  - Assert critic obs *does* change when hidden state changes
    (sanity check the centralized side works).
  - Static check: grep / linter rule that `build_actor_obs_phase1.cpp`
    does not include or call any function iterating over hidden heroes.
- `tests/integration/test_random_rollout.cpp` (or
  `python/tests/test_random_rollout.py`)
  - 10 episodes of random Actions (sampled from the Python env's
    `action_space`), assert:
    - Every episode terminates within `round_length` ticks.
    - At least one kill, at least one objective tick, in the aggregate.
    - No NaN in observations or rewards.
    - Total reward per team is finite and bounded by the terminal ±10
      plus the shaping clip.

### Step 10 — Phase 0 determinism runner (closing the existing gate)

**Files:**
- `python/train/train.py` — replace the "not implemented" stub with a
  real driver for `phase: 0` configs:
  1. Load YAML.
  2. Construct `Xushi2Env` from `sim.*` section.
  3. Instantiate the scripted `basic` bot on both teams via a new
     `python/xushi2/bots.py` wrapper around the C++ bot.
  4. Run `episodes` rounds twice with the same seed.
  5. Assert the `state_hash` trajectories match tick-for-tick.
  6. Print a short summary (`episodes`, `decisions`, `final_hash`).

This closes the Phase 0 gate end-to-end from Python, which was always
promised by `experiments/configs/phase0_determinism.yaml` but never
wired up.

## Critical files to change

| Path | Change |
|---|---|
| `src/common/include/xushi2/common/limits.hpp` | add Phase 1 constants |
| `src/sim/include/xushi2/sim/sim.h` | reload / respawn / rewards state |
| `src/sim/src/sim.cpp` | pipeline steps 7, 10–16 |
| `src/sim/include/xushi2/sim/obs.h` (new) | obs builder declarations |
| `src/sim/src/obs.cpp` (new) | actor + critic builders |
| `src/common/include/xushi2/common/obs_utils.hpp` (new) | leak-safe utilities |
| `src/bots/src/bot.cpp` | real `basic` bot |
| `src/python_bindings/...` | expose obs builders, constants, rewards |
| `python/xushi2/__init__.py` | re-export new symbols |
| `python/xushi2/env.py` (new) | Gymnasium wrapper |
| `python/xushi2/obs_manifest.py` (new) | field-order manifest |
| `python/xushi2/bots.py` (new) | Python wrapper around C++ `basic` bot |
| `python/train/train.py` | Phase 0 determinism driver |
| `tests/sim/test_combat.cpp` (new) | Ranger combat tests |
| `tests/sim/test_respawn.cpp` (new) | Death/respawn tests |
| `tests/sim/test_objective.cpp` (new) | All 5 objective cases |
| `tests/sim/test_determinism.cpp` | extend to full Phase 1 kit |
| `tests/observations/test_actor_leak.cpp` (new) | leak prevention |
| `tests/integration/test_random_rollout.cpp` (new) | end-to-end smoke |
| `docs/determinism_rules.md` | update `state_hash()` manifest if hero state grows |

## Existing functions / utilities to reuse

- `common::canonicalize_action` (`src/common/include/xushi2/common/action_canon.hpp`)
  — call once at the sim boundary, as the current `step` already does.
- `common::clampf`, `common::wrap_angle`, `common::normalize_move_input`,
  `common::scale`, `common::add` (`src/common/include/xushi2/common/math.hpp`)
  — the existing tick pipeline already uses these for movement/aim.
- `common::quantize_pos`, `common::quantize_hp`
  (`src/common/include/xushi2/common/action_canon.hpp`) — used in
  `state_hash()` today; reuse for any new hashed fields.
- `X2_REQUIRE`, `X2_INVARIANT`, `X2_UNREACHABLE`
  (`src/common/include/xushi2/common/assert.hpp`) — Tier 0 assertion
  primitives.
- `common::kTickHz`, `common::kTeamSize`, `common::kDefaultActionRepeat`,
  `common::kAimDeltaMax` (`limits.hpp`) — extend, don't duplicate.
- `bots::IBot`, `bots::make_basic_bot` (`src/bots/include/xushi2/bots/bot.h`)
  — keep the interface, swap the no-op implementation.

## Open design questions

1. **Reward emission API.** Three candidates: `Sim::last_rewards()`
   accessor, `step_decision` returning a `StepResult`, or an output
   buffer written in place. The third matches Tier 0's zero-copy rule
   best. Confirm before Step 4.
2. **Combat Roll direction when both movement input and aim are zero.**
   Spec says "movement direction, else aim." If both are zero, the
   ability should either no-op (feels bad) or dash along current aim
   (default). Proposal: use current aim_angle; document in
   `action_spec.md`. Needs confirmation.
3. **Auto-reload interruption on Combat Roll during active reload.**
   Already spec'd in `game_design.md` §6 ("Combat Roll cancels an
   in-progress auto-reload and instantly fills to 6") — implement as
   stated.
4. **Shaping clip state location.** Cumulative-shaping counter lives in
   `MatchState` (so it's part of state_hash) vs. in the reward
   accumulator (not hashed). Proposal: in `MatchState`, to keep hashes
   self-contained and determinism easy to verify.
5. **Scripted bot RNG source.** If the bot ever wants a random tiebreak,
   it should draw from a per-bot `std::mt19937_64` seeded from
   `MatchConfig::seed + team_tag`, not from `state.rng` (keeps bot logic
   independent of sim PRNG consumption). Proposal: store
   `bot_rng_a`, `bot_rng_b` in `MatchState`.

## Verification plan

End-to-end smoke once all steps land:

```bash
# 1. Build & run unit tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# 2. Python install
cd python && pip install -e . && cd ..

# 3. Random-action rollout (validates the full loop without PPO)
python -m pytest python/tests/test_random_rollout.py -v

# 4. Phase 0 end-to-end determinism (closes the existing gate)
python -m train.train --config experiments/configs/phase0_determinism.yaml
# expected: "[xushi2] phase0 determinism: OK (N episodes, M decisions, final_hash=0x...)"
```

Observable success criteria:
- `ctest` green across combat, respawn, objective, determinism, and
  actor-leak tests.
- Random-action rollout produces at least one kill and one objective
  score tick across 10 episodes.
- Phase 0 runner prints identical final hashes on two runs with the
  same seed.
- Actor obs dim matches `PHASE1_ACTOR_OBS_DIM`; critic obs dim matches
  `PHASE1_CRITIC_OBS_DIM`.

## Out of scope (deferred)

- PPO trainer (next unit of work).
- Fog of war / LoS raycasting (Phase 7).
- Vanguard and Mender kits (Phase 6).
- Entity-token observations (Phase 5).
- Egocentric grid observations (Phase 6).
- Map randomization (Phase 8).
- Snapshot opponent pool (Phase 9).
- Replay file writer/reader (spec'd in `docs/replay_format.md`; lands
  when the first trainer wants checkpointed rollouts).
