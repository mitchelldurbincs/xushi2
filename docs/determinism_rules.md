# Xushi2 — Determinism Rules

**Coding philosophy:** Tier 0 sim code follows `coding_philosophy.md`.
This document is the canonical source for determinism-specific flags,
PRNG discipline, the `state_hash()` manifest, and golden-replay policy.

Canonical rulebook for the C++ simulation's determinism guarantees.
**This document is the single source of truth for compiler flags,
source-level rules, PRNG discipline, the state-hash manifest, and
golden-replay policy.** Other docs (`rl_design.md` §9, §10 and
`game_design.md` §10) must link here rather than duplicating.

## What we guarantee (MVP)

**Same-machine, same-binary, same-compiler reproducibility.** Given the
same `MatchConfig::seed` and the same per-decision canonical action
stream, the sim produces bit-identical state trajectories, verifiable
via `Sim::state_hash()`.

We do **not** yet guarantee cross-machine or cross-compiler
reproducibility. Fixed-point math is deferred. If and when cross-machine
determinism becomes a requirement (distributed training, external
replay sharing, native-↔-WASM replay exactness), revisit.

**WASM caveat.** Any raylib + emscripten viewer stretch is best treated
as visual-only. A replay recorded in a native training run is **not**
guaranteed to play bit-identically in a WASM build: floating-point
semantics across the two toolchains are not guaranteed to match. Load
debug traces with snapshots and interpolate visually; do not assume
exact sim replay across targets.

## Compiler / build flags

The top-level `CMakeLists.txt` enforces these. Other docs must not
duplicate this list — reference this section.

### Required MVP flags

- **MSVC:** `/fp:precise /fp:except-`
- **GCC / Clang:** `-fno-fast-math`, `-fno-associative-math`,
  `-fno-reciprocal-math`, `-fsignaling-nans`, `-fno-finite-math-only`

### Forbidden flags

- `-ffast-math` (GCC/Clang)
- `/fp:fast` (MSVC)

**Never** enable any of the forbidden flags in any TU that links into
the sim. This is a project-level invariant.

### Optional diagnostic flags

- `-ffloat-store` (GCC) — blunt tool, increases memory traffic, only
  enable if specifically diagnosing an x87 extended-precision issue.
  Not default.
- `-ftrapv` / UBSan — useful in debug builds to catch UB.

## Source-level rules

These are not enforced by the compiler. They are enforced by code review
and by the golden-replay CI test.

1. **No `std::unordered_map` / `std::unordered_set` iteration** in any path
   that affects game state. Iteration order is unspecified and may differ
   between runs. Use `std::map`, sorted `std::vector`, or iterate by
   explicit stable entity ID.
2. **No wall-clock time inside the sim.** No `std::chrono`, no `time()`,
   no `QueryPerformanceCounter`. The viewer is allowed to use wall-clock
   for rendering; the sim is not.
3. **One PRNG per match.** All randomness flows from `MatchState::rng`
   (seeded `std::mt19937_64`). No `std::random_device`, no global RNG.
4. **No `std::uniform_real_distribution`, `std::normal_distribution`,
   or any other `<random>` distribution inside deterministic sim
   logic.** The engine algorithm for `std::mt19937_64` is standardized
   across implementations, but distribution implementations are not.
   Use project-owned helpers only:
   - `rng_u64()`
   - `rng_int_bounded(rng, lo, hi)`
   - `rng_float01_canonical(rng)`

   RL policy sampling (Gaussians, categoricals) lives in Python /
   PyTorch and is outside the sim. The sim should only need randomness
   for map perturbation and scripted bots.
5. **Stable entity IDs.** Assigned once at spawn, never reused within a
   match. Deterministic tie-breaks on simultaneous events go by entity ID.
6. **No reliance on allocator behavior or pointer ordering.** Don't sort
   by address, don't use pointer identity as a key.
7. **Quantize at tick-end.** Position rounded to 1/1000 unit, HP to 1/100.
   Cooldowns stored as integer ticks (never floats). Objective score
   and capture progress stored as integer ticks (see `game_design.md`
   §3).
8. **Avoid intrinsics whose implementation may vary by compiler or
   platform.** Prefer `std::sin` etc. over SSE/NEON intrinsics in game
   logic. If a specific math impl is required, ship exactly one.

## Tick pipeline is never silently reordered

The tick pipeline in `game_design.md` §11 is the contract. Changing step
order, or introducing parallelism within a step, is a semantic change and
must regenerate the golden replay (see below).

## `state_hash()` manifest

`Sim::state_hash()` must include every piece of simulator state that can
affect future behavior. Omitting a field from the hash means golden
replays can pass while hidden state diverges — which defeats the entire
purpose of the test.

The hash must include:

- Tick index
- RNG state (the `std::mt19937_64` internal state, not just the seed)
- All hero positions, velocities, aim directions (quantized)
- HP, shield values, temporary buffs/debuffs
- Cooldowns, reload timers, ammo counts, weapon modes
- Any per-agent button state *that still lives in sim state* (none
  expected under impulse semantics — see `action_spec.md`)
- Active barrier state (owner, HP, orientation)
- Mender beam lock target (source/target entity IDs, beam active flag)
- Last-seen memory contents (per-agent ghosts + decay timers)
- Objective state: `owner`, `cap_team`, `cap_progress_ticks`,
  `team_score_ticks` per team
- Respawn timers
- Map layout geometry (hash of wall / pillar positions)

When you add a new field to simulator state, update this manifest **and**
the `state_hash()` implementation. A CI test exists (or will exist) that
cross-checks the manifest against the implementation.

## Golden-replay tests

Lives at `tests/replay/test_golden_replay.cpp`. Runs under CI on every
build. Fails when the sim behavior changes.

Two distinct tests live here, and they must stay separate.

### Sim golden replay

**Input:** a committed replay with a recorded canonical action stream
and `hash_mode = dense_golden` (every sim tick).

**What the test does:**

1. Loads the golden replay from `data/replays/`.
2. Feeds the recorded action stream directly into a fresh `Sim` (**does
   not** call bot policy code).
3. Asserts every emitted `Sim::state_hash()` equals the stored value,
   at every sim tick.
4. Asserts the final tick count matches.

This test catches any change to sim semantics. It is independent of bot
logic by design: a bot refactor must not force a golden regeneration.

### Bot regression test

**Input:** a scripted-bot configuration and a fixed seed.

**What the test does:**

1. Runs the bot policies from the seed/config.
2. Asserts the bot's behavior matches a pinned expectation (e.g., a
   smaller hash set, or a specific decision trace).

This test is allowed to change when bot logic changes. It must not use
the sim golden replay as its source of truth.

### Regenerating the sim golden replay

Only after an intentional, reviewed sim change. Re-running the generator
silently on every build would defeat the test.

    # deliberate, human-initiated:
    build/bin/xushi2_golden_recorder \
        --seed 0xD1CEDA7A \
        --bot basic \
        --hash-mode dense_golden \
        --out data/replays/golden_phase0.replay

Commit the new replay with the sim change in the same PR.

## Additional invariants

- **Intra-process determinism test** — `tests/sim/test_determinism.cpp`
  runs the same seed twice in one process and asserts hash equality.
- **Cross-process determinism test** (future) — spawn two fresh processes,
  run the same seed, compare. Not yet implemented; low priority for MVP.
- **Build-flag sanity test** (future) — CMake emits a small program that
  asserts certain compile flags (e.g., `-ffast-math` disabled). Not yet
  implemented; placeholder.

## When determinism breaks

Symptoms:
- `test_determinism` passes but sim golden replay fails → sim semantics
  changed; regenerate the golden *intentionally*.
- `test_determinism` fails intra-process → non-determinism introduced
  somewhere. Most likely causes, in decreasing order:
  1. `std::unordered_*` iteration affecting state
  2. Accidentally introduced wall-clock time in the sim
  3. A `<random>` distribution slipped into sim logic
  4. A compile flag drifted (check `-ffast-math`)
  5. A new dependency that pulls in non-deterministic code (tree search,
     parallel algorithms, etc.)

First diagnostic: binary-search the commit that introduced the regression.
The golden replay makes this cheap.
