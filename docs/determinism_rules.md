# Xushi2 — Determinism Rules

Canonical rulebook for the C++ simulation's determinism guarantees.
Cross-reference for `rl_design.md` §9, §10 and `game_design.md` §10.

## What we guarantee (MVP)

**Same-machine, same-binary, same-compiler reproducibility.** Given the same
`MatchConfig::seed` and the same per-tick action stream, the sim produces
bit-identical state trajectories, verifiable via `Sim::state_hash()`.

We do **not** yet guarantee cross-machine or cross-compiler reproducibility.
Fixed-point math is deferred. If and when cross-machine determinism becomes
a requirement (distributed training, external replay sharing), revisit.

## Compiler / build flags

The top-level `CMakeLists.txt` enforces:

- **MSVC:** `/fp:precise /fp:except-`
- **GCC / Clang:** `-fno-fast-math`, `-fno-associative-math`,
  `-fno-reciprocal-math`, `-fsignaling-nans`, `-fno-finite-math-only`

**Never** enable `-ffast-math` (or MSVC `/fp:fast`) in any TU that links
into the sim. This is a project-level invariant.

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
4. **Stable entity IDs.** Assigned once at spawn, never reused within a
   match. Deterministic tie-breaks on simultaneous events go by entity ID.
5. **No reliance on allocator behavior or pointer ordering.** Don't sort
   by address, don't use pointer identity as a key.
6. **Quantize at tick-end.** Position rounded to 1/1000 unit, HP to 1/100.
   Cooldowns stored as integer ticks (never floats).
7. **Avoid intrinsics whose implementation may vary by compiler or
   platform.** Prefer `std::sin` etc. over SSE/NEON intrinsics in game
   logic. If a specific math impl is required, ship exactly one.

## Tick pipeline is never silently reordered

The tick pipeline in `game_design.md` §11 is the contract. Changing step
order, or introducing parallelism within a step, is a semantic change and
must regenerate the golden replay (see below).

## Golden-replay tests

Lives at `tests/replay/test_golden_replay.cpp`. Run under CI on every
build. Fails when the sim behavior changes.

**What the test does:**

1. Loads a checkpointed "golden" replay from `data/replays/` — a binary
   file containing the seed, scripted-bot configuration, and a list of
   `(tick, state_hash)` pairs.
2. Re-runs the same seed and scripted actions in a fresh `Sim`.
3. Asserts every emitted `state_hash` matches the golden at every tick.
4. Asserts the final tick count matches.

**Regenerating the golden replay:**

Only after an intentional, reviewed sim change. Re-running the generator
silently on every build would defeat the test.

    # deliberate, human-initiated:
    build/bin/xushi2_golden_recorder \
        --seed 0xD1CEDA7A \
        --bot basic \
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
- `test_determinism` passes but `test_golden_replay` fails → sim semantics
  changed; regenerate the golden *intentionally*.
- `test_determinism` fails intra-process → non-determinism introduced
  somewhere. Most likely causes, in decreasing order:
  1. `std::unordered_*` iteration affecting state
  2. Accidentally introduced wall-clock time in the sim
  3. A compile flag drifted (check `-ffast-math`)
  4. A new dependency that pulls in non-deterministic code (tree search,
     parallel algorithms, etc.)

First diagnostic: binary-search the commit that introduced the regression.
The golden replay makes this cheap.
