# C++ Determinism Checklist (Tier 0/1)

Operational checklist extracted from `docs/coding_philosophy.md` for deterministic C++ code review.

## Scope

- **Tier 0 (strict):** `src/sim/`, deterministic parts of `src/common/`, `src/replay/`, `src/python_bindings/`, and sim/replay/observation tests.
- **Tier 1 (moderate):** `src/viewer/`, `src/tools/`, `src/bots/`.

## Tier 0 — Must

### Determinism and replay integrity

- **Must** preserve bit-identical trajectories for same seed + same canonical action stream on same machine/binary.
- **Must** avoid unordered iteration or any order-dependent behavior that can change authoritative state.
- **Must** keep replay parser/writer changes backward-safe or versioned so golden replay reconstruction remains exact.
- **Must** preserve deterministic failure behavior (explicit error path, not silent fallback randomness).

### Hidden-information boundaries

- **Must** ensure actor-visible outputs do not expose hidden enemy state.
- **Must** treat observation and replay boundaries as high-risk leakage points and assert them explicitly.

### Error handling and assertions

- **Must** use explicit boundary validation (`X2_REQUIRE`/`X2_ENSURE`-style checks and returned errors) instead of exceptions at sim boundaries.
- **Must** keep checks side-effect free and active in release sim builds.
- **Must** assert meaningful preconditions/postconditions/invariants in non-trivial functions (target average: 2+ meaningful checks/function at codebase level).
- **Must** assert finite numerics on movement/positions/derived outputs that can become NaN/Inf.
- **Must** validate enum/index/range/state-machine transitions for actions, objective state, cooldowns, ammo, beam/barrier links, etc.

### Structure and runtime behavior

- **Must** keep fixed or explicitly bounded loops in deterministic hot paths.
- **Must** avoid post-init dynamic allocation in deterministic hot paths.
- **Must** keep data-path copies minimal/intentional, especially across sim/replay/python boundaries.
- **Must** compile warning-clean with warnings-as-errors for Tier 0 targets.

## Tier 0 — Should

- **Should** model invalid states out of existence with tighter structs/state machines rather than scattered defensive checks.
- **Should** use deep checks (`X2_CHECK_DEEP`-style) in tests/fuzzing/golden tooling for expensive invariants.
- **Should** keep deterministic code “boring”: explicit control flow, explicit ownership, explicit error returns.

## Tier 1 — Must

- **Must** never mutate authoritative sim state except by submitting public `Action` structs through sim APIs.
- **Must** keep sim correctness independent of viewer/tool code (deleting viewer cannot break sim CI/golden replays).

## Tier 1 — Should

- **Should** preallocate per-frame render buffers and avoid per-frame hot-path allocation unless measured and accepted.
- **Should** follow Tier 0 style for determinism-relevant utilities reused by sim/runtime paths.

## Used in PR review

For PRs touching Tier 0/1 C++ paths:

- Reviewer **must** check each “Must” item and call out violations explicitly.
- Author **should** include a short “Determinism checklist notes” section in PR description with:
  - determinism/replay impact,
  - hidden-information impact,
  - assertion/validation changes,
  - test evidence (golden replay or equivalent).
- Any accepted exception **must** be documented with rationale and follow-up owner.
