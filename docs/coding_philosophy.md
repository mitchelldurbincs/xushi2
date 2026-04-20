# Xushi2 — Coding Philosophy

> **Xushi2 code should be boring, bounded, explicit, deterministic, and
> easy to falsify. Types define structure; assertions defend meaning.**
>
> **The simulation is critical code. The viewer, tools, and trainer are
> clients.**

Companion to `determinism_rules.md`, `action_spec.md`,
`observation_spec.md`, `replay_format.md`, and `rl_design.md`. These
rules are the spirit of Holzmann's *Power of Ten* adapted to a
deterministic game-sim + RL environment: the sim is a scientific
instrument, and its core value depends on reproducibility, replay
correctness, and the absence of hidden enemy information in actor
observations.

Apply the rules in **tiers**: hardest to the sim core, softer to the
viewer, softer still to the Python trainer. Do not make raylib UI or
PyTorch training miserable in the name of symmetry.

## 1. Purpose

Xushi2 is a deterministic 3v3 control-point sim used as a multi-agent
RL environment. Three things must be true, at all times:

1. The same seed plus the same canonical action stream produces
   bit-identical state trajectories (same machine, same binary).
2. Actor observations never contain hidden enemy state.
3. Replay files reconstruct matches exactly; golden replays in CI
   catch any drift.

Everything in this document exists to make those three statements
boring and obviously true.

## 2. Codebase tiers

Not all code has the same job. Rules below are tagged by tier; where no
tier is tagged, the rule applies to **all** tiers.

### Tier 0 — critical deterministic code

Strict rules. This is where the philosophy matters most.

```
src/sim/
src/common/            # parts used by sim
src/replay/            # deterministic input replay parser/writer
src/python_bindings/   # sim boundary + array validation
tests/sim/
tests/replay/
tests/observations/
```

Tier 0 obeys the full rules below: no post-init dynamic allocation,
fixed loop bounds, assertion density target, warnings-as-errors, static
analysis clean, zero copies in the data path, no unordered iteration
affecting state, deterministic failure behavior, `Result<T>` returns
instead of exceptions.

### Tier 1 — debug / viewer / tools

Softer rules.

```
src/viewer/            # raylib viewer
src/tools/             # offline tools (golden recorder, replay dump, ...)
src/bots/              # scripted bots
```

The viewer may allocate UI strings, use dynamic containers, and run a
main loop until the window closes. But:

- **Viewer code never mutates authoritative sim state** except by
  submitting `Action` structs through the public sim API. Rendering
  visualizes; it never affects simulation (`game_design.md` Pillar 2).
- **No viewer logic may be required for sim correctness.** Deleting
  the viewer must not break the sim, golden replays, or CI.
- **No dynamic allocation in the per-frame hot path** unless measured
  and accepted. Preallocate render buffers during window init.

### Tier 2 — Python training / eval

Practical ML rules.

```
python/train/
python/eval/
python/xushi2/
```

Python cannot follow "no dynamic allocation after init" literally, and
that is fine. What Tier 2 enforces:

- Shape, dtype, and device assertions on tensors crossing function
  boundaries
- NaN / Inf checks on losses, advantages, observations, and actions
- Seed logging and checkpoint reproducibility
- Replay export for every eval game
- Strict actor / critic separation — no hidden-enemy leakage via the
  Python obs builder either (see `observation_spec.md` and
  `rl_design.md` §10)

## 3. Assertions and defensive checks

Assertions check logic and state that the type system cannot prove.
Raw `assert()` is **not** the primary mechanism in Tier 0: it aborts,
it may be compiled out under `NDEBUG`, and it does not integrate with
the explicit-recovery rule for boundary functions.

### Assertion primitives (Tier 0)

Define four project macros / functions:

```cpp
X2_REQUIRE(cond, error_code)    // precondition / argument check
X2_ENSURE(cond, error_code)     // postcondition / return-value check
X2_INVARIANT(cond, error_code)  // state that must always hold
X2_UNREACHABLE(error_code)      // impossible control path
```

For expensive checks, two cost levels:

```cpp
X2_CHECK_FAST(...)   // always on, including release training builds
X2_CHECK_DEEP(...)   // on in tests, fuzzing, debug viewer, golden recorder
```

All four primitives are **side-effect free** and **active in all sim
builds** (not compiled out in release). Boundary functions translate
a failed check into a returned `ErrorCode`; internal functions may
abort, because by construction their preconditions were already
validated upstream.

Example:

```cpp
Result<void> apply_action(GameState& s, HeroId id, const Action& a) {
    X2_REQUIRE(is_valid_hero_id(id), Error::InvalidHeroId);
    X2_REQUIRE(a.move_x >= -1.0f && a.move_x <= 1.0f, Error::BadAction);
    X2_REQUIRE(a.move_y >= -1.0f && a.move_y <= 1.0f, Error::BadAction);

    Hero& h = s.heroes[id.index];

    X2_INVARIANT(h.hp_centihp >= 0, Error::CorruptHeroHp);
    X2_INVARIANT(h.cooldowns.guard_step_ticks >= 0, Error::CorruptCooldown);

    // logic...

    X2_ENSURE(std::isfinite(h.pos.x), Error::NonFinitePosition);
    X2_ENSURE(std::isfinite(h.pos.y), Error::NonFinitePosition);
    return Ok();
}
```

### Assertion density (Tier 0)

Target: **average ≥ 2 meaningful checks per non-trivial function.**
This is a codebase-level smell metric, not a mindless per-function
quota. Tiny accessors may have zero; observation builders, replay
parsers, action canonicalization, hitscan resolvers, visibility
computation, objective state-machine transitions, and the Python
boundary should have more.

A **meaningful** assertion checks one of:

- Precondition / postcondition
- State invariant or impossible branch
- Range, index, or enum validity
- Finite numeric value
- State-machine transition
- Format correctness at a boundary

A **non-meaningful** assertion (do not write these):

- `assert(true)`
- `assert(x == x)` (use an explicit NaN check instead)
- Conditions the compiler or type system already proves
- Conditions established on the previous line

This matches the *Power of Ten* spirit: assertions check anomalous
conditions, not statically provable ones.

### Make invalid states hard to represent; assert the rest

Assertions defend state machines — they do **not** replace them. If
you find yourself writing many asserts to keep a bag of loose fields
consistent, turn the bag into a struct with a narrow update function
(`ObjectiveState`, `RangerReloadState`, `MenderBeamState`,
`VanguardBarrierState`, `RespawnState`). Then the asserts defend
transitions, not representability.

### What to assert in Xushi2

**Action canonicalization** (see `action_spec.md`):

```
move_x, move_y finite
move_x, move_y in [-1, 1] after canonicalization
aim_delta finite
aim_delta in [-π/4, π/4] after canonicalization
target_slot == 0 while disabled (Phase 1–9)
action_repeat ∈ {2, 3}
decision tick alignment valid
```

**Sim state**:

```
hero IDs / team IDs valid
HP in [0, max_hp]
cooldowns >= 0 (integer ticks)
ammo in [0, 6]
positions finite
positions inside legal world bounds (or spawn / dead state)
objective owner / cap_team enums valid
team_score_ticks never decreases
last-seen memory age in range
beam source/target valid if beam active
barrier HP in [0, max_barrier_hp]
```

**Observation builders** (see `observation_spec.md`):

```
actor obs contains no hidden enemy position
visible enemy slots have visible=true
hidden / dead slots are zeroed and their presence flag is 0
presence flags match slot contents
team-relative coordinates finite
critic obs includes full state
actor and critic manifests have expected dimensions
```

**Replay parsing** (see `replay_format.md`):

```
magic bytes correct
schema_version known
sim_version compatible
num_decisions <= kMaxReplayDecisionRecords
decision start_ticks monotonic and within match duration
actions canonicalized
hash sample ticks monotonic
footer correct
```

## 4. Bounded control flow

**Every Tier 0 loop must have a visible, fixed maximum bound.**

Allowed:

```cpp
for (uint32_t i = 0; i < kMaxHeroes; ++i) { ... }

X2_REQUIRE(map.num_walls <= kMaxWalls, Error::CapacityExceeded);
for (uint32_t i = 0; i < map.num_walls; ++i) { ... }

X2_REQUIRE(events.size <= kMaxEventsPerTick, Error::CapacityExceeded);
for (uint32_t i = 0; i < events.size; ++i) { process(events[i]); }
```

Disallowed in Tier 0:

- Unbounded `while` loops whose termination depends on unbounded input
- Recursion
- Linked-list traversal without an explicit max iteration count
- Queue-drain loops without an explicit max
- Range-for over a dynamically-resizing container

Example rewrites:

```cpp
// BAD
while (node != nullptr) { node = node->next; }

// GOOD
for (uint32_t i = 0; i < kMaxNodes; ++i) {
    if (node == nullptr) break;
    node = node->next;
}
X2_REQUIRE(node == nullptr, Error::CapacityExceeded);
```

```cpp
// BAD
while (!events.empty()) { process(events.pop()); }

// GOOD
X2_REQUIRE(events.size <= kMaxEventsPerTick, Error::CapacityExceeded);
for (uint32_t i = 0; i < events.size; ++i) { process(events[i]); }
```

### Exceptions

- **Tier 1 application loops** (`while (!WindowShouldClose())`,
  `while (run)` in a tool) are fine. They are not sim logic.
- **Tier 2 training drivers** (`while global_steps < target_steps`)
  are fine. The sim step function called inside the loop is still
  finite and bounded.

Tier 0 step functions must themselves be finite:

```cpp
sim.step(actions);  // exactly one tick or one decision window
```

## 5. Memory allocation

Tier 0 performs **no dynamic memory allocation after initialization**.

### Initialization phase

```
- load config
- allocate BatchedSim
- allocate fixed-capacity state arrays
- allocate observation / reward / done buffers
- allocate replay buffers if used
- allocate viewer resources, if applicable
```

### Run phase (Tier 0)

```
- no new / delete / malloc / free
- no container resize / push_back past capacity
- no std::string construction in the sim hot path
- no Python object allocation per sim step
- no exceptions thrown or caught
- no virtual-dispatch allocation patterns
```

### Fixed-capacity containers

Use `std::array<T, N>` where possible, or a project `FixedVector`:

```cpp
template <typename T, size_t N>
struct FixedVector {
    std::array<T, N> data;
    uint32_t size;
};
```

Simple, auditable, no hidden allocator.

### Project capacity constants

Defined once (e.g., `src/common/include/xushi2/common/limits.h`):

```cpp
constexpr uint32_t kMaxHeroes               = 6;
constexpr uint32_t kMaxWalls                = 128;
constexpr uint32_t kMaxBarriers             = 6;
constexpr uint32_t kMaxEventsPerTick        = 128;
constexpr uint32_t kMaxRaysPerTick          = 32;
constexpr uint32_t kMaxReplayDecisionRecords = 240 * 15;  // 240 s @ 15 Hz
```

Exceeding any of these at runtime is an error, not a resize.

### Python boundary

The sim does **not** allocate Python objects per step. Python receives
**views** into preallocated output buffers wherever possible (zero-copy
via `py::array_t` / the buffer protocol). Observation, reward, and
done arrays are owned by `BatchedSim` and handed to Python as read-only
views. See `rl_design.md` §9.

## 6. Data layout and zero-copy rules

Avoid copies **in the data plane**. Allow explicit, named copies **at
trust boundaries**.

### Data plane (zero-copy)

```
C++ sim step
batched env stepping
actor obs buffer generation
critic obs buffer generation
replay action playback
```

Rules:

- Canonicalize actions **once** at the boundary; the sim consumes the
  canonical form (see `action_spec.md`, `replay_format.md`).
- Write observations directly into preallocated contiguous buffers.
- Pass `std::span` / views, not owning copies.
- Do not serialize/deserialize inside the sim hot path.
- Do not repeatedly transform data between equivalent layouts.

### Control / debug plane (copies allowed)

```
viewer panels
debug traces
offline replay conversion
metrics summaries
```

A copy here that makes debugging simpler is worth it.

### Prefer simple arrays of fixed structs

For 6 heroes and a handful of walls, a `std::array<HeroState, 6>` is
better than a general ECS:

```cpp
struct HeroState {
    Vec2 pos;
    Vec2 vel;
    float aim_angle;
    int32_t hp_centihp;
    int32_t cooldown_ticks[kMaxAbilities];
    uint8_t team;
    uint8_t hero_type;
    uint8_t alive;
    uint8_t padding;
};

struct GameState {
    std::array<HeroState, kMaxHeroes> heroes;
    ObjectiveState objective;
    MapState map;
    RngState rng;
};
```

### Align hot structs intentionally, not reflexively

Use `alignas(64)` where false-sharing or cache-line placement actually
matters: batched env state blocks, observation output buffers,
thread-local worker states, frequently-written counters shared across
cores. Do **not** cargo-cult `alignas(64)` onto `Vec2`. Use
`static_assert(sizeof(...))` on any struct whose layout crosses a file
or Python boundary.

### AoS vs SoA

- Inside one `GameState`: simple AoS is fine and clearer.
- Across `BatchedSim` (128 parallel envs): contiguous per-field arrays
  where it demonstrably helps throughput.
- Observation buffers: dense row-major arrays.

## 7. Determinism

`determinism_rules.md` is canonical. This document does **not**
duplicate compiler flags, PRNG rules, or `state_hash()` manifest.

Tier 0 code must not use, anywhere:

- Wall-clock time
- Unordered iteration in state-affecting paths
- Pointer order as logic
- Global RNG or `std::random_device`
- Fast-math
- Hidden threading inside a sim tick
- `<random>` distribution types in sim logic

See `determinism_rules.md` for canonical flags, project PRNG helpers
(`rng_u64`, `rng_int_bounded`, `rng_float01_canonical`), the
`state_hash()` manifest, and the split between sim-golden and
bot-regression tests.

## 8. Error handling and recovery

Tier 0 public / boundary functions return `Result<T>`, not exceptions.

```cpp
enum class ErrorCode : uint16_t {
    Ok,
    InvalidAction,
    InvalidHeroId,
    CorruptState,
    CorruptHeroHp,
    CorruptCooldown,
    NonFiniteFloat,
    NonFinitePosition,
    BadAction,
    CapacityExceeded,
    ReplayBadMagic,
    ReplayVersionMismatch,
    ReplayHashMismatch,
    ObservationLeakDetected,
};

template <typename T>
struct Result {
    bool ok;
    T value;
    ErrorCode error;
};
```

Public sim / replay / observation functions:

```cpp
Result<StepResult> step(GameState& s, ActionBatch actions);
Result<Replay>     load_replay(std::span<const uint8_t> bytes);
Result<void>       build_actor_obs(/* ... */);
```

Internal pure functions that cannot fail (all invalid inputs are
impossible by construction and asserted upstream) may return `void`:

```cpp
void update_cooldowns(GameState& s);
```

Rules:

- **No exceptions in Tier 0**, including no standard-library calls
  that throw on allocation failure in the hot path.
- **Python bindings translate `ErrorCode` into Python exceptions** at
  the boundary — the sim does not throw across the FFI.
- **Viewer** (Tier 1) may display an error to the user, but must not
  continue stepping a sim that returned `CorruptState` or
  `ObservationLeakDetected`.
- **Never return partial undefined state** from a parser. A bad replay
  file returns an error, not a half-populated `Replay`.

## 9. Static analysis and compiler warnings

All Tier 0 code compiles warning-clean with warnings-as-errors.

### Compiler flags

GCC / Clang:

```
-Wall -Wextra -Wpedantic
-Wconversion -Wsign-conversion
-Wshadow
-Wdouble-promotion
-Wfloat-equal
-Wundef
-Wcast-align -Wcast-qual
-Wformat=2
-Werror
```

MSVC:

```
/W4 (or /Wall if tolerable)
/WX
/permissive-
```

Do not mandate `/Wall` if system headers or raylib integration create
unavoidable noise. Use the strictest level that is actually clean.

### Static analysis

CI runs on Tier 0:

- `clang-tidy`
- `cppcheck`
- (optional) `include-what-you-use`

Sanitizer builds during development / nightly:

- AddressSanitizer
- UndefinedBehaviorSanitizer
- ThreadSanitizer (when/if threading arrives)

### Suppressions

Suppressing a warning or analyzer finding requires a comment
explaining why the tool is wrong, with a link to an issue or test.
"Silencing until quiet" is not allowed.

## 10. C++ subset

Deliberately boring.

### Encouraged (all tiers)

```
std::array, std::span
enum class
constexpr, static_assert
plain structs, POD-style layout for boundary types
fixed-capacity containers
small pure functions
explicit widths: uint32_t, int32_t, etc.
RAII during initialization
```

### Restricted in Tier 0

```
new / delete after init
std::vector::push_back after init (unless capacity is fixed and checked)
std::string in the hot path
std::unordered_map / std::unordered_set in state-affecting paths
exceptions
RTTI / dynamic_cast
shared_ptr / weak_ptr
virtual dispatch in the sim hot path
recursion
goto / setjmp / longjmp
global mutable state
threading inside a single sim tick
```

Tier 1 / Tier 2 relax these as needed. Tier 2 is Python/PyTorch and
follows its own idioms.

## 11. Testing and fuzzing

Minimum required test surface before a feature is "done":

- Unit tests for every Tier 0 state machine (objective, reload, beam,
  barrier, respawn).
- Determinism tests: intra-process (same seed twice in one process)
  and cross-process (future).
- Sim golden replay test (dense hash mode).
- Bot regression test (separate from the sim golden).
- Observation leak tests: actor obs does not change when any hidden
  enemy moves / aims / reloads / swaps weapon / changes cooldowns /
  fires.
- Replay round-trip fuzzing: random byte perturbations to replay
  files must fail cleanly, never crash or parse into inconsistent
  state.
- Action canonicalization round-trip test: live action stream =
  stored action stream, bit-for-bit.

## 12. Performance philosophy

Throughput at the MVP scale (~10–50k sim steps/sec at 128 parallel
envs) is not the bottleneck — PPO updates are. Do not premature-
optimize. But also do not write code that trivially ruins throughput:

- No per-step heap allocation
- No per-agent Python object creation
- No copying full state just to inspect it
- No repeated serialization / deserialization in the data plane

Measure before optimizing further. Keep the sim simple and correct
first; the scientific value comes from determinism and leak safety,
not from micro-optimizations.

## 13. Review checklist

For each Tier 0 change, ask:

- [ ] Are all loops bounded by a visible constant or checked runtime
      max?
- [ ] Are all allocations initialization-only?
- [ ] Are preconditions, postconditions, and invariants asserted with
      `X2_REQUIRE` / `X2_ENSURE` / `X2_INVARIANT`?
- [ ] Are action / observation / replay formats canonicalized exactly
      once at their boundary?
- [ ] Could hidden enemy state leak into actor observations? (Run
      `tests/observations/test_actor_leak`.)
- [ ] Could iteration order change sim behavior?
- [ ] Does `state_hash()` include any new state? (Update the manifest
      in `determinism_rules.md`.)
- [ ] Does the golden replay need deliberate regeneration?
- [ ] Does the code compile warnings-clean at the project strictness
      level?
- [ ] Does `clang-tidy` / `cppcheck` pass?
- [ ] For new public functions: do they return `Result<T>` and never
      throw?
- [ ] For new state: is it expressed as a typed struct with a narrow
      update function, rather than a bag of loose fields?

## 14. Cross-document references

This document is the umbrella philosophy. The canonical sources of
truth live in their respective specs:

- **Compiler flags, PRNG discipline, `state_hash()` manifest, golden
  replay policy**: `determinism_rules.md`.
- **Action schema, held vs impulse semantics, canonicalization**:
  `action_spec.md`.
- **Actor/critic observation layout, leak rules, phase-by-phase
  fields**: `observation_spec.md`.
- **Replay byte layout, `DecisionRecord`, hash modes, debug trace
  blocks**: `replay_format.md`.
- **RL algorithm choice, curriculum, reward design, evaluation
  protocol**: `rl_design.md`.
- **Game rules, tick pipeline, hero kits, objective state machine**:
  `game_design.md`.

When in doubt: the spec is canonical; this document is the attitude.
