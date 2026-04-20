# Xushi2 — Replay Format

On-disk format for deterministic match replays and debug traces. Used
by:

- `tests/replay/test_golden_replay.cpp` — pinned golden replays for CI
- Evaluation harness — record every eval game for post-hoc inspection
- Viewer — load-and-play recorded matches for debugging

**Status: Phase 0 — format specified, not yet implemented.**

## Design goals

1. **Fully deterministic reconstruction.** Given the replay and the same
   binary, the sim reproduces every tick bit-identically.
2. **Compact deterministic replays.** Store inputs, not full state
   trajectories. Deterministic input replays should be < 1 MB for a
   full 240-second match.
3. **Rich debug traces.** The viewer and trainer want reward-event
   flashes, observation inspection, policy-output inspection, and
   invalid-action reasons. These do not have to be small.
4. **Versioned.** Format is guaranteed to round-trip within the same
   `schema_version`. Breaking schema changes bump the version.
5. **Language-agnostic.** Both the C++ sim and Python trainer can read
   and write the format. The format is defined as a byte-level schema;
   C++ structs are implementation helpers, not the canonical spec.

## Two replay modes

### Mode 1 — Deterministic input replay

Small, committed alongside goldens, used for CI and regression tests.
Stores only what is needed to exactly reconstruct the sim.

Contents:
- Header (sim version, schema version, seed, full `MatchConfig` blob,
  config hashes)
- Canonical action stream (one record per policy decision; see
  `DecisionRecord` below)
- Sparse state hashes for verification (default: once per simulated
  second + terminal tick)

### Mode 2 — Debug / eval trace

Larger, generally not committed. Intended for learning, inspection,
reward-hack debugging, and the viewer's timeline UI.

Contents (all optional, controlled by flags in the header):
- Everything in Mode 1
- Reward events (per-component, per-team, with source/target entity
  IDs where applicable)
- Actor observation summaries (per agent, per decision — e.g.,
  normalized tensor dumps or hashes + key fields)
- Critic observation summaries (per decision)
- Policy outputs (logits, logprobs, sampled action)
- Invalid-action reasons (cooldown, ammo, LoS, target mask, etc.)
- Sparse full-state snapshots (opt-in; default interval 1 sim-second)

The deterministic input replay alone cannot answer questions like
"what did the actor obs tensor contain at this tick?" or "which reward
component fired?" — that is what Mode 2 is for.

## Byte-level schema

All integer fields are little-endian. Booleans are a single byte (0 or
1). Strings are fixed-width ASCII, null-padded. Variable-length arrays
are length-prefixed by a `uint32_t` count. There is **no implicit
padding** — any padding bytes are explicitly named and their count is
part of the schema. All record sizes are `static_assert`ed in C++, and
the Python parser has a round-trip test against a known fixture.

```
ReplayFile {
    char     magic[8];                    // "XUSHI2RP"
    uint32_t schema_version;              // start at 1
    uint32_t sim_version;                 // Sim::build_version_hash
    uint8_t  trace_flags;                 // bit 0: input replay (always 1)
                                          // bit 1: reward events
                                          // bit 2: actor obs summaries
                                          // bit 3: critic obs summaries
                                          // bit 4: policy outputs
                                          // bit 5: invalid-action reasons
                                          // bit 6: full-state snapshots
    uint8_t  hash_mode;                   // 0 = sparse_eval (default),
                                          // 1 = dense_golden,
                                          // 2 = custom_interval
    uint32_t hash_interval_ticks;         // only read when hash_mode == 2
    uint8_t  explicit_padding[2];         // reserved; must be 0

    // --- Match configuration (everything needed to reproduce the run) ---
    uint64_t seed;                        // MatchConfig::seed
    uint32_t tick_hz;                     // e.g. 30
    uint32_t action_repeat;               // sim ticks per decision
    uint32_t round_length_seconds;
    uint8_t  fog_of_war_enabled;
    uint8_t  randomize_map;
    uint8_t  explicit_padding2[6];

    uint64_t config_hash;                 // hash of the full MatchConfig
    uint32_t hero_config_version;
    uint32_t map_config_version;
    uint64_t map_seed;
    uint64_t reward_config_hash;

    // Bot / policy identifiers for each of the 6 agent slots.
    // Each is a 16-byte null-padded ASCII tag — e.g., "basic_bot",
    // "snapshot_00042", "human".
    char     agent_tag[6][16];

    // Full serialized MatchConfig for forensic reproducibility.
    // Negligible size compared to a few seconds of replay.
    uint32_t match_config_blob_bytes;
    uint8_t  match_config_blob[match_config_blob_bytes];

    // --- Canonical action stream ---
    uint32_t num_decisions;
    DecisionRecord decisions[num_decisions];

    // --- Verification hashes ---
    uint32_t num_state_hashes;
    HashSample hash_samples[num_state_hashes];

    // --- Optional trace payloads (each length-prefixed, each guarded by
    //     a bit in trace_flags). Consumers skip blocks whose flag is 0. ---
    uint32_t num_reward_events;      // if trace_flags bit 1
    RewardEvent reward_events[num_reward_events];

    uint32_t num_actor_obs_samples;  // if trace_flags bit 2
    ActorObsSample actor_obs_samples[num_actor_obs_samples];

    uint32_t num_critic_obs_samples; // if trace_flags bit 3
    CriticObsSample critic_obs_samples[num_critic_obs_samples];

    uint32_t num_policy_outputs;     // if trace_flags bit 4
    PolicyOutputSample policy_outputs[num_policy_outputs];

    uint32_t num_invalid_actions;    // if trace_flags bit 5
    InvalidActionEvent invalid_actions[num_invalid_actions];

    uint32_t num_state_snapshots;    // if trace_flags bit 6
    StateSnapshot state_snapshots[num_state_snapshots];

    char     footer[8];              // "XUSHI2RP"
}

DecisionRecord {
    uint32_t start_tick;             // sim tick at which the decision applies
    uint8_t  action_repeat;          // number of sim ticks this decision is held
    uint8_t  explicit_padding[3];
    // 6 agents × packed action, canonicalized exactly as the sim consumes them:
    //   move_x, move_y, aim_delta: int16 quantized (scale 1/10000)
    //   primary_fire, ability_1, ability_2: bitfield in one byte
    //   target_slot: uint8
    // 6 × (3×2 + 1 + 1) = 6 × 8 = 48 bytes
    uint8_t  actions[48];
}

HashSample {
    uint32_t tick;
    uint64_t state_hash;
}

RewardEvent {
    uint32_t tick;
    uint8_t  team;                   // 0 = A, 1 = B
    uint16_t component_id;           // enum: KILL, DEATH, OBJECTIVE, HEAL, SHIELD, TERMINAL, ...
    uint8_t  explicit_padding;
    float    amount;
    uint32_t source_entity_id;       // 0xFFFFFFFF if none
    uint32_t target_entity_id;       // 0xFFFFFFFF if none
}

InvalidActionEvent {
    uint32_t tick;
    uint8_t  agent_slot;
    uint8_t  field;                  // enum: PRIMARY, ABILITY_1, ABILITY_2, TARGET
    uint16_t reason;                 // enum: COOLDOWN, AMMO, LOS, TARGET_MASK, DEAD, ...
}

// ActorObsSample / CriticObsSample / PolicyOutputSample / StateSnapshot
// are length-prefixed opaque blobs at the byte level. Their internal
// layout is versioned by sim_version and described in the companion
// manifest files (python/xushi2/obs_manifest.py, etc.). Consumers that
// do not understand the layout can still skip them using the length
// prefix.
```

## Canonical action representation

The action stream consumed by the live sim is **exactly** the action
stream stored in replay. Continuous fields are quantized to `int16`
(scale `1/10000`) before entering the sim; booleans are packed;
`target_slot` is clamped. There is no second "float32 live path" —
training, human play, and replay all go through the same canonicalized
`Action`. This is the single most important rule in this document:

> **The action stream consumed by live sim must be exactly the action
> stream stored in replay.**

A tiny aim or movement difference can change a raycast hit/miss, which
can cascade into a totally different fight.

## Per-decision, not per-tick

Decisions are the unit of record — not sim ticks. A `DecisionRecord`
carries `start_tick` and `action_repeat` so the sim can reconstruct the
held action window deterministically. `move_x` / `move_y` apply every
tick of the window; `aim_delta` applies once at `start_tick`; held
buttons re-evaluate each tick; impulse buttons evaluate once at
`start_tick`. See `action_spec.md` for the held/impulse per-field rules.

## Versioning

- `schema_version` changes when the replay file layout changes.
- `sim_version` changes when the sim's semantics change (regenerate
  golden replays in a human-reviewed step — see `determinism_rules.md`).

Loaders must:
- Accept any `schema_version ≤ current_schema_version`; migrate in
  memory if needed.
- Reject a replay whose `sim_version` differs from the current sim,
  unless explicitly asked to replay against an old sim build.

## Hash modes

`hash_mode` controls state-hash density:

- `sparse_eval` (default for training/eval replays) — once per
  `tick_hz` ticks (one per simulated second) + terminal tick. Roughly
  181 samples per 180s match = 2.4 KB.
- `dense_golden` — every sim tick. Used by the golden-replay CI test so
  that the first divergent tick is caught exactly.
- `custom_interval` — every `hash_interval_ticks` ticks.

**Policy:** golden replay CI uses `dense_golden`. Training/eval
replays use `sparse_eval` by default.

Every hash sample is a verification point: when replaying, the loader
asserts the live `Sim::state_hash()` equals the stored value. First
divergence fails the test fast.

## Cross-platform / WASM note

Deterministic replay exactness is guaranteed only for the same binary,
same compiler, same machine. A WASM viewer may be visual-only: it can
load debug traces with snapshots and interpolate, but exact sim replay
across native ↔ WASM is not guaranteed unless and until we move to
fixed-point or cross-platform-exact math. See `determinism_rules.md`.

## Paths / conventions

```
data/replays/
  ├─ golden_phase0.replay       # committed to the repo; CI golden
  ├─ golden_phase1.replay       # (future) per-phase goldens
  └─ runs/
      ├─ 2026-04-20T12-30-00.replay
      └─ ...                    # .gitignored eval / training replays
```

Generator tools live in `src/tools/` (C++) and `python/scripts/` (Python
wrappers). Recording from inside a training run routes through the
Python side; golden-replay regeneration uses the C++ tool directly to
avoid any Python-layer interference.

## What is NOT stored in deterministic input replays

- Full per-tick state snapshots. The sim is deterministic; inputs plus
  seed reconstruct state. Snapshots would bloat the file 100×.
- RL policy weights. Replays reference a snapshot by tag, not by value.
- Reward values. Derived from state; recomputed on load.
- Observations. Derived from state; recomputed on load.

These derived fields **are** stored in debug/eval traces (Mode 2) when
the corresponding trace flag is set, because post-hoc recomputation
will reflect *current* reward/obs code, not the code that produced the
run — which is exactly what you don't want when debugging a reward
hack or an observation regression.
