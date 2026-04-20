# Xushi2 — Replay Format

On-disk format for deterministic match replays. Used by:

- `tests/replay/test_golden_replay.cpp` — pinned golden replays for CI
- Evaluation harness — record every eval game for post-hoc inspection
- Viewer — load-and-play recorded matches for debugging

**Status: Phase 0 — format specified, not yet implemented.**

## Design goals

1. **Fully deterministic reconstruction.** Given the replay and the same
   binary, the sim reproduces every tick bit-identically.
2. **Compact.** Store inputs, not full state trajectories. Replays should
   be < 1 MB for a full 240-second match.
3. **Versioned.** Format is guaranteed to round-trip within the same
   `schema_version`. Breaking schema changes bump the version.
4. **Language-agnostic.** Both the C++ sim and Python trainer can read
   and write the format.

## File structure

Little-endian binary, no compression at MVP (revisit if files become
large).

```
struct ReplayFile {
    char     magic[8];              // "XUSHI2RP"
    uint32_t schema_version;        // start at 1
    uint32_t sim_version;           // Sim::build_version_hash — breaks on sim changes
    uint64_t seed;                  // MatchConfig::seed
    uint32_t round_length_seconds;
    uint8_t  fog_of_war_enabled;
    uint8_t  randomize_map;
    uint8_t  padding[6];            // align next field to 8 bytes

    // Bot / policy identifiers for each of the 6 agent slots.
    // Each is a 16-byte null-padded ASCII tag — e.g., "basic_bot",
    // "snapshot_00042", "human".
    char     agent_tag[6][16];

    uint32_t num_ticks;             // number of tick records that follow
    TickRecord ticks[num_ticks];

    uint32_t num_state_hashes;      // sparse state hashes for CI verification
    HashSample hash_samples[num_state_hashes];

    char     footer[8];             // "XUSHI2RP"
};

struct TickRecord {
    // 6 agents × 7 action fields (packed).
    //   move_x, move_y, aim_delta: int16 quantized (scale: 1/10000)
    //   primary_fire, ability_1, ability_2: bitfield in one byte
    //   target_slot: uint8
    // 6 × (3 × 2 + 1 + 1) = 6 × 8 = 48 bytes per tick
    uint8_t actions[48];
};

struct HashSample {
    uint32_t tick;
    uint64_t state_hash;
};
```

## Versioning

- `schema_version` changes when the replay file layout changes.
- `sim_version` changes when the sim's semantics change (regenerate
  golden replays in a human-reviewed step — see `determinism_rules.md`).

Loaders must:
- Accept any `schema_version ≤ current_schema_version`; migrate in
  memory if needed.
- Reject a replay whose `sim_version` differs from the current sim,
  unless explicitly asked to replay against an old sim build.

## Hash sampling

`num_state_hashes` controls how densely we checkpoint hashes. For MVP,
hash every `TICK_HZ` ticks (one per simulated second) plus the terminal
tick. Roughly 181 samples per 180s match = 2.4 KB; negligible.

Every hash sample is a verification point: when replaying, the loader
asserts the live `Sim::state_hash()` equals the stored value. First
divergence fails the test fast.

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

## What is NOT stored

- Full per-tick state snapshots. The sim is deterministic; inputs plus
  seed reconstruct state. Snapshots would bloat the file 100×.
- RL policy weights. Replays reference a snapshot by tag, not by value.
- Reward values. Derived from state; recomputed on load.
- Observations. Derived from state; recomputed on load.
