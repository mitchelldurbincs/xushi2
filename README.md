# Xushi2 虚实

A deterministic, top-down 2D, 3v3 control-point team shooter designed as a
multi-agent reinforcement learning environment that is also playable by humans.

Named after the Sun Tzu concept of *虚实* (feint and substance): information
warfare and deception under partial observation.

## Status

**Phase 4 scaffold is trainable.** Phase 2 cleared; Phase 3 produced a
warm-start Ranger policy; Phase 4 now has 3v3 Ranger spawning, a 135-float
centralized critic observation, a 3v3 MAPPO-shaped Gymnasium env, a sync
vector wrapper for MAPPO rollouts, and a recurrent MAPPO smoke trainer wired
through `phase: 4`. The raylib viewer is a minimal replay/debug viewer — see
[Current state](#current-state) for a precise breakdown.

## Project layout

```
xushi2/
├── docs/                  design documents — authoritative spec
├── src/
│   ├── sim/               C++ deterministic simulation core        (working)
│   ├── viewer/            raylib viewer (human play + debug)       (scaffold)
│   ├── bots/              scripted bots for tests / eval baselines (stubs)
│   ├── python_bindings/   pybind11 module exposing sim to Python   (working)
│   ├── tools/             offline tools (replay inspector, etc.)   (empty)
│   └── common/            shared types / utilities                 (working)
├── tests/                 C++ tests (GoogleTest)                    (working)
├── python/                Python trainer, eval harness, helpers    (skeleton)
├── experiments/           configs, notes, checkpoints              (empty)
├── data/                  replays, eval results, map files         (empty)
├── assets/                fonts, shaders, viewer UI resources      (empty)
└── third_party/           vendored deps (most via FetchContent)    (empty)
```

See `docs/game_design.md` and `docs/rl_design.md` for the full project
specification. The README is a quick-start only.

## Documents

| File | Purpose |
|---|---|
| `docs/game_design.md` | Game rules, heroes, fog of war, combat, tick pipeline |
| `docs/rl_design.md` | MAPPO algorithm, obs/action spaces, curriculum, eval |
| `docs/coding_philosophy.md` | Maturity tiers, determinism discipline, code ownership |
| `docs/determinism_rules.md` | Float determinism discipline, golden replays |
| `docs/observation_spec.md` | Exact actor- and critic-side observation layouts |
| `docs/action_spec.md` | Exact action schema + held / edge-triggered rules |
| `docs/replay_format.md` | On-disk replay file format |

## Build (C++ side)

Requires CMake ≥ 3.24, a C++20 compiler, and Python 3.10+ for the Python
module.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `XUSHI2_BUILD_VIEWER` | ON | Build the raylib viewer |
| `XUSHI2_BUILD_PYTHON_MODULE` | ON | Build the pybind11 Python module |
| `XUSHI2_BUILD_TESTS` | ON | Build tests |
| `XUSHI2_WARNINGS_AS_ERRORS` | OFF | Fail the build on warnings |

## Python side

```bash
cd python
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .
```

The Python package `xushi2` re-exports the `xushi2_cpp` extension module, so
the C++ module must be built first (via the CMake step above, with
`XUSHI2_BUILD_PYTHON_MODULE=ON`).

## Current state

What works today:

- ✅ **Deterministic sim core** (`src/sim/src/sim.cpp`) — 30 Hz fixed
  tick, reset/step, objective capture state machine, hitscan Revolver
  combat, magazine/reload, Combat Roll, death/respawn, seeded
  `std::mt19937_64`, float-determinism flags.
- ✅ **Common types** (`src/common/include/xushi2/common/`) — `Team`,
  `Role`, `HeroKind` enums; `Vec2`, `Action`; fixed-capacity constants.
- ✅ **Observation builders** (`src/sim/src/actor_obs.cpp`,
  `critic_obs.cpp`, `obs_utils.cpp`) — Phase-1 flat actor obs plus the
  Phase-4 135-float centralized critic obs, zero-copy into caller-provided
  numpy buffers, structural actor-leak test green.
- ✅ **Scripted bots** (`src/bots/src/bot.cpp`) —
  `walk_to_objective`, `hold_and_shoot`, `basic`, `noop`.
- ✅ **pybind11 module** (`src/python_bindings/module.cpp`) — `Sim`,
  `MatchConfig`, `Action`, enums, `build_actor_obs` /
  `build_critic_obs`, `scripted_bot_action`.
- ✅ **Python env wrapper** (`python/xushi2/env.py`) — single-env
  Gymnasium interface for 1v1 Ranger vs a named scripted opponent.
- ✅ **Reward calculator** (`python/xushi2/reward.py`) — terminal-dominant
  ±10/0, shaped events symmetrized and per-episode clipped to ±3.
- ✅ **GoogleTest + pytest suites** — 87 C++ tests, 50+ Python tests;
  `ctest` and `pytest` clean.
- ✅ **xushi2-eval CLI** — determinism golden dump plus Phase-1b
  `--dump-obs` / `--dump-reward` trajectory dumps.
- ✅ **Recurrent PPO trainer package** (`python/train/ppo_recurrent/`,
  7 files: config, trainer, losses, evaluate, orchestration,
  lr_schedule, plus package init) — env-agnostic GRU-backed PPO with
  BPTT, value normalization, and cosine LR schedule.
- ✅ **Rollout buffer + actor/critic models**
  (`python/train/rollout_buffer.py`, `python/train/models.py`) — shared
  across feedforward and recurrent trainers.
- ✅ **Per-head gradient instrumentation** — `actor_grad_norm`,
  `critic_grad_norm`, `trunk_grad_norm`, `terminal_adv_std`,
  `mean_log_std` logged per update.
- ✅ **Phase-2 memory-toy env + eval** (`python/envs/memory_toy.py`,
  `python/eval/eval_memory_toy.py`) — recurrent-vs-feedforward gate
  cleared (gap 0.889, see `docs/plans/2026-04-23-phase2-result.md`).
- ✅ **Phase-3 ranger env + eval** (`python/envs/phase3_ranger.py`,
  `python/eval/eval_phase3.py`) — C++ sim wired into the recurrent PPO
  trainer for 1v1 Ranger.
- ✅ **Phase-4 MAPPO smoke path** (`python/envs/phase4_mappo.py`,
  `python/train/mappo.py`) — 3v3 Ranger env, centralized critic buffer,
  sync vector wrapper, phase registry entry, smoke config, checkpointing,
  and one-update pytest coverage.
- ✅ **Phase-5 entity-attention probe** (`python/envs/phase5_entity_mappo.py`,
  `python/train/entity_attention.py`) — Phase-4 actor obs wrapped into
  self/enemy/objective entity tokens, masked attention pooling in MAPPO, a
  compact objective-path config, checkpointing, replay dumping, and pytest
  coverage. This is still a fixed-map/full-vision diagnostic, not the scaled
  Phase-5 acceptance run.
- ✅ **Phase-6 entity-grid probe** (`python/envs/phase6_grid_mappo.py`,
  `python/xushi2/grid_obs.py`) — appends a small 32x32x3 egocentric grid to
  the Phase-5 token obs, fuses it with a CNN branch in MAPPO, and has a compact
  objective-path run plus replay dumping. This is a diagnostic adapter, not the
  full OA5-analog roster/acceptance gate.
- ✅ **Phase-7 team-shared fog probe** (`python/envs/phase7_fog_mappo.py`,
  `python/xushi2/partial_obs.py`) — masks enemy tokens/grid channel outside a
  configurable visibility radius and native cover/wall line-of-sight with
  team-shared or per-agent visibility, routes the Phase-1 native actor enemy
  fields through `MatchConfig::fog_of_war_enabled` LoS filtering, exposes a
  native per-viewer observable-enemy slot mask, carries diagnostic last-seen
  enemy markers with adapter-level hidden-state leak tests, runs compact
  objective-path diagnostics, and tags viewer replays with `fog=1` /
  `last_seen=1` / `fog_mode=...`. Widening actor obs to true multi-enemy
  tokens is still open.
- ✅ **Phase-8 randomized-map probe** (`python/envs/phase8_random_map_mappo.py`,
  `python/xushi2/map_randomization.py`) — applies deterministic per-episode
  arena-bounds randomization, installs symmetric circular cover pillars and
  wall segments into native movement/hitscan/beam/tether/LoS checks, carries
  explicit map bounds, cover/wall geometry, and stable layout hashes into env
  info/replay headers/viewer rendering, and has a compact objective-path
  diagnostic.
- ✅ **Phase-9 snapshot-opponent probe** (`python/envs/phase9_snapshot_mappo.py`,
  `python/xushi2/snapshot_policy.py`) — loads frozen MAPPO checkpoints as
  env-side opponents, samples weighted `latest` / `historical` / `anchor`
  snapshot groups and `current` / `snapshot` / `anchor` match types
  deterministically, writes exact six-slot replays with nonzero opponent
  actions, Phase-8 cover/wall topology, league metadata, and PPO `loss_mask`
  metadata, writes a capped `snapshot_league.json` retention manifest that
  preserves matrix-gated final snapshots when `run.matrix_eval` is enabled,
  adds per-agent value/GAE support for opposing-team returns, and has a compact
  objective-path diagnostic.
- ✅ **MAPPO matchup matrix diagnostic** (`python/scripts/eval_mappo_matrix.py`) —
  evaluates MAPPO checkpoints against anchor bots and frozen snapshot
  checkpoints, prints compact win/draw/reward/score rows, can write JSON for
  later league gates, and can run automatically after MAPPO training via
  `run.matrix_eval`. `python/scripts/check_mappo_matrix.py` gates matrix JSON
  against compact row-count / win-rate / draw-rate thresholds.
- ✅ **Phase-11 current/mixed self-play probe**
  (`python/envs/phase11_current_selfplay_mappo.py`) — controls all six agents
  with the current MAPPO policy in one 3v3 current match, returns six actor obs/action
  rows, emits separate Team-A and Team-B rewards, builds per-agent team critic
  observations for `ppo.value_per_agent: true`, carries Phase-8 map/fog
  topology, samples compact `current` / `snapshot` / `anchor` league matches
  with dynamic PPO loss masks for externally controlled Team-B slots, writes
  `run.matrix_eval` selfplay/bot/snapshot rows, and runs one-update
  current-vs-current and mixed-league smoke configs.
- ✅ **Phase-10 target-slot probe** (`python/envs/phase10_target_slot_mappo.py`) —
  enables the seventh `target_slot` categorical action factor in MAPPO, appends
  a valid-target mask to actor observations, accepts/clamps targets in an env
  wrapper, runs a mirrored Vanguard/Ranger/Mender composition through the sim,
  gives Mender Staff beam, Staff/Sidearm weapon swap, Sidearm hitscan, and Tether,
  gives Vanguard a held barrier that absorbs Ranger shots, a Guard Step dash,
  and a short-range Warhammer primary diagnostic, gives Ranger a targeted
  Mark Target diagnostic,
  tags viewer replays with `target_slot=1` and hero kinds, and has a compact
  objective-path diagnostic. Full second-hero ability kits are still open.
- ✅ **Minimal viewer** (`src/viewer/src/main.cpp`) — raylib arena/objective
  rendering, heroes with HP/facing, score/cap panel, shot tracers, replay
  playback for legacy 1v1 and text 3v3 action streams, pause/single-step/reset,
  playback speed controls, replay metadata panels, cover/wall rendering, and
  fog replay LoS debug rays with visible/total counts, plus Phase-10
  target-token replay annotations. Build with
  `-DXUSHI2_BUILD_VIEWER=ON
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5` while the project is pinned to raylib
  5.0 on newer CMake.
- ✅ **Async MAPPO vector env** (`python/xushi2/vector_env.py`) — keeps the
  existing synchronous wrapper as default and adds a multiprocessing backend
  selectable with `ppo.vector_env: async` for later expensive phases.

What's not there yet:

- ❌ Scaled Phase-4 training run / acceptance gate beyond the smoke path.
- ❌ Scaled Phase-5 attention training run / acceptance gate beyond the compact
  objective probe.
- ❌ Native/full Phase-6 grid observation stack and scaled acceptance gate.
- ❌ Full multi-enemy Phase-7 fog state and scaled Phase-7b per-agent fog gate
  beyond the compact diagnostic.
- ❌ Scaled current-vs-current self-play gates and roster expansion beyond the
  compact Phase-11 probe.
- ❌ Full second-hero ability kits (Vanguard, Mender — Phase 10+); roster,
  movement/HP/role state, target-slot action plumbing, Mender Staff beam,
  weapon swap, Sidearm hitscan, and Tether plus Vanguard barrier/Guard Step/
  Warhammer diagnostics are present. Phase 1 stays 1v1 Ranger by design.

## Training

The training entrypoint lives at `python/train/train.py`. The curriculum
ladder is laid out in `docs/rl_design.md` §6.

```bash
# Phase 2 memory-toy (recurrent vs feedforward reference run):
python -m train.train --config experiments/configs/phase2_memory_toy.yaml

# Phase 3 C++ sim + recurrent PPO (smoke):
python -m train.train --config experiments/configs/phase3_ranger_smoke.yaml

# Phase 4 3v3 recurrent MAPPO (smoke):
python -m train.train --config experiments/configs/phase4_mappo_smoke.yaml

# Phase 4 async vector-env smoke:
python -m train.train --config experiments/configs/phase4_mappo_async_smoke.yaml

# Phase 4 compact objective-path probe (BC warm start + one MAPPO update):
python -m train.train --config experiments/configs/phase4_mappo_objective_probe.yaml

# Phase 4 3v3 recurrent MAPPO vs noop scripted bot:
python -m train.train --config experiments/configs/phase4_mappo_noop.yaml

# Phase 4 objective-discovery probe with stronger shaping:
python -m train.train --config experiments/configs/phase4_mappo_noop_probe.yaml

# Phase 4 3v3 recurrent MAPPO vs basic scripted bot:
python -m train.train --config experiments/configs/phase4_mappo_basic.yaml

# Phase 4 known-good objective path + viewer replay dump:
python scripts/diag_phase4_walk_objective.py --dump-replay ../data/replays/phase4_walk_objective_debug.txt

# Dump a Phase 4 MAPPO checkpoint replay for the viewer:
python scripts/dump_replay.py \
  --checkpoint runs/phase4_mappo_objective_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase4_mappo_objective_probe_eval.replay \
  --max-decisions 20

# Phase 5 entity-attention objective probe + viewer replay:
python -m train.train --config ../experiments/configs/phase5_entity_attention_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase5_entity_attention_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase5_entity_attention_probe_eval.replay \
  --max-decisions 20

# Phase 6 entity+grid objective probe + viewer replay:
python -m train.train --config ../experiments/configs/phase6_entity_grid_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase6_entity_grid_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase6_entity_grid_probe_eval.replay \
  --max-decisions 20

# Phase 7 team-shared fog objective probe + eval gate + viewer replay:
python -m train.train --config ../experiments/configs/phase7_team_fog_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase7_team_fog_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase7_team_fog_probe_eval.replay \
  --max-decisions 20

# Phase 7b per-agent fog objective probe + eval gate + viewer replay:
python -m train.train --config ../experiments/configs/phase7_per_agent_fog_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase7_per_agent_fog_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase7_per_agent_fog_probe_eval.replay \
  --max-decisions 20

# Phase 8 randomized-map and cover-marker objective probe + viewer replay:
python -m train.train --config ../experiments/configs/phase8_random_map_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase8_random_map_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase8_random_map_probe_eval.replay \
  --max-decisions 20

# Phase 9 frozen-snapshot opponent probe + viewer replay:
python -m train.train --config ../experiments/configs/phase9_snapshot_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase9_snapshot_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase9_snapshot_probe_eval.replay \
  --max-decisions 20

# Phase 10 target-slot action probe + viewer replay:
python -m train.train --config ../experiments/configs/phase10_target_slot_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase10_target_slot_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase10_target_slot_probe_eval.replay \
  --max-decisions 20

# Phase 11 current-vs-current self-play probe:
python -m train.train --config ../experiments/configs/phase11_current_selfplay_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase11_current_selfplay_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase11_current_selfplay_probe_eval.replay \
  --max-decisions 20

# Phase 11 mixed current/snapshot/anchor league probe:
python -m train.train --config ../experiments/configs/phase11_mixed_league_probe.yaml
python scripts/dump_replay.py \
  --checkpoint runs/phase11_mixed_league_probe/mappo/ckpt_final.pt \
  --output ../data/replays/phase11_mixed_league_probe_eval.replay \
  --seed 1 \
  --max-decisions 20

# Compact MAPPO win-rate/matchup matrix; Phase 9 also writes this automatically
# via run.matrix_eval:
python scripts/eval_mappo_matrix.py \
  --checkpoint runs/phase9_snapshot_probe/mappo/ckpt_final.pt \
  --anchor-bot noop \
  --opponent-checkpoint runs/phase8_random_map_probe/mappo/ckpt_final.pt \
  --episodes 1 \
  --output runs/phase9_snapshot_probe/mappo/matrix_eval.json
python scripts/check_mappo_matrix.py \
  --matrix runs/phase9_snapshot_probe/mappo/matrix_eval.json \
  --min-rows 2 \
  --min-win-rate bot=1.0 \
  --min-win-rate snapshot=1.0 \
  --max-draw-rate bot=0.0 \
  --max-draw-rate snapshot=0.0 \
  --output runs/phase9_snapshot_probe/mappo/matrix_gate.json
```

## License

TBD.
