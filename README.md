# Xushi2 虚实

A deterministic, top-down 2D, 3v3 control-point team shooter designed as a
multi-agent reinforcement learning environment that is also playable by humans.

Named after the Sun Tzu concept of *虚实* (feint and substance): information
warfare and deception under partial observation.

## Status

**Phase 2 cleared; Phase 3 in progress.** The Phase 2 memory-toy gate is
green (recurrent final −0.107, feedforward final −0.995, gap 0.889) — see
[`docs/plans/2026-04-23-phase2-result.md`](docs/plans/2026-04-23-phase2-result.md)
for the sign-off. The recurrent PPO trainer package, value normalization,
and cosine LR schedule all landed with that result. Phase 3 — wiring the
C++ sim into the recurrent PPO trainer via a 1v1 Ranger env — is in
progress right now. The raylib viewer is still a scaffold — see
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
  `critic_obs.cpp`, `obs_utils.cpp`) — Phase-1 flat actor + critic obs,
  zero-copy into caller-provided numpy buffers, structural actor-leak
  test green.
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
  trainer for 1v1 Ranger; training in progress.

What's a scaffold:

- 🚧 **Viewer** (`src/viewer/src/main.cpp`) — raylib window + 30 Hz loop;
  no keyboard/mouse → action binding, no rendering yet. Build excluded
  from CI due to a known raylib 5.0 + newer-CMake incompatibility;
  build with `-DXUSHI2_BUILD_VIEWER=OFF` until raylib bumps.

What's not there yet:

- ❌ Multi-agent MAPPO (Phase 4+).
- ❌ Batched / vectorized env for parallel rollouts.
- ❌ Fog of war and LoS (Phase 7+).
- ❌ Second heroes (Vanguard, Mender — Phase 10+); Phase 1 stays 1v1
  Ranger by design.

## Training

The training entrypoint lives at `python/train/train.py`. The curriculum
ladder is laid out in `docs/rl_design.md` §6.

```bash
# Phase 2 memory-toy (recurrent vs feedforward reference run):
python -m train.train --config experiments/configs/phase2_memory_toy.yaml

# Phase 3 C++ sim + recurrent PPO (smoke):
python -m train.train --config experiments/configs/phase3_ranger_smoke.yaml
```

## License

TBD.
