# Xushi2 虚实

Deterministic 2D 3v3 control-point shooter simulator for multi-agent RL, with replay tooling and a lightweight viewer.

## Status (May 8, 2026)

- Core C++ sim, Python bindings, replay format, and deterministic test coverage are in place.
- Phase ladder is implemented through **Phase 11 probes** (current self-play + mixed current/snapshot/anchor league).
- MAPPO stack supports centralized critic, entity/grid observations, fog modes, randomized maps, snapshot opponents, target-slot action probing, and sync/async vector env backends.
- Current work is focused on **scaling/acceptance runs** (many phases are probe-complete but not yet fully gated at scale).

## Repo map

- `src/sim/` — deterministic simulation core
- `src/python_bindings/` — `xushi2_cpp` pybind module
- `src/viewer/` — replay/debug viewer
- `python/xushi2/` — env wrappers, obs/reward adapters, vector envs
- `python/train/` — recurrent PPO/MAPPO training stack
- `python/envs/` — phase-specific envs (Phase 3/4/5/6/7/8/9/10/11)
- `experiments/configs/` — training configs
- `docs/` — specs, design docs, plans/results

## Build + test (C++)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Python setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Common training entrypoints

```bash
# Phase 4 MAPPO smoke
python -m train.train --config ../experiments/configs/phase4_mappo_smoke.yaml

# Phase 4 async vector-env smoke
python -m train.train --config ../experiments/configs/phase4_mappo_async_smoke.yaml

# Phase 11 current self-play probe
python -m train.train --config ../experiments/configs/phase11_current_selfplay_probe.yaml

# Phase 11 mixed league probe
python -m train.train --config ../experiments/configs/phase11_mixed_league_probe.yaml
```

## Key docs

- `docs/game_design.md`
- `docs/rl_design.md`
- `docs/observation_spec.md`
- `docs/action_spec.md`
- `docs/replay_format.md`
- `docs/determinism_rules.md`

## License

TBD.
