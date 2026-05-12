# Xushi2 虚实

Deterministic 2D 3v3 control-point shooter simulator for multi-agent RL, with replay tooling and a lightweight viewer.

## Project overview

Xushi2 combines a deterministic C++ simulation core with Python training/evaluation tooling for recurrent PPO/MAPPO experiments across a phased environment ladder.

## Quickstart

- `src/sim/` — deterministic simulation core
- `src/python_bindings/` — `xushi2_cpp` pybind module
- `src/viewer/` — replay/debug viewer
- `python/xushi2/` — env wrappers, obs/reward adapters, vector envs
- `python/train/` — recurrent PPO/MAPPO training stack
- `python/envs/` — phase-specific envs (Phase 3/4/5/6/7/8/9/10/11)
- `experiments/configs/` — training configs
- `docs/` — specs, design docs, plans/results

## Recommended commands (Makefile)

```bash
make build-cpp
make test-cpp
make bench-cpp
make run-bench
make bench-smoke
make py-install
make train-smoke
make format
make lint
make clean
```

## Advanced: underlying raw commands

### C++ build + test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```



### C++ benchmarks

```bash
# Configure benchmark build in Release
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_BENCHMARKS=ON

# Build benchmark targets
cmake --build build -j --target benchmarks

# Run all discovered benchmark binaries via Makefile wrapper
make run-bench

# Or run a benchmark binary directly (Google Benchmark flags)
./build/benchmarks/<benchmark_binary> \
  --benchmark_repetitions=5 \
  --benchmark_min_time=0.1 \
  --benchmark_out=build/benchmarks/results/<benchmark_binary>.json \
  --benchmark_out_format=json
```

### Python setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Common training entrypoints

```bash
# Phase 4 MAPPO smoke
python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml

# Phase 4 async vector-env smoke
python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_async_smoke.yaml

# Phase 11 current self-play probe
python -m train.train --config ../experiments/configs/phase11/probe/phase11_current_selfplay_probe.yaml

# Phase 11 mixed league probe
python -m train.train --config ../experiments/configs/phase11/probe/phase11_mixed_league_probe.yaml
```

### Python quality tools

```bash
cd python
ruff format .
ruff check .
```

## Documentation

See the structured docs index: [`docs/README.md`](docs/README.md).
