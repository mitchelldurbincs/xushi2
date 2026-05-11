# Xushi2 虚实

Deterministic 2D 3v3 control-point shooter simulator for multi-agent RL, with replay tooling and a lightweight viewer.

## Project overview

Xushi2 combines a deterministic C++ simulation core with Python training/evaluation tooling for recurrent PPO/MAPPO experiments across a phased environment ladder.

## Quickstart

### Build + test (C++)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
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
python -m train.train --config ../experiments/configs/phase4_mappo_smoke.yaml

# Phase 11 current self-play probe
python -m train.train --config ../experiments/configs/phase11_current_selfplay_probe.yaml
```

## Documentation

See the structured docs index: [`docs/README.md`](docs/README.md).
