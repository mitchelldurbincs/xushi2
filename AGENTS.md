# xushi2 Agent Instructions

## Project identity

xushi2 is a deterministic 2D 3v3 control-point hero-shooter simulator for multi-agent reinforcement learning.

The repo uses MAPPO, not MAPO.

The project already has:
- C++20 simulation core
- pybind11 Python bindings producing `xushi2_cpp`
- Python training/env wrappers
- phase-driven experiment configs under `experiments/configs/`
- W&B integration
- TensorBoard support
- replay tooling
- pytest + GoogleTest
- CI
- determinism tests

Do not reinvent experiment tracking, baseline config structure, or phase metadata unless a card explicitly asks for it.

## Experiment identity

An experiment is identified by:

```text
git_commit + phase_config_path + seeds + W&B run URL + replay/artifact paths