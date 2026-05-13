# Python layer architecture contract

This document defines import-direction and ownership boundaries for the Python stack.

## Layers

### 1) `python/xushi2/` — core runtime-facing primitives

`xushi2` owns reusable building blocks that should stay phase-agnostic:
- C++ binding-facing APIs (`xushi2_cpp` wrappers, sim config, rollout runner).
- Stable observation/reward utility modules used by multiple phases.
- Vector-env/runtime helpers and reusable policy/snapshot/map helpers.

`xushi2` **must not import** from `envs` or `train`.

### 2) `python/envs/` — phase wrappers and compositions

`envs` owns phase-specific Gymnasium env definitions and compositions:
- phase-specific wrappers (`phase3_*`, `phase4_*`, ...), plus standalone envs (e.g., memory toy).
- adaptation from `xushi2` primitives into per-phase observation/action contracts.
- public env entrypoints/factories exported by `envs.__init__`.

`envs` may depend on `xushi2`, but should not depend on `train` internals.

### 3) `python/train/` — algorithm and training orchestration

`train` owns algorithmic training/eval orchestration:
- PPO/MAPPO model/trainer/update logic.
- checkpointing, logging, run orchestration, phase dispatch.

`train` should depend on **env interfaces** (public env entrypoints/factories), not phase-private env modules.

## Import direction rules

Allowed direction:
- `xushi2` → (stdlib/third-party only)
- `envs` → `xushi2`
- `train` → `envs` public APIs and `xushi2` reusable primitives
- `scripts` / `eval` → public APIs (`train.*`, `envs.*`, `xushi2.*`), not `envs.phase*` private modules

Disallowed direction examples:
- `xushi2` importing from `envs`/`train`
- `train/*` importing `envs.phase4_mappo` directly
- `scripts/*` or `eval/*` importing `envs.phase*_...` directly

## Enforcement

```bash
python -m scripts.check_import_boundaries
```

The check fails on forbidden import directions and prints file+line diagnostics.
