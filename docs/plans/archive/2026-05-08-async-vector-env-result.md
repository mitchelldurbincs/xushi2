# Async MAPPO vector env result

Status: a multiprocessing vector backend landed for MAPPO smoke and later
expensive phases. The synchronous backend remains the default.

What changed:

- Added `XushiAsyncVectorEnv` beside `XushiVectorEnv` with the same
  `reset`, `step`, `critic_obs`, and `close` API.
- Added `make_xushi_vector_env(..., backend=...)` and wired MAPPO to
  `ppo.vector_env`, accepting `sync` or `async`.
- Added `experiments/configs/phase4/smoke/phase4_mappo_async_smoke.yaml`, a one-update
  Phase-4 smoke config that exercises the async backend.
- Updated the README phase-run list and status checklist.

Verification:

- `python/.venv/bin/python -m pytest tests/test_vector_env.py tests/test_phase_registry.py::test_phase4_config_can_select_async_vector_backend -q`
  -> 6 passed.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_async_smoke.yaml`
  -> one MAPPO update completed; eval produced a timeout draw at tick 90 and
  `mappo_final=0.000`, which is expected for a no-op smoke config.
