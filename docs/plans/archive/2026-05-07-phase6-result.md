# Phase 6 entity-grid probe result

Status: compact diagnostic path landed; full Phase-6 OA5-analog acceptance run
is still open.

What changed:

- Added `python/xushi2/grid_obs.py`, which appends a tiny 32x32x3
  egocentric grid to the Phase-5 entity-token observation.
- Added `python/envs/phase6_grid_mappo.py`, preserving the Phase-4 sim,
  reward, action, and centralized critic paths while exposing entity tokens
  plus grid features to the actor.
- Added a Phase-6 registry entry and `phase: 6` support in the MAPPO
  entrypoint and replay-dump path.
- Extended `MappoActorCritic` with an `entity_attention_grid` encoder:
  masked entity attention, a small CNN grid branch, and a fusion layer before
  the recurrent actor core.
- Added `experiments/configs/phase6_entity_grid_probe.yaml`, a compact
  BC-warm-start plus one-update MAPPO objective-path run.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase6_grid_obs.py tests/test_phase_registry.py tests/test_phase4_checkpoint_replay_dump.py -q`
  -> 21 passed.
- `python/.venv/bin/python -m pytest tests -q` -> 199 passed after the
  Phase-7/8/9 probe additions.
- `ctest --test-dir build --output-on-failure` -> 94 passed.
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase6_entity_grid_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +13.000 and mean score
  2.67/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 2.67/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase6_entity_grid_probe/mappo/ckpt_final.pt --output data/replays/phase6_entity_grid_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=6` and `team_size=3` in
  the header.

Remaining:

- This grid is a diagnostic adapter over current flat observations. Native
  sim-side grid construction, richer map channels, and the full
  Vanguard/Ranger/Mender roster remain open.
- Run less-shaped/scaled Phase-5/6 diagnostics once the compact probes stay
  stable.
- Phase 7 fog/LoS must be introduced only after the full-vision grid path has
  a stronger acceptance signal.
