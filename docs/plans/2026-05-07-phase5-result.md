# Phase 5 entity-attention probe result

Status: compact diagnostic path landed; scaled Phase-5 acceptance run is still
open.

What changed:

- Added `python/xushi2/entity_obs.py`, a Phase-4-flat-observation adapter that
  emits self/enemy/objective entity tokens plus a valid-token mask.
- Added `python/envs/phase5_entity_mappo.py`, which keeps the Phase-4 sim,
  reward, action, and centralized critic paths but exposes entity-token actor
  observations.
- Added a Phase-5 registry entry and `phase: 5` support in the MAPPO training
  entrypoint.
- Wired `MappoActorCritic` to select either the original flat MLP encoder or
  the masked `EntityAttentionEncoder`.
- Added `experiments/configs/phase5_entity_attention_probe.yaml`, a compact
  BC-warm-start plus one-update MAPPO objective-path run.
- Extended checkpoint replay dumping so Phase-5 MAPPO checkpoints produce the
  same six-slot text replay format as Phase 4.
- Added replay `phase=<n>` metadata and displayed it in the raylib viewer's
  replay panel.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase5_entity_obs.py tests/test_entity_attention.py tests/test_phase_registry.py -q`
  -> 19 passed.
- `python/.venv/bin/python -m pytest tests/test_phase4_checkpoint_replay_dump.py -q`
  -> 2 passed.
- `python/.venv/bin/python -m pytest tests -q` -> 199 passed after the
  Phase-6/7/8/9 probe additions.
- `ctest --test-dir build --output-on-failure` -> 94 passed.
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase5_entity_attention_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +12.978 and mean score
  1.23/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +12.979 and mean score 1.27/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase5_entity_attention_probe/mappo/ckpt_final.pt --output data/replays/phase5_entity_attention_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=5` and `team_size=3` in
  the header.

Remaining:

- Replace the adapter with native Phase-5 entity-token observation construction
  once the sim exposes richer per-entity data.
- Run a less-shaped/scaled Phase-5 training diagnostic after the compact path is
  stable.
- Add Phase-6 egocentric grid observations and CNN fusion as the next ladder
  delta.
