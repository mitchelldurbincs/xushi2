# Phase 4 Result

## Survey

Phase 4 already had two major slices in place:

- 3v3 sim plumbing: `MatchConfig::team_size`, six-slot spawn/respawn, and
  C++ tests for the 3v3 path.
- Centralized critic observations: `build_critic_obs` emits the 135-float
  Phase-4 layout, mirrored by `python/xushi2/obs_manifest.py`.
- MAPPO-shaped env: `python/envs/phase4_mappo.py` exposes `(3, 31)` actor
  observations, `(3, 6)` actions, per-agent team-broadcast rewards, and a
  caller-buffered `build_critic_obs(out)` hook.

The missing Phase-4 piece was integration: `phase: 4` was absent from the
phase registry and training entrypoint, and the existing recurrent PPO trainer
only accepted single-agent flat env outputs. It could not consume
multi-agent actor observations or the separate centralized critic buffer.

## Plan

1. Register Phase 4 as a first-class phase with actor/critic dimensions and a
   `Phase4MappoEnv` factory.
2. Add a recurrent MAPPO smoke trainer with a shared recurrent actor over
   per-agent actor obs and a centralized critic over the 135-float critic obs.
3. Wire `phase: 4` through `python/train/train.py`, add a smoke config, and
   checkpoint the MAPPO model.
4. Add pytest coverage for registry shape, config parsing, one MAPPO update,
   and checkpoint creation.
5. Run Python and C++ verification.

## Completed

- Added `python/train/mappo.py`.
- Added `phase: 4` registry and training entrypoint support.
- Added `experiments/configs/phase4_mappo_smoke.yaml`.
- Added `experiments/configs/phase4_mappo_basic.yaml` for the first real
  scripted-opponent diagnostic run.
- Expanded MAPPO eval logging to include wins/losses/draws, terminal vs
  truncation counts, final tick, score, and kills.
- Updated `README.md` current-state and training instructions.
- Adjusted Phase-4 env `state_hash` info to hex text so uint64 hashes do not
  overflow Gymnasium vector/info collation.

Verification:

- `python/.venv/bin/python -m pytest tests -q` -> 149 passed.
- `ctest --test-dir build --output-on-failure` -> 94 passed.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase4_mappo_smoke.yaml`
  -> two MAPPO smoke updates completed and wrote checkpoints.
- A one-update `phase4_mappo_basic.yaml` diagnostic override completed:
  untrained policy lost 0/2 eval episodes against `basic`, ending at tick 900
  with mean score 0/7. This establishes the expected pre-training baseline for
  the longer diagnostic run.
- `python/.venv/bin/python scripts/diag_phase4_walk_objective.py` -> hardcoded
  Team A walk-to-objective policy scores 7/0 against `noop` through the public
  Phase4MappoEnv action path. This confirms the env/action/objective path works;
  the current MAPPO failure to score is an exploration/training-signal issue.
