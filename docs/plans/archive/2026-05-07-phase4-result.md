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
- Added `python/xushi2/vector_env.py` and routed MAPPO rollout collection
  through the sync vector wrapper.
- Added `phase: 4` registry and training entrypoint support.
- Added `experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml`.
- Added `experiments/configs/phase4/probe/phase4_mappo_objective_probe.yaml` for a compact
  BC-warm-start plus one-update MAPPO objective-path run.
- Added `experiments/configs/phase4/baseline/phase4_mappo_basic.yaml` for the first real
  scripted-opponent diagnostic run.
- Fixed `ckpt_final` selection so a BC pretrain eval can remain the best
  checkpoint if the first PPO/MAPPO update regresses.
- Expanded MAPPO eval logging to include wins/losses/draws, terminal vs
  truncation counts, final tick, score, and kills.
- Upgraded the raylib viewer with pause, single-step, reset, playback speed,
  and mode/replay-position panel readouts.
- Extended text replay playback to support all six Phase-4 action slots while
  preserving the legacy slot0/slot3 replay format.
- Added `python/scripts/diag_phase4_walk_objective.py --dump-replay` for a
  small known-good 3v3 objective replay.
- Extended `python/scripts/dump_replay.py` to load Phase-4 MAPPO checkpoints
  and write six-slot text replays for the viewer.
- Added the initial Phase-5 entity-attention path after the Phase-4 result:
  tokenized actor-observation adapter, masked attention MAPPO encoder, compact
  probe config, and Phase-5 replay dump support. See
  `docs/plans/2026-05-07-phase5-result.md`.
- Updated `README.md` current-state and training instructions.
- Adjusted Phase-4 env `state_hash` info to hex text so uint64 hashes do not
  overflow Gymnasium vector/info collation.

Verification:

- `python/.venv/bin/python -m pytest tests -q` -> 199 passed after the
  Phase-5/6/7/8/9 probe additions.
- `ctest --test-dir build --output-on-failure` -> 94 passed.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml`
  -> two MAPPO smoke updates completed and wrote checkpoints.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase4/probe/phase4_mappo_objective_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +13.000 and mean score
  1.40/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 1.47/0.00.
- `cmake -S . -B build_viewer -DXUSHI2_BUILD_VIEWER=ON -DXUSHI2_BUILD_TESTS=OFF -DXUSHI2_BUILD_PYTHON_MODULE=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5`
  followed by `cmake --build build_viewer --target xushi2_viewer --parallel`
  -> viewer target compiled successfully.
- A one-update `phase4_mappo_basic.yaml` diagnostic override completed:
  untrained policy lost 0/2 eval episodes against `basic`, ending at tick 900
  with mean score 0/7. This establishes the expected pre-training baseline for
  the longer diagnostic run.
- `python/.venv/bin/python scripts/diag_phase4_walk_objective.py` -> hardcoded
  Team A walk-to-objective policy scores 7/0 against `noop` through the public
  Phase4MappoEnv action path. This confirms the env/action/objective path works;
  the current MAPPO failure to score is an exploration/training-signal issue.
- `python/.venv/bin/python python/scripts/diag_phase4_walk_objective.py --dump-replay data/replays/phase4_walk_objective_debug.txt`
  -> wrote a 301-line ignored debug replay for `xushi2_viewer --replay`.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase4_mappo_objective_probe/mappo/ckpt_final.pt --output data/replays/phase4_mappo_objective_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay from the compact Phase-4 MAPPO
  objective-probe checkpoint.
