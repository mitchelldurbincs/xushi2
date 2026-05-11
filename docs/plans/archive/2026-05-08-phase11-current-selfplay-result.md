# Phase 11 current self-play probe result

Status: compact current-vs-current and mixed current/snapshot/anchor MAPPO
integration is in place. This is a smoke/probe, not the scaled self-play gate.

What changed:

- Added `python/envs/phase11_current_selfplay_mappo.py`, a six-agent Gymnasium
  env where the current policy controls both Team A and Team B in one 3v3 match.
- The env returns `(6, ENTITY_GRID_OBS_DIM)` actor observations, accepts `(6, 6)`
  actions, and emits separate per-agent rewards: slots 0-2 receive Team-A
  reward, slots 3-5 receive Team-B reward.
- The env builds a flattened `6 * CRITIC_DIM` critic buffer for MAPPO
  `ppo.value_per_agent: true`, repeating Team-A centralized critic observations
  across Team-A slots and Team-B centralized critic observations across Team-B
  slots.
- The Phase-8 map/fog diagnostic stack is carried forward: deterministic
  randomized bounds, cover circles, wall segments, layout hashes, partial
  entity-grid observations, and native line-of-sight masking.
- Added Phase-11 registry/entrypoint support and
  `experiments/configs/phase11/probe/phase11_current_selfplay_probe.yaml`, a one-update
  current-vs-current smoke config.
- Extended `run.matrix_eval` so Phase-11 can emit a current-vs-current
  self-play row after training. The compact Phase-11 config writes
  `matrix_eval.json` with `opponent_type=selfplay`.
- Added mixed Phase-11 league sampling through the existing
  `SelfPlaySchedule`: current matches train all six policy slots, while
  snapshot/anchor matches drive Team B from a frozen checkpoint or anchor bot
  and return a dynamic `loss_mask=1,1,1,0,0,0` so PPO only trains Team A for
  those steps.
- Added `experiments/configs/phase11/probe/phase11_mixed_league_probe.yaml`, which samples
  compact `current` / `snapshot` / `anchor` matches and writes Phase-11 matrix
  rows for selfplay, a noop bot anchor, and a frozen Phase-8 snapshot.
- Extended `python/scripts/dump_replay.py` so Phase-11 checkpoints dump all six
  greedy current-policy action slots for current matches and env-side Team-B
  opponent actions for snapshot/anchor matches, with sampled `match_type`,
  schedule, snapshot metadata, and six-entry loss masks in the replay header.
- Extended the viewer replay panel to display `match_type` metadata.
- Fixed MAPPO rollout collection so per-agent value tensors are stored as
  `(num_envs, n_agents, rollout_len)` instead of using the scalar critic index
  path.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase11_current_selfplay.py -q`
  -> 7 passed.
- `python/.venv/bin/python -m pytest tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase11_current_selfplay_checkpoint tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase11_mixed_snapshot_checkpoint tests/test_phase11_current_selfplay.py tests/test_mappo_loss_mask.py tests/test_phase_registry.py::test_phase11_mixed_league_probe_config_is_compact -q`
  -> 16 passed.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase11/probe/phase11_current_selfplay_probe.yaml`
  -> one MAPPO update completed, eval drew 1/1 with mean reward +0.000, wrote
  `python/runs/phase11_current_selfplay_probe/mappo/ckpt_final.pt`, and wrote
  `python/runs/phase11_current_selfplay_probe/mappo/matrix_eval.json` with a
  current-selfplay draw row.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase11/probe/phase11_mixed_league_probe.yaml`
  -> one MAPPO update completed, eval drew 1/1, wrote
  `python/runs/phase11_mixed_league_probe/mappo/ckpt_final.pt`, and wrote
  `python/runs/phase11_mixed_league_probe/mappo/matrix_eval.json` with
  selfplay, noop-bot, and frozen-snapshot rows.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase11_current_selfplay_probe/mappo/ckpt_final.pt --output data/replays/phase11_current_selfplay_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=11`,
  `match_type=current`, `team_size=3`, `loss_mask=1,1,1,1,1,1`, Phase-8
  cover/wall topology, fog metadata, and six current-policy action slots.
- `python/.venv/bin/python scripts/dump_replay.py --checkpoint runs/phase11_mixed_league_probe/mappo/ckpt_final.pt --output ../data/replays/phase11_mixed_league_probe_eval.replay --seed 1 --max-decisions 20`
  -> wrote a 20-decision replay with `phase=11`, `match_type=snapshot`,
  `schedule=current:0.34,snapshot:0.33,anchor:0.33`,
  `loss_mask=1,1,1,0,0,0`, snapshot metadata, and env-side Team-B snapshot
  actions.
- `cmake --build build-viewer --target xushi2_viewer --parallel`
  -> viewer still builds with the Phase-11 mixed replay metadata.

Remaining:

- Scaled current-vs-current/mixed-league gates are still open.
