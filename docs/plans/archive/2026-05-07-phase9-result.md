# Phase 9 snapshot-opponent probe result

Status: compact frozen-snapshot opponent diagnostic now includes deterministic
weighted league-manifest sampling, Phase-8 randomized cover/wall topology, and
an explicit PPO agent loss mask plus a capped snapshot-retention manifest.
Current-vs-current integration has moved into the compact Phase-11 probe.

What changed:

- Added `python/xushi2/snapshot_policy.py`, which loads a frozen MAPPO
  checkpoint, converts current sim actor observations into that checkpoint's
  expected observation format, and emits greedy recurrent actions.
- Added `SnapshotPool`, a deterministic uniform sampler over snapshot paths.
- Added `SnapshotLeague`, a deterministic weighted sampler over named
  `latest` / `historical` / `anchor` snapshot groups, plus compact league
  summaries surfaced through env info, checkpoint config, replay headers, and
  the viewer replay panel.
- Added `python/xushi2/self_play_schedule.py`, a deterministic sampler for the
  70/20/10 `current` / `snapshot` / `anchor` match-type mix. The Phase-9 probe
  now carries this `self_play_schedule` metadata in checkpoint configs.
- Extended `Phase4MappoEnv` with an optional env-side opponent policy and
  surfaced opponent actions in `info` for exact replay dumping.
- Added `python/envs/phase9_snapshot_mappo.py`, preserving the Phase-8
  randomized-map/fog observation stack while Team B is driven by a frozen
  snapshot. Reset now passes the same deterministic cover circles, wall
  segments, and layout hash into the sim and env info that replay dumping
  writes into the header.
- Added `ppo.agent_loss_mask` support in the MAPPO trainer. The default mask is
  all learner slots, old checkpoints without the field still load as all-on,
  and focused tests cover masked reward averaging and config validation.
- Added MAPPO `value_per_agent` rollout/update support as the value-function
  foundation for current-vs-current self-play. A focused test verifies that
  opposing Team-A/Team-B rewards keep separate positive and negative GAE streams
  instead of collapsing into one zero-mean env reward.
- Added `python/xushi2/snapshot_retention.py`, which writes a compact
  `snapshot_league.json` manifest with capped `latest` entries, best-scoring
  `historical` preservation, and configured anchors. The Phase-9 smoke enables
  this with `max_latest: 2` and `preserve_best: 1`.
- Added a Phase-9 registry entry and `phase: 9` support in the MAPPO entrypoint
  and replay-dump path.
- Added `experiments/configs/phase9_snapshot_probe.yaml`, a compact
  frozen-snapshot-opponent BC-warm-start plus one-update MAPPO run with a
  70/20/10 league manifest. For the smoke run all groups point at the same
  known-good Phase-8 checkpoint so the sampling path is exercised without
  changing the expected objective outcome.
- Extended six-slot replay dumping and the viewer replay panel so Phase-9
  replays include both learner actions, nonzero snapshot-opponent actions,
  league/snapshot metadata, `schedule=...`, and `loss_mask=...`.
- Added `python/scripts/eval_mappo_matrix.py`, a compact matchup evaluator
  that runs 3-agent MAPPO checkpoints against anchor bots and frozen snapshot
  checkpoints, prints win/draw/reward/score rows, and can write JSON for later
  league gates.
- Added `run.matrix_eval` support to MAPPO training. The Phase-9 probe now
  emits `matrix_eval.json` after `ckpt_final.pt`, using one noop anchor row and
  one frozen Phase-8 snapshot row.
- Added `python/xushi2/mappo_matrix_gate.py` and
  `python/scripts/check_mappo_matrix.py`, which gate matrix rows with compact
  row-count, win-rate, and draw-rate thresholds. The Phase-9 probe now writes
  `matrix_gate.json` and fails training if the compact matrix gate fails.
- Extended `SnapshotRetention` so matrix-evaluated checkpoints carry
  `matrix_score`, `matrix_rows`, and optional `matrix_gate_passed` metadata in
  `snapshot_league.json`; historical preservation now prefers matrix-gated
  checkpoints over score-only records. The integrated Phase-9 run records
  `ckpt_final.pt` as the historical snapshot when the compact matrix gate
  passes.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase9_snapshot.py tests/test_phase_registry.py::test_phase9_snapshot_probe_config_is_compact -q`
  -> 7 passed, including deterministic weighted league sampling,
  current/snapshot/anchor match-type scheduling, snapshot retention manifest
  capping, env info, and compact config metadata.
- `python/.venv/bin/python -m pytest tests/test_mappo_loss_mask.py tests/test_phase9_snapshot.py tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase9_snapshot_checkpoint tests/test_phase_registry.py::test_phase9_snapshot_probe_config_is_compact -q`
  -> 10 passed, including PPO loss-mask validation, Phase9 topology
  propagation, and replay-header metadata.
- `python/.venv/bin/python -m pytest tests/test_mappo_loss_mask.py -q`
  -> 5 passed, including per-agent GAE for opposing current-vs-current rewards.
- `python/.venv/bin/python -m pytest tests -q` -> 226 passed, 1 warning.
- `ctest --test-dir build --output-on-failure` -> 114 passed after the Phase-8
  native-cover additions. Full suite not rerun after the Phase-9
  league-manifest metadata slice.
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase9_snapshot_probe.yaml`
  -> BC eval won 2/2 vs the frozen Phase-8 league-sampled snapshot with mean
  reward +13.000 and mean score 7.00/0.00; one conservative-LR MAPPO update
  then completed and post-update eval also won 2/2 with mean reward +13.000
  and mean score 7.00/0.00. The run wrote
  `python/runs/phase9_snapshot_probe/mappo/snapshot_league.json` with one
  latest checkpoint, one historical checkpoint, and one configured anchor in
  the compact probe.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase9_snapshot_probe/mappo/ckpt_final.pt --output data/replays/phase9_snapshot_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=9`, `fog=1`, `team_size=3`,
  explicit `map_*` bounds, cover/wall geometry, all six action slots,
  `loss_mask=...`, `schedule=...`, and `league=...`, `snapshot_group=...`,
  and `snapshot=...` metadata. Latest header:
  `format=xushi2-replay-v1 phase=9 seed=3519994490 round_seconds=30 action_repeat=3 mech_dmg=7500 mech_fcd=15 mech_hbr=0.75 mech_resp=240 map_min_x=2.467344030501863 map_min_y=-1.2757270715002313 map_max_x=47.53265596949814 map_max_y=51.27572707150023 layout=0xce9d6eea322176aa cover=16.483:15.528:1.000,21.225:18.938:1.000,28.775:31.062:1.000,33.517:34.472:1.000 walls=16.887:18.680:16.887:23.680:0.250,33.113:26.320:33.113:31.320:0.250 team_size=3 loss_mask=1,1,1 schedule=current:0.7,snapshot:0.2,anchor:0.1 league=latest:0.7:1,historical:0.2:1,anchor:0.1:1 snapshot_group=latest snapshot=ckpt_final.pt fog=1 last_seen=1 fog_mode=team_shared`
- `python/.venv/bin/python -m pytest tests/test_mappo_matrix_eval.py -q`
  -> 1 passed, covering bot and snapshot matrix rows plus JSON output.
- `python/.venv/bin/python scripts/eval_mappo_matrix.py --checkpoint runs/phase9_snapshot_probe/mappo/ckpt_final.pt --anchor-bot noop --opponent-checkpoint runs/phase8_random_map_probe/mappo/ckpt_final.pt --episodes 1 --output runs/phase9_snapshot_probe/mappo/matrix_eval.json`
  -> wrote a compact matrix with 1/1 wins against both noop and the frozen
  Phase-8 snapshot in this diagnostic setup.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase9_snapshot_probe.yaml`
  -> verified the integrated `run.matrix_eval` hook: the one-update Phase-9
  run completed with `mappo_final=13.000` and wrote
  `python/runs/phase9_snapshot_probe/mappo/matrix_eval.json` with the noop and
  frozen-snapshot rows, plus a passing
  `python/runs/phase9_snapshot_probe/mappo/matrix_gate.json`.
- `python/.venv/bin/python -m pytest tests/test_mappo_matrix_gate.py tests/test_mappo_matrix_eval.py tests/test_phase_registry.py::test_phase9_snapshot_probe_config_is_compact -q`
  -> 6 passed, covering gate pass/fail summaries, CLI output, matrix script
  output, and training-hook artifact output.
- `python/.venv/bin/python -m pytest tests/test_phase9_snapshot.py::test_snapshot_retention_caps_latest_and_preserves_best tests/test_phase9_snapshot.py::test_snapshot_retention_prefers_matrix_passing_records tests/test_mappo_matrix_eval.py::test_matrix_eval_updates_snapshot_retention_manifest tests/test_mappo_matrix_eval.py::test_train_config_matrix_eval_writes_post_training_artifact -q`
  -> 4 passed, covering score-only retention, matrix-preferred retention, and
  the training hook that writes final checkpoint matrix metadata into the
  manifest.
- `python/.venv/bin/python scripts/check_mappo_matrix.py --matrix runs/phase9_snapshot_probe/mappo/matrix_eval.json --min-rows 2 --min-win-rate bot=1.0 --min-win-rate snapshot=1.0 --max-draw-rate bot=0.0 --max-draw-rate snapshot=0.0 --output runs/phase9_snapshot_probe/mappo/matrix_gate_cli.json`
  -> PASS against the real compact Phase-9 matrix artifact.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase9_snapshot_probe.yaml`
  -> after the matrix-retention slice, the real compact run wrote
  `snapshot_pool_matrix score=+1.000 gate=pass`; the resulting
  `python/runs/phase9_snapshot_probe/mappo/snapshot_league.json` has
  `historical: [runs/phase9_snapshot_probe/mappo/ckpt_final.pt]` and records
  `matrix_score=1.0`, `matrix_rows=2`, `matrix_gate_passed=true`.

Remaining:

- Phase-9 itself remains a frozen-snapshot opponent diagnostic; current
  self-play is covered by Phase-11 rather than mixed into this env.
- Multi-checkpoint matrix gates across a larger opponent pool are still open.
