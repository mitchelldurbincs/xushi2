# Phase 10 target-slot probe result

Status: target-slot action-interface plumbing, valid-target masking, a mirrored
Vanguard/Ranger/Mender roster smoke, a Mender Staff/Sidearm weapon-swap
diagnostic plus Staff beam, Sidearm hitscan, and Tether, and Vanguard barrier /
Guard Step / Warhammer diagnostics, and Ranger Mark Target landed; full
second-hero ability kits are still open.

What changed:

- Added an optional MAPPO categorical target head via `target_action_dim`.
- Extended action sampling, greedy eval, PPO logprob/entropy recomputation, and
  rollout metrics to handle the seventh `target_slot` factor.
- Appended the fixed-order entity-token valid-target mask to Phase-10 actor
  observations and masked invalid target logits during sampling, greedy eval,
  and PPO recomputation.
- Added `python/envs/phase10_target_slot_mappo.py`, which preserves the Phase-8
  randomized-map/fog observation stack, accepts `(3, 7)` learner actions, clamps
  the target-slot column to `{0, 1, 2}`, and forwards all seven controls to the
  current simulator.
- Added slot-indexed `MatchConfig.hero_kinds`, Python config parsing, and spawn
  initialization for Vanguard/Ranger/Mender kind, role, HP, and respawn kind
  preservation.
- Added a Phase-10 diagnostic Mender `ability_1` Weapon Swap that toggles
  Staff/Sidearm, breaks any beam lock, and arms a short cooldown.
- Added a Phase-10 diagnostic Mender Staff beam that locks the nearest aimed
  ally in range, heals while held, breaks on release/swap/death/range, and is
  drawn in the viewer. Added explicit coverage that allied Vanguard barriers do
  not block or absorb Staff healing.
- Added a Phase-10 diagnostic Mender Sidearm primary: held hitscan pistol,
  fire-rate gated, blocked by enemy Vanguard barriers, with viewer tracers.
- Added a Phase-10 diagnostic Mender Tether that snaps near the nearest aimed
  ally in range, arms cooldown, and draws a short viewer trail.
- Added a Phase-10 diagnostic Vanguard held barrier that slows movement, absorbs
  Ranger revolver shots before hero HP, breaks at zero barrier HP, and arms a
  redeploy cooldown.
- Added a Phase-10 diagnostic Vanguard `ability_2` Guard Step impulse that
  dashes forward along aim direction, clamps to map bounds, and arms cooldown.
- Added a Phase-10 diagnostic Vanguard Warhammer primary that damages the
  nearest enemy in a short forward cone, arms weapon cooldown, and is suppressed
  while the barrier is active.
- Added a Phase-10 diagnostic Ranger `ability_2` Mark Target impulse that
  consumes the enemy `target_slot`, marks the nearest visible enemy in Revolver
  range for 3 seconds, respects cover line-of-sight, and arms a 6-second
  cooldown.
- Added a Phase-10 registry entry and `phase: 10` support in the MAPPO
  entrypoint and replay-dump path.
- Added `experiments/configs/phase10_target_slot_probe.yaml`, a compact
  target-slot BC-warm-start plus one-update MAPPO run over a mirrored
  Vanguard/Ranger/Mender composition.
- Tagged Phase-10 replay headers with `target_slot=1` and `heroes=...`, writes
  the seventh action column to replay lines, and surfaced target-slot,
  hero-kind, Mender weapon, and Ranger mark metadata in the raylib viewer.
- Upgraded the viewer target-slot replay path to show current target-token
  selections (`self` / `enemy` / `objective`) in the replay panel and annotate
  selected tokens near heroes in the arena. Enemy-token selections draw a light
  diagnostic line toward the paired enemy slot.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase10_target_slot.py tests/test_phase_registry.py::test_phase10_registry_declares_target_slot_shapes tests/test_phase_registry.py::test_phase10_target_slot_probe_config_is_compact tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase10_target_slot_checkpoint -q`
  -> 6 passed.
- `ctest --test-dir build --output-on-failure -R "RangerMarkTarget|Combat"`
  -> 25 passed.
- `ctest --test-dir build -R "MenderTether|MenderStaffBeam|MenderSidearm|MenderAbility|VanguardWarhammer|VanguardGuardStep|VanguardBarrier" --output-on-failure`
  -> 15 passed after adding the allied-barrier Staff-beam interaction test.
- `ctest --test-dir build -R "MenderStaffBeamHealsThroughAlliedBarrier|MenderStaffBeam|VanguardBarrier" --output-on-failure`
  -> 6 passed.
- `python/.venv/bin/python -m pytest tests/test_phase10_target_slot.py -q`
  -> 3 passed.
- `python/.venv/bin/python -m pytest tests/test_phase10_target_slot.py tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase10_target_slot_checkpoint -q`
  -> 4 passed after the viewer target-token overlay slice.
- `python/.venv/bin/python -m pytest tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase10_target_slot_checkpoint -q`
  -> 1 passed.
- `python/.venv/bin/python -m pytest tests -q` -> 216 passed, 1 warning.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase10_target_slot_probe.yaml`
  -> mixed-roster BC eval won 2/2 with mean reward +13.000 and mean score
  7.00/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 7.00/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase10_target_slot_probe/mappo/ckpt_final.pt --output data/replays/phase10_target_slot_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=10`,
  `heroes=vanguard,ranger,mender,vanguard,ranger,mender`, `target_slot=1`,
  `fog=1`, `team_size=3`, explicit `map_*` bounds, and 43 fields per decision
  line (`tick + 6 slots * 7 action fields`).
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python scripts/dump_replay.py --checkpoint runs/phase10_target_slot_probe/mappo/ckpt_final.pt --output ../data/replays/phase10_target_slot_probe_eval.replay --max-decisions 20`
  -> refreshed the Phase-10 target-slot viewer replay artifact.
- `ctest --test-dir build --output-on-failure` -> 118 passed.
- `git diff --check` -> clean.

Remaining:

- Vanguard/Mender currently differ by kind, role, HP, and movement speed; Mender
  has Staff beam, Staff/Sidearm weapon swap, Sidearm hitscan, and Tether;
  Vanguard has a held barrier diagnostic, Guard Step diagnostic, and Warhammer
  primary diagnostic. Other dedicated ability kits remain open.
