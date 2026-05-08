# Phase 8 randomized-map probe result

Status: compact randomized-map diagnostic now includes deterministic circular
cover pillars and wall segments wired into native movement, hitscan,
beam/tether line-of-sight, Phase-7 fog masking, stable map-layout hashes,
replay headers, and viewer rendering. Scaled gates are still open.

What changed:

- Added `python/xushi2/map_randomization.py`, a deterministic per-seed map
  bounds randomizer that keeps the arena center fixed while varying width and
  height.
- Added deterministic symmetric cover markers to the Phase-8 map randomizer,
  with a configurable radius. Phase-8 env resets now pass them through as
  native `sim.cover_circles`.
- Added deterministic symmetric wall segments to the Phase-8 map randomizer,
  with configurable count, length, jitter, and half-width. Phase-8 env resets
  now pass them through as native `sim.wall_segments`.
- Added stable 64-bit Phase-8 layout hashes over rounded arena bounds and cover
  circle/wall geometry. The env exposes `map_layout_hash`, replay headers
  include `layout=...`, and the viewer shows it in the replay panel.
- Added explicit `map` bounds, `cover_circles`, and `wall_segments` plumbing through
  `python/xushi2/runner.py` and the pybind `MatchConfig` binding.
- Added native circular cover behavior in the C++ sim:
  movement overlap resolution, Ranger/Mender hitscan blocking, Vanguard
  Warhammer target LoS blocking, Mender Staff/Tether LoS blocking, and a
  `Sim::line_of_sight` helper used by the Phase-7 fog wrapper.
- Added native wall-segment behavior in the C++ sim: config validation,
  hitscan/line-of-sight blocking, wall collision overlap resolution, and a
  movement crossing guard for wall tunneling.
- Added `python/envs/phase8_random_map_mappo.py`, preserving the Phase-7
  observation stack while rebuilding the wrapped env with deterministic
  randomized bounds on each reset.
- Added a Phase-8 registry entry and `phase: 8` support in the MAPPO entrypoint
  and replay-dump path.
- Added `experiments/configs/phase8_random_map_probe.yaml`, a compact
  randomized-map BC-warm-start plus one-update MAPPO objective-path run.
- Extended replay headers and the viewer loader with explicit map bounds,
  layout hashes, radius-aware cover positions, and wall segment markers so a
  dumped randomized-map eval uses the same arena geometry in the viewer.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase8_random_map.py tests/test_phase_registry.py::test_phase8_registry_declares_random_map_shapes tests/test_phase_registry.py::test_phase8_random_map_probe_config_is_compact tests/test_phase_registry.py::test_phase8_random_map_bc_eval_can_be_best_result tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase8_random_map_checkpoint -q`
  -> 7 passed for the compact Phase-8 cover-marker/bounds/replay checks.
- `python/.venv/bin/python -m pytest tests/test_phase7_partial_obs.py tests/test_phase8_random_map.py tests/test_phase_registry.py::test_phase8_random_map_probe_config_is_compact tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase8_random_map_checkpoint -q`
  -> 14 passed for cover-aware fog masking, cover config translation,
  randomized cover circles, and replay headers.
- `python/.venv/bin/python -m pytest tests/test_phase8_random_map.py tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase8_random_map_checkpoint -q`
  -> 10 passed for layout-hash determinism, env/replay propagation, randomized
  cover circles/walls, and config translation.
- `ctest --test-dir build --output-on-failure -R "WallSegment|CoverCircle|Combat|MatchConfig"`
  -> 37 passed, including wall segment hitscan, LoS, movement, and
  config-validation tests.
- `python/.venv/bin/python -m pytest tests -q` -> 222 passed, 1 warning.
- `ctest --test-dir build --output-on-failure` -> 123 passed.
- `git diff --check` -> clean.
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase8_random_map_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +13.000 and mean score
  7.00/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 7.00/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase8_random_map_probe/mappo/ckpt_final.pt --output data/replays/phase8_random_map_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=8`, `fog=1`,
  `team_size=3`, explicit `map_*` bounds, `layout=...`, radius-aware
  `cover=` marker coordinates, and `walls=` marker coordinates in the header.
  Latest header:
  `format=xushi2-replay-v1 phase=8 seed=3519994490 round_seconds=30 action_repeat=3 mech_dmg=7500 mech_fcd=15 mech_hbr=0.75 mech_resp=240 map_min_x=2.467344030501863 map_min_y=-1.2757270715002313 map_max_x=47.53265596949814 map_max_y=51.27572707150023 layout=0xee086c211b794aea cover=17.061:15.534:1.000,20.258:18.553:1.000,29.742:31.447:1.000,32.939:34.466:1.000 walls=16.887:18.680:16.887:23.680:0.250,33.113:26.320:33.113:31.320:0.250 team_size=3 fog=1 last_seen=1 fog_mode=team_shared`

Remaining:

- Richer fixed topology remains open.
- Scaled randomized-map gates should wait until the less-shaped Phase 7/8
  probes are stable.
