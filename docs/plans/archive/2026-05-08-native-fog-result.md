# Native fog actor-observation slice result

Status: the native actor enemy fields now honor `MatchConfig::fog_of_war_enabled`
and cover/wall line-of-sight, and 3v3 actor observations use counterpart enemy
slots instead of always reading the first opposite-team hero. This closes the
single-enemy native fog leak and the paired-slot leak, but not the final
multi-enemy token observation layout.

What changed:

- Added a `Sim`-aware `obs_utils::visible_enemy_1v1` overload. When native fog
  is enabled, alive enemies blocked by native cover/wall line-of-sight are
  hidden from actor observations.
- Kept the raw `MatchState` helper for critic/tests that intentionally require
  full-state access.
- Updated `build_actor_obs_phase1` so all enemy fields use the native fog-aware
  helper.
- In 3v3 configs, the `Sim` overload now resolves the counterpart enemy slot
  (`0->3`, `1->4`, `2->5`, and mirrored for Team B) before applying fog,
  instead of always returning the first occupied enemy.
- Added `obs_utils::observable_enemy_slots` plus the Python binding
  `xushi2_cpp.observable_enemy_slots`, giving Python env adapters a native
  six-slot visibility mask for all opposite-team heroes under fog/LoS rules.
- Updated Phase-7/Phase-11 env fog wrappers to consume that native mask and
  avoid the previous global team-shared union where one visible paired enemy
  could reveal every row's enemy token.
- Added Python binding regressions proving that the same LoS-blocked enemy is
  hidden when `fog_of_war_enabled=true` and visible when it is false.
- Upgraded the viewer fog replay path to draw native LoS debug rays between
  opposing live heroes. Green rays are visible; red rays are blocked. The replay
  panel now reports visible/total LoS pair counts.

Verification:

- `cmake --build build --target xushi2_cpp --parallel` -> rebuilt the C++ sim
  and Python extension.
- `ctest --test-dir build -R "ObsUtils|Obs|Combat|Config" --output-on-failure`
  -> 89/89 focused C++ tests passed.
- `python/.venv/bin/python -m pytest tests/test_bindings_obs.py tests/test_phase7_partial_obs.py tests/test_phase11_current_selfplay.py -q`
  -> 29 passed.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase7_team_fog_probe.yaml`
  -> compact Phase-7 team-shared fog run completed after the native per-slot
  visibility update, with `mappo_final=13.000`.
- `python/.venv/bin/python scripts/dump_replay.py --checkpoint runs/phase7_team_fog_probe/mappo/ckpt_final.pt --output ../data/replays/phase7_team_fog_probe_eval.replay --max-decisions 20`
  -> refreshed the Phase-7 fog replay artifact.
- `cmake --build build-viewer --target xushi2_viewer --parallel`
  -> viewer target built successfully with the LoS debug overlay.

Remaining:

- Widening actor observations to true multi-enemy tokens remains open. The
  current entity-grid adapter still has one enemy token per actor row, now
  backed by native per-slot fog instead of the old first-enemy/global-union
  behavior.
