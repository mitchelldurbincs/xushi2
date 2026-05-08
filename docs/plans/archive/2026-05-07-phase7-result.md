# Phase 7 partial-observation probe result

Status: compact team-shared and per-agent fog diagnostics plus last-seen
markers and adapter-level leak tests landed; native per-slot LoS/fog is now
wired into the diagnostic adapter.

What changed:

- Added `python/xushi2/partial_obs.py`, which masks the enemy token and enemy
  grid channel when the enemy is outside a configurable visibility radius.
- Added diagnostic last-seen enemy markers: when an enemy leaves visibility,
  the actor receives a stale enemy-marker token/grid cell with no live HP,
  velocity, aim, ammo, or cooldown state.
- Added an explicit hidden-live-state regression test that mutates hidden enemy
  HP, position, velocity, and objective-presence fields and verifies the
  resulting actor observation is unchanged except for the stale last-seen
  marker.
- Added `python/envs/phase7_fog_mappo.py`, preserving the Phase-4 sim, reward,
  action, and centralized critic paths while exposing Phase-6 entity+grid
  observations with diagnostic fog masking and last-seen state.
- Updated the env wrapper to consume the native six-slot
  `observable_enemy_slots` mask, so team-shared diagnostic fog no longer
  globally reveals every row's enemy token when one paired enemy is visible.
- Added a Phase-7 registry entry and `phase: 7` support in the MAPPO entrypoint
  and replay-dump path.
- Added `experiments/configs/phase7_team_fog_probe.yaml`, a compact
  team-shared fog BC-warm-start plus one-update MAPPO objective-path run.
- Added `experiments/configs/phase7_per_agent_fog_probe.yaml`, the same compact
  objective-path smoke with `fog_mode: per_agent`.
- Added a generic MAPPO eval gate helper and wired the Phase-7 compact probes to
  require 2/2 post-update wins, no draws, reward >= +10.0, score A >= 7.0,
  and score B <= 0.0, writing `eval_gate.json` in the run directory.
- Added replay `fog=1` / `last_seen=1` / `fog_mode=...` metadata and displayed
  it in the raylib viewer replay panel.

Verification:

- `python/.venv/bin/python -m pytest tests/test_phase7_partial_obs.py tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase7_team_fog_checkpoint tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase7_per_agent_fog_checkpoint -q`
  -> 9 passed.
- `python/.venv/bin/python -m pytest tests/test_phase7_partial_obs.py -q`
  -> 8 passed, including the hidden live-state leak regression.
- `python/.venv/bin/python -m pytest tests -q` -> 216 passed after the
  Phase-10 kit additions.
- `ctest --test-dir build --output-on-failure` -> 118 passed.
- `cmake --build build-viewer --target xushi2_viewer --parallel` -> viewer
  target compiled successfully.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase7_team_fog_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +13.000 and mean score
  7.00/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 7.00/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase7_team_fog_probe/mappo/ckpt_final.pt --output data/replays/phase7_team_fog_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=7`, `fog=1`,
  `last_seen=1`, `fog_mode=team_shared`, and `team_size=3` in the header.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase7_per_agent_fog_probe.yaml`
  -> BC eval won 2/2 vs `noop` with mean reward +13.000 and mean score
  7.00/0.00; one conservative-LR MAPPO update then completed and post-update
  eval also won 2/2 with mean reward +13.000 and mean score 7.00/0.00.
- `python/.venv/bin/python python/scripts/dump_replay.py --checkpoint python/runs/phase7_per_agent_fog_probe/mappo/ckpt_final.pt --output data/replays/phase7_per_agent_fog_probe_eval.replay --max-decisions 20`
  -> wrote a 20-decision ignored replay with `phase=7`, `fog=1`,
  `last_seen=1`, `fog_mode=per_agent`, and `team_size=3` in the header.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase7_team_fog_probe.yaml`
  -> rerun after the native per-slot fog update; compact probe still completed
  with `mappo_final=13.000`.
- `python/.venv/bin/python -m train.train --config ../experiments/configs/phase7_team_fog_probe.yaml`
  -> rerun after adding the eval gate; post-update eval passed the gate with
  2/2 wins, mean reward +13.000, mean score 7.00/0.00, and wrote
  `python/runs/phase7_team_fog_probe/mappo/eval_gate.json`.
- `python/.venv/bin/python scripts/dump_replay.py --checkpoint runs/phase7_team_fog_probe/mappo/ckpt_final.pt --output ../data/replays/phase7_team_fog_probe_eval.replay --max-decisions 20`
  -> refreshed the viewer replay artifact from the updated fog path.

Remaining:

- Actor observations still have one enemy token per row; true multi-enemy
  token observations are still a later widening step.
