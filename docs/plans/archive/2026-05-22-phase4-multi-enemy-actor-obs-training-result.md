# Phase 4 Multi-Enemy Actor Observation Training Result

Date: 2026-05-22

## Status

`NOT_CLEARED`

## Scope

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
- Seed: `3519994490`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes.
- W&B:
  https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw
- Output directory:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/`

## Run

The bounded W&B run completed 100/100 updates using the opt-in multi-enemy
actor observation and `warm_start_migration: compatible_exact`.

Preflight passed:

- `py -3.13 -m scripts.check_import_boundaries`: PASS
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q`:
  `11 passed`
- `py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py -q`:
  `8 passed`

W&B URL was captured from:
`python/wandb/run-20260522_141504-ud4c09jw/files/output.log`.

## Metrics

Best eval was update 50:

- `0W/50L/0D`
- mean reward `-11.0`
- score `0.00/37.00`
- Team A hit/fire `0.0`

Final eval at update 100:

- `0W/50L/0D`
- mean reward `-11.0`
- score `0.00/37.00`
- Team A hit/fire `0.0`
- objective focus fraction `0.0`

Matrix eval:

- vs `noop`: `0W/0L/50D`, score `0.00/0.00`
- vs `weak_basic_v2`: `0W/50L/0D`, score `0.00/37.00`
- vs `basic`: `0W/50L/0D`, score `0.00/37.00`

Transfer summary gate status: `evidence_insufficient`.

## Artifacts

- Checkpoint manifest:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/checkpoint_manifest.json`
- Matrix eval:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/matrix_eval.json`
- Transfer summary:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.json`
  and
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.md`
- W&B metadata:
  `python/wandb/run-20260522_141504-ud4c09jw/files/wandb-metadata.json`
- Replays:
  `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay`
  and
  `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`
- Viewer command:
  `xushi2-viewer --replay data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay`

Replay analyzer:

- Greedy: Team A issued no fire commands and did no damage; Team B scored
  `37.0`.
- Stochastic: Team A fired continuously but produced only `1000` centi-HP
  damage over five detected episodes, with Team A hit/fire
  `0.0002281022`; Team B scored `37.0`.

The worker attempted the replay dump smoke test after replays existed, but it
timed out after 124 seconds. The replay files and direct replay analyzer output
were verified separately.

## Decision

`NOT_CLEARED`

Objective checks did not pass, so no human replay inspection is required for a
clearance decision. Do not retry this same config unchanged. The direct
multi-enemy-visible scripted teacher succeeds, but the neural PPO run did not
learn objective pressure, firing, or scoring from the widened actor observation
alone.

## Completion Metadata

```json
{
  "changed_files": [],
  "verification": [
    "W&B run metadata and output log verified",
    "checkpoint manifest verified",
    "matrix_eval.json verified",
    "transfer_summary.json verified",
    "greedy and stochastic replay files verified",
    "replay analyzer completed for both final-checkpoint replays"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw",
  "replay_artifacts": [
    "data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay",
    "data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q",
    "py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py -q",
    "py -3.13 -m pytest tests\\smoke\\test_phase_checkpoint_replay_dump_smoke.py -q (timed out after replay artifacts existed)"
  ],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [],
  "blocked_reason": null,
  "residual_risk": [
    "replay dump smoke timeout leaves a test gap, but replay artifacts and analyzer output were verified",
    "neural trainability of the widened actor observation may require a separate supervised or curriculum bridge"
  ],
  "decision": "NOT_CLEARED"
}
```
