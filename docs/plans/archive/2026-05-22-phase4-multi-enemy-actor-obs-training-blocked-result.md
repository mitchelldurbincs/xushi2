# Phase 4 Multi-Enemy Actor Observation Training Blocked Result

Date: 2026-05-22

## Status

`BLOCKED`: the assigned W&B training run started but crashed before usable
metrics.

## Scope

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
- Seed: `3519994490`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes.
- W&B:
  https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/bw9jsxte
- Output directory:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/`

## Result

Preflight passed:

- `py -3.13 -m scripts.check_import_boundaries`: PASS
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q`:
  `11 passed`

The training command launched from `python/`:

```powershell
py -3.13 -m train.train --config ..\experiments\configs\phase4\probe\phase4_mappo_multi_enemy_actor_obs_v1.yaml
```

W&B authenticated and created the run. The process then failed during
`run.init_from_checkpoint` before any eval/gate/matrix metrics were emitted.
The configured checkpoint,
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, is a flat-actor model.
Strict loading into the new `entity_attention_grid` actor topology failed with
missing `actor_entity_encoder`, `actor_grid_encoder`, and `actor_fusion` keys,
plus unexpected `actor_embed` keys.

Verified artifacts:

- W&B metadata:
  `python/wandb/run-20260522_135640-bw9jsxte/files/wandb-metadata.json`
- W&B output log:
  `python/wandb/run-20260522_135640-bw9jsxte/files/output.log`
- W&B summary contains only runtime fields; no training metrics were produced.
- No checkpoint manifest, gate artifact, matrix eval, or replay artifacts were
  produced by this run.

The only file in the run output directory remains the prior direct diagnostic:
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json`.

## Decision

Do not retry the same config unchanged. It will fail at the same strict
warm-start load before producing evidence.

The next bounded assignment should fix the warm-start topology mismatch in an
opt-in way for this actor-observation migration. It must preserve strict
default warm-start behavior for ordinary same-topology runs and must not change
rewards, sim rules, action semantics, replay format, phase-gate thresholds, or
existing W&B metric schema.

## Completion Metadata

```json
{
  "changed_files": [],
  "verification": [
    "worker preflight passed import boundary and focused tests",
    "W&B run metadata verified under python/wandb/run-20260522_135640-bw9jsxte",
    "output.log verified strict warm-start state_dict mismatch",
    "no checkpoint, gate, matrix, replay, or training metrics were produced"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/bw9jsxte",
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q"
  ],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [],
  "blocked_reason": "Strict warm-start from flat actor checkpoint into multi_enemy_entity_grid actor model failed before usable metrics due incompatible actor encoder state_dict keys.",
  "residual_risk": [
    "neural policy trainability for the multi-enemy actor-observation ablation remains untested"
  ],
  "decision": "BLOCKED"
}
```
