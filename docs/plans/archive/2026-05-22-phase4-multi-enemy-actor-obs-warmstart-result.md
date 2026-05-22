# Phase 4 Multi-Enemy Actor Observation Warm-Start Result

Date: 2026-05-22

## Status

`IMPLEMENTED_PRETRAINING_NOT_RUN`

## Scope

Implemented a narrow opt-in warm-start migration for the Phase 4
multi-enemy actor-observation probe.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
- Seed: `3519994490`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes.
- W&B: none; no training launched in this assignment.

## Implementation

Added `run.warm_start_migration: compatible_exact`.

The default warm-start path remains strict. The migration path is explicit and
loads only same-name, same-shape tensors from the checkpoint. It reports
missing model keys, unexpected checkpoint keys, and same-name shape mismatches.
The multi-enemy probe config opts into this mode.

No reward, sim-rule, tick-pipeline, action-semantics, replay-format,
phase-gate-threshold, or existing W&B schema changes were made.

## Verification

Worker and master-side verification passed:

- `py -3.13 -m scripts.check_import_boundaries`: PASS
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q`:
  `11 passed`
- `py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py -q`:
  `8 passed`
- Non-training config/model smoke loaded
  `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` into the
  `entity_attention_grid` model without error.

Smoke result:

- `actor_obs=multi_enemy_entity_grid`
- `obs_dim=3167`
- `action_dim=6`
- `target_selection_dim=0`
- `warm_start_migration=compatible_exact`
- loaded compatible tensors: `17`
- skipped unexpected flat actor keys:
  `actor_embed.0.weight`, `actor_embed.0.bias`
- missing new actor topology keys included `actor_entity_encoder`,
  `actor_grid_encoder`, and `actor_fusion`

## Decision

Ready for one separate bounded W&B training assignment using the same config.
Do not treat this implementation smoke as phase-gate evidence.

## Completion Metadata

```json
{
  "changed_files": [
    "python/train/mappo_pretrain_hooks.py",
    "python/tests/test_mappo_pretrain_hooks.py",
    "python/tests/test_phase4_multi_enemy_actor_obs.py",
    "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"
  ],
  "verification": [
    "import boundary PASS",
    "multi-enemy actor-observation and diagnostic tests passed",
    "warm-start hook tests passed",
    "config/model smoke loaded compatible_exact warm-start without error"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": null,
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q",
    "py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py -q"
  ],
  "behavior_changes": [
    "Added opt-in run.warm_start_migration=compatible_exact for checkpoint topology migration"
  ],
  "reward_changes": [],
  "config_changes": [
    "phase4_mappo_multi_enemy_actor_obs_v1.yaml now opts into warm_start_migration: compatible_exact"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "neural policy trainability remains untested until the separate W&B run"
  ],
  "decision": "IMPLEMENTED_PRETRAINING_NOT_RUN"
}
```
