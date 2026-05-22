# Phase 4 Multi-Enemy Actor Observation Preflight Result

Date: 2026-05-22

## Status

Implementation and preflight complete. W&B training was not launched.

## Scope

Implemented one opt-in Phase 4 actor-observation ablation:

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
- Seed: `3519994490`
- Output/artifact directory:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/`
- W&B: none; this was a local implementation and diagnostic assignment only.

The default Phase 4 actor observation remains flat `(3, 31)`. The new path is
activated only with:

```yaml
env:
  actor_obs: multi_enemy_entity_grid
```

## Implementation

Added `Phase4MultiEnemyMappoEnv`, an opt-in wrapper around `Phase4MappoEnv`.
It delegates simulation, reward, action semantics, critic observation, and
episode info to the existing Phase 4 env, and only transforms actor
observations.

Enemy tokens are filled only when the corresponding enemy slot is currently
visible through the native C++ line-of-sight mask. Masked enemy tokens are
zeroed so hidden or non-visible enemy state does not contribute actor-visible
payload values. The wrapper does not use Phase 7 last-seen/stale enemy markers.

Runtime plumbing now lets legacy Phase 4 configs opt into
`multi_enemy_entity_grid` actor observations while preserving the existing
flat default. The model uses the existing `entity_attention_grid` encoder and
the existing action shape remains 6.

The direct diagnostic script now supports a `multi_enemy_visible` teacher that
uses the widened actor-side tokens to walk to the objective and aim/fire at the
nearest currently visible enemy token. The diagnostic emits direct actions
only; it does not train.

## Direct Diagnostic

Command, from `python/`:

```powershell
py -3.13 -m scripts.diagnose_full_env_teacher --config ..\experiments\configs\phase4\probe\phase4_mappo_multi_enemy_actor_obs_v1.yaml --episodes 10 --seed 3519994490 --teacher multi_enemy_visible --output runs\phase4_mappo_multi_enemy_actor_obs_v1\multi_enemy_visible_teacher_diagnostic.json
```

Artifact:
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json`

Result vs `weak_basic_v2`:

- `10W/0L/0D`
- mean score `9.20/0.00`
- Team A hit/fire `0.09`
- visible-fire rate `1.0`
- objective_on_point `0.875`
- mean Team A damage `162000` centi-HP

This is not Phase 4 gate evidence because no neural policy was trained. It is
preflight evidence that the widened actor-visible surface can support a direct
hit-and-hold policy in the full `weak_basic_v2` distribution.

## Verification

- `git rev-parse HEAD`: `f776104eb95f64bea44975f0050af29f595f46af`
- `Get-Process | Where-Object { $_.ProcessName -like '*python*' }`: no Python
  processes were running during preflight.
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py -q`: `7 passed`
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q`: `11 passed`
- `py -3.13 -m scripts.check_import_boundaries`: PASS
- `py -3.13 -m pytest tests/test_phase7_partial_obs.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py -q`: `12 passed`
- `py -3.13 -m pytest tests/test_phase4_mappo_env.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py -q`: `38 passed`
- `.\build\tests\Release\test_actor_leak.exe`: `5 passed`
- `.\build\tests\Release\test_actor_obs.exe`: `12 passed`
- `.\build\tests\Release\test_critic_obs.exe`: `8 passed`
- `.\build\tests\Release\test_obs_dims.exe`: `3 passed`

## Boundaries

No reward, sim-rule, tick-pipeline, action semantics, replay format,
phase-gate threshold, or existing W&B schema changes were made. No W&B
training run was launched.

## Decision

`IMPLEMENTED_PRETRAINING_NOT_RUN`

The implementation preflight is ready for one separate bounded W&B training
assignment using
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`.
That next assignment must collect normal Phase 4 evidence: W&B URL, replay
paths, matrix/gate artifacts, and viewer/human inspection if objective checks
pass.

## Completion Metadata

```json
{
  "changed_files": [
    "python/envs/phase4_multi_enemy_mappo.py",
    "python/envs/runtime_factory.py",
    "python/train/phases.py",
    "python/train/runtime_specs.py",
    "python/xushi2/multi_enemy_obs.py",
    "python/scripts/diagnose_full_env_teacher.py",
    "python/tests/test_phase4_multi_enemy_actor_obs.py",
    "python/tests/test_full_env_teacher_diagnostic.py",
    "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"
  ],
  "verification": [
    "direct diagnostic wrote python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json",
    "hidden/non-visible enemy state leak tests passed",
    "visible enemy mutation tests passed",
    "Team A/Team B frame tests passed",
    "import boundary check passed",
    "C++ actor/critic obs tests passed"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": null,
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py -q",
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_phase7_partial_obs.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py -q",
    "py -3.13 -m pytest tests/test_phase4_mappo_env.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py -q",
    ".\\build\\tests\\Release\\test_actor_leak.exe",
    ".\\build\\tests\\Release\\test_actor_obs.exe",
    ".\\build\\tests\\Release\\test_critic_obs.exe",
    ".\\build\\tests\\Release\\test_obs_dims.exe"
  ],
  "behavior_changes": [
    "new opt-in Phase 4 actor_obs=multi_enemy_entity_grid wrapper emits currently visible enemy tokens"
  ],
  "reward_changes": [],
  "config_changes": [
    "added phase4_mappo_multi_enemy_actor_obs_v1.yaml"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "direct diagnostic is scripted-action evidence only; neural policy trainability remains unproven until a separate W&B run"
  ],
  "decision": "IMPLEMENTED_PRETRAINING_NOT_RUN"
}
```
