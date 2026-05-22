# Phase 4 Multi-Enemy Supervised Bridge Result

Date: 2026-05-22

## Status

`NOT_REACHED`

## Scope

Implemented one bounded opt-in pre-PPO supervised bridge for the Phase 4
multi-enemy actor-observation probe. No W&B training was launched.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`
- Seed: `3519994490`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes.
- Output:
  `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/`
- W&B: none; this was an implementation/preflight and direct gate assignment.

## Implementation

Added an opt-in `run.multi_enemy_supervised_bridge` path for
`actor_obs: multi_enemy_entity_grid`. It uses the existing actor-visible
`multi_enemy_visible` teacher labels for movement, aim, and fire, then runs a
direct neural-policy gate before PPO.

The bridge is off by default and requires `entity_attention_grid` actor
observations with no action-space target fields. The implementation reuses the
full-env rehearsal pretrain/gate plumbing, adds a no-W&B helper script, and
wires the bridge before composition/BC/PPO in the MAPPO checkpoint path.

No reward, sim-rule, tick-pipeline, action-semantics, replay-format,
phase-gate-threshold, or existing W&B schema changes were made.

## Verification

Master-verified checks:

- `py -3.13 -m scripts.check_import_boundaries`: PASS
- `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_phase4_multi_enemy_actor_obs.py tests/test_mappo_pretrain_hooks.py -q`:
  `29 passed`
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q`:
  `22 passed`
- `git diff --check` on touched bridge files: no whitespace errors.
- Python process check after verification showed only VS Code Jedi language
  server processes, not a training job.

Worker also verified `xushi2_cpp` import and wrote the direct gate artifacts.

## Direct Gate

Artifacts:

- Summary:
  `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/multi_enemy_supervised_bridge_summary.json`
- Gate:
  `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/multi_enemy_supervised_bridge_gate.json`
- Checkpoint:
  `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`

The bridge ran 2000 supervised steps. Labels converged:

- loss `0.0005421519745141268`
- move loss `0.000041060040530283004`
- aim loss `0.00035681703593581915`
- fire loss `0.00014427490532398224`

The pre-PPO neural-policy gate failed:

- Team A visible fire rate `1.0` >= `0.01`
- Team A hit/fire `0.0088888889` < `0.04`
- objective_on_point `0.0116666667` < `0.25`
- mean score A `0.0` < `1.0`
- losses `50` > `49`
- wins `0`
- mean score B `31.9666666667`

## Decision

`NOT_REACHED`

Do not launch PPO or W&B training from this bridge result. The supervised
bridge learned the labels mechanically, but the gated neural policy still did
not produce enough hit/fire, objective pressure, score, or wins in the full
`weak_basic_v2` distribution.

The next move should be an offline failure audit/design decision. Do not retry
this same supervised bridge config unchanged, do not increase bridge length as
the next move, and do not force PPO past the failed pre-PPO gate.

## Completion Metadata

```json
{
  "changed_files": [
    "python/train/full_env_rehearsal.py",
    "python/train/mappo_pretrain_hooks.py",
    "python/train/mappo_eval_checkpoint.py",
    "python/scripts/run_multi_enemy_supervised_bridge_gate.py",
    "python/tests/test_full_env_rehearsal.py",
    "python/tests/test_mappo_pretrain_hooks.py",
    "python/tests/test_phase4_multi_enemy_actor_obs.py",
    "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml"
  ],
  "verification": [
    "import boundary PASS",
    "focused bridge/full-env/pretrain tests passed",
    "direct neural-policy gate artifact verified",
    "checkpoint artifact verified"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": null,
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_phase4_multi_enemy_actor_obs.py tests/test_mappo_pretrain_hooks.py -q",
    "py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q"
  ],
  "behavior_changes": [
    "Added opt-in multi_enemy_visible supervised bridge for actor_obs=multi_enemy_entity_grid",
    "Added local pre-PPO neural-policy gate thresholds for visible firing, hit/fire, objective pressure, score, and losses",
    "Added no-W&B script to run warm-start, supervised bridge, checkpoint, and gate without PPO"
  ],
  "reward_changes": [],
  "config_changes": [
    "Added phase4_mappo_multi_enemy_supervised_bridge_v1.yaml"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "The supervised bridge learned labels but failed the pre-PPO neural-policy gate; PPO should not be launched from this result"
  ],
  "decision": "NOT_REACHED"
}
```
