# Phase 4 Multi-Enemy Closed-Loop Supervised Bridge Result

Date: 2026-05-22

## Status

`NOT_REACHED`. Stop before PPO and W&B training.

## Identity

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`
- Seed: `3519994490`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes
- W&B: none; this was a local no-W&B implementation/preflight and pre-PPO gate
  assignment
- Output:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/`

## Implementation

The worker added an opt-in closed-loop path for
`run.multi_enemy_supervised_bridge.closed_loop.enabled`. The bridge rolls out
the current neural policy against `weak_basic_v2`, queries the existing
`multi_enemy_visible` teacher labels on those policy-induced states, and runs
bounded supervised update rounds. It also writes a policy-state agreement
artifact for movement, aim, and fire.

The implementation kept the feature opt-in/off-by-default and did not launch
PPO or W&B. No reward, sim-rule, tick-pipeline, action semantics,
action-space-facing target field, replay format, phase-gate threshold, or
existing W&B metric/schema changes were made.

Changed implementation files reported by the worker:

- `python/train/full_env_rehearsal.py`
- `python/train/mappo_pretrain_hooks.py`
- `python/scripts/run_multi_enemy_supervised_bridge_gate.py`
- `python/tests/test_mappo_pretrain_hooks.py`
- `python/tests/test_phase4_multi_enemy_actor_obs.py`
- `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`

## Verification

Master verification:

- `py -3.13 -m scripts.check_import_boundaries` passed.
- `py -3.13 -m pytest tests/test_phase4_multi_enemy_actor_obs.py tests/test_mappo_pretrain_hooks.py -q`
  passed: `20 passed`.
- `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py -q`
  passed: `22 passed`.
- `git diff --check` on the touched bridge/instruction files passed, with only
  LF/CRLF warnings.
- No relevant Python training process was running after verification.

Worker-reported local gate command:

```powershell
py -3.13 -m scripts.run_multi_enemy_supervised_bridge_gate --config ..\experiments\configs\phase4\probe\phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml --output runs\phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1\multi_enemy_closed_loop_supervised_bridge_summary.json
```

## Artifacts

- Summary:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/multi_enemy_closed_loop_supervised_bridge_summary.json`
- Gate:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`
- Agreement:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`
- Checkpoint:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`

No replay artifacts were produced.

## Metrics

Policy-state agreement at the final round:

- movement MSE: `0.0154950926`
- aim absolute error: `0.2026730925`
- fire accuracy: `1.0`
- fire positive recall: `1.0`
- policy fire rate: `1.0`
- teacher fire rate: `1.0`

Pre-PPO neural-policy gate:

- status: `NOT_REACHED`
- Team A visible-fire rate: `1.0 >= 0.01`
- Team A hit/fire: `0.0427777778 >= 0.04`
- objective_on_point: `0.29 >= 0.25`
- mean score A: `0.0 < 1.0`
- wins: `0`
- losses: `0 <= 49`
- mean score B: `0.0`

The closed-loop bridge improved the previous one-shot bridge result by
recovering hit/fire and objective pressure, but it still did not convert any
score. Because the configured pre-PPO gate requires nonzero score, the gate
correctly stopped before PPO.

## Decision

Do not launch PPO or W&B training from
`phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`. Do not retry
the same closed-loop config unchanged and do not force PPO past the failed
score check.

The next move should be an offline audit of the zero-score draw behavior: why
the policy now fires, hits, and occupies the point enough to pass those floors
but still produces `0.00/0.00` scoring. That audit should inspect the gate
artifact, config, and available policy behavior before proposing any next
implementation or W&B assignment.

