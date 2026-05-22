# Phase 4 full-env rehearsal result

Date: 2026-05-22

## Summary

The full-env teacher rehearsal implementation and first probe run completed as
`phase4_mappo_full_env_rehearsal_v1`. The supervised rehearsal stage learned
the scripted labels mechanically, but the pre-PPO full-env gate failed, so PPO
was not allowed to start.

Status: `NOT_REACHED`.

## Evidence

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty working
  tree Phase 4 changes.
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu
- Run directory: `python/runs/phase4_mappo_full_env_rehearsal_v1/`
- Pre-PPO gate:
  `python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/full_env_rehearsal_gate.json`
- Checkpoint manifest:
  `python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/checkpoint_manifest.json`
- Replays:
  - `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay`
- Viewer command:
  `xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay`

## Key metrics

The supervised rehearsal loss converged:

- step 1: loss `2.1260`, move `0.1225`, aim `0.1533`, fire `1.3187`,
  target `1.0630`.
- step 2000: loss `0.0012`, move `0.0007`, aim `0.0002`, fire `0.0003`,
  target `0.0001`.

The pre-PPO gate failed:

```json
{
  "status": "NOT_REACHED",
  "metrics": {
    "team_a_hit_fire": 0.0005555555555555556,
    "objective_on_point": 0.014999999999999998,
    "losses": 50.0,
    "wins": 0.0,
    "mean_score_a": 0.0,
    "mean_score_b": 37.0
  },
  "thresholds": {
    "min_team_a_hit_fire": 0.04,
    "min_objective_on_point": 0.25,
    "max_losses": 49.0
  }
}
```

Because the pre-PPO gate returned `NOT_REACHED`, there is intentionally no
post-PPO phase-gate decision for this run.

## Verification

- `py -3.13 -m pytest tests/test_full_env_rehearsal.py
  tests/test_mappo_pretrain_hooks.py -q` -> `9 passed`.
- `py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py
  tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py
  tests/test_phase7_partial_obs.py -q` -> `31 passed`.
- `py -3.13 -m pytest tests/test_full_env_rehearsal.py
  tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py
  tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py
  tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q`
  -> `70 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- `py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q`
  -> `11 passed`.

## Decision

Stop this full-env scripted rehearsal v1 path before PPO. The scripted labels
are learnable, but they do not produce enough full-env hit/fire or objective
contact to justify PPO.

Recommended next move: redesign the rehearsal teacher or gate before another
run. The evidence points to a label-quality/distribution problem, not a
supervised optimizer problem.

## Completion metadata

```json
{
  "changed_files": [
    "GOAL_INSTRUCTIONS_MASTER.md",
    "GOAL_INSTRUCTIONS_WORKER.md",
    "docs/journal/reinforcement_learning_journal.md",
    "docs/plans/README.md",
    "docs/plans/active/2026-05-22-phase4-full-env-teacher-rehearsal-design.md",
    "docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-result.md",
    "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml",
    "python/train/full_env_rehearsal.py",
    "python/train/mappo_eval_checkpoint.py",
    "python/train/mappo_pretrain_hooks.py",
    "python/tests/test_full_env_rehearsal.py",
    "python/tests/test_mappo_pretrain_hooks.py"
  ],
  "verification": [
    "9 passed full-env rehearsal/pretrain-hook tests",
    "31 passed focus/leak-adjacent tests",
    "70 passed broader focused suite",
    "check_import_boundaries PASS",
    "11 passed replay smoke suite",
    "full_env_rehearsal_gate status NOT_REACHED"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu",
  "replay_artifacts": [
    "data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay",
    "data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py -q",
    "py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py -q",
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q"
  ],
  "behavior_changes": [
    "Adds optional full-env scripted rehearsal before PPO when run.full_env_rehearsal.enabled is true",
    "Writes full_env_rehearsal_gate.json and skips PPO when the pre-PPO gate returns NOT_REACHED"
  ],
  "reward_changes": [],
  "config_changes": [
    "Adds phase4_mappo_full_env_rehearsal_v1 probe config"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "Only one seed was run.",
    "The worktree was dirty, so this result is tied to commit plus explicit working-tree delta.",
    "No post-PPO phase gate was run because the pre-PPO rehearsal gate intentionally blocked PPO."
  ]
}
```
