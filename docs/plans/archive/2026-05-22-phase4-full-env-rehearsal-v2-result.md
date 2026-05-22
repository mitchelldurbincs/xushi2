# Phase 4 full-env rehearsal v2 result

Date: 2026-05-22

## Summary

The v1 failure audit found a concrete teacher bug: actor aim vectors are
documented as `(sin theta, cos theta)`, but the full-env rehearsal teacher used
`atan2(y, x)` for target angle. V2 fixed the target angle convention, added a
regression test, and ran one corrected probe.

Status: `NOT_REACHED`.

## Evidence

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty working
  tree Phase 4 changes.
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr
- Run directory: `python/runs/phase4_mappo_full_env_rehearsal_v2/`
- Pre-PPO gate:
  `python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/full_env_rehearsal_gate.json`
- Checkpoint manifest:
  `python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/checkpoint_manifest.json`
- Replays:
  - `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay`
  - `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`
- Viewer command:
  `xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay`

## Key metrics

The corrected supervised rehearsal still converged:

- step 1: loss `2.2337`, move `0.1387`, aim `0.2181`, fire `1.3455`,
  target `1.0629`.
- step 2000: loss `0.0012`, move `0.0007`, aim `0.0003`, fire `0.0002`,
  target `0.0001`.

The pre-PPO gate failed:

```json
{
  "status": "NOT_REACHED",
  "metrics": {
    "team_a_hit_fire": 0.0,
    "objective_on_point": 0.01,
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

Because the pre-PPO gate returned `NOT_REACHED`, PPO was intentionally skipped
and there is no post-PPO phase-gate decision.

Replay analysis shows the fix moved the failure:

- v1 greedy Team A mean nearest-visible aim error: `2.5406` rad, hit/fire
  `0.0005556`.
- v2 greedy Team A mean nearest-visible aim error: `0.6073` rad, hit/fire
  `0.0`.
- v2 stochastic Team A mean nearest-visible aim error: `0.6376` rad, hit/fire
  `0.0`.

Team A still fired continuously at visible enemies and still produced
`0` damage in the v2 greedy/stochastic replays, so the remaining gap is not the
axis convention alone.

## Verification

- `py -3.13 -m pytest tests/test_full_env_rehearsal.py
  tests/test_mappo_pretrain_hooks.py -q` -> `10 passed`.
- `py -3.13 -m pytest tests/test_full_env_rehearsal.py
  tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py
  tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py
  tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q`
  -> `71 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- `py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q`
  -> `11 passed`.
- `py -3.13 -m scripts.analyze_replay_combat --replay
  ..\data\replays\phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay`
  completed.
- `py -3.13 -m scripts.analyze_replay_combat --replay
  ..\data\replays\phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`
  completed.

## Decision

Stop the corrected full-env scripted rehearsal v2 path before PPO. The audit
found and fixed a real aim-label bug, and the replay analyzer confirms aim
error improved substantially, but the pre-PPO behavior still cannot score,
hold point, or land shots against `weak_basic_v2`.

Recommended next move: do not extend v2 length or force PPO. The next design
needs a higher-fidelity full-env teacher that accounts for shooting geometry
and objective timing, or a separate supervised diagnostic proving that a
scripted action stream can hit and hold in the same full-env distribution
before training a neural policy against those labels.

## Completion metadata

```json
{
  "changed_files": [
    "docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v2-result.md",
    "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml",
    "python/train/full_env_rehearsal.py",
    "python/tests/test_full_env_rehearsal.py"
  ],
  "verification": [
    "10 passed full-env rehearsal/pretrain-hook tests",
    "71 passed broader focused suite",
    "check_import_boundaries PASS",
    "11 passed replay smoke suite",
    "full_env_rehearsal_gate status NOT_REACHED",
    "replay analyzer completed for greedy and stochastic v2 replays"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr",
  "replay_artifacts": [
    "data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay",
    "data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py -q",
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q",
    "py -3.13 -m scripts.analyze_replay_combat --replay ..\\data\\replays\\phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay",
    "py -3.13 -m scripts.analyze_replay_combat --replay ..\\data\\replays\\phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay"
  ],
  "behavior_changes": [
    "Corrects full-env rehearsal scripted aim labels to use the documented actor aim convention"
  ],
  "reward_changes": [],
  "config_changes": [
    "Adds phase4_mappo_full_env_rehearsal_v2 probe config"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "Only one seed was run.",
    "The worktree was dirty, so this result is tied to commit plus explicit working-tree delta.",
    "No post-PPO phase gate was run because the pre-PPO rehearsal gate intentionally blocked PPO."
  ]
}
```
