# Phase 4 Full-Env Rehearsal V3 cpp_basic Result

Date: 2026-05-22

## Summary

The bounded `cpp_basic` full-env rehearsal probe completed and returned
`NOT_REACHED` at the pre-PPO gate. PPO was intentionally skipped.

This was a negative result, not a blocker. The privileged full-state C++
teacher is competent when run directly, and the supervised labels converged,
but the actor policy did not recover enough contested hit/fire, objective
pressure, score, or wins from the unchanged flat Phase 4 actor observation.

## Identity

- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
  working-tree Phase 4 changes.
- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`
- Seed: `3519994490`
- W&B:
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2`
- Output:
  `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/`

## Run Result

The run warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` and trained the
full-env rehearsal stage for 2000 supervised steps with `teacher: cpp_basic`.
The supervised loss converged from step 1 `loss=0.9507` to step 2000
`loss=0.0264` (`move=0.0050`, `aim=0.0211`, `fire=0.0002`).

The pre-PPO gate artifact is:

```text
python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/mappo/full_env_rehearsal_gate.json
```

Gate status: `NOT_REACHED`.

Metrics:

- Team A hit/fire: `0.0061111111 < 0.04`
- Objective on-point: `0.0166666667 < 0.25`
- Losses: `50 > 49`
- Wins: `0`
- Mean score: `0.00/37.00`

Because the pre-PPO gate failed, PPO was not forced and no post-PPO
`gate_decision.json` was produced.

## Matrix Eval

Matrix eval was still written for `ckpt_final.pt`:

```text
python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/mappo/matrix_eval.json
```

- vs `noop`: `0W/0L/50D`, score `0.00/0.00`.
- vs `weak_basic_v2`: `0W/50L/0D`, score `0.00/37.00`.
- vs `basic`: `0W/50L/0D`, score `0.00/37.00`, Team B kills `30.0`.

## Replays

Replays were dumped:

- `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay`
- `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay`

Viewer command:

```powershell
xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay
```

Replay analysis artifacts:

- `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/replay_analysis_greedy.json`
- `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/replay_analysis_stochastic.json`

Greedy replay: Team B won `37.00/0.00`; Team A produced `11` damage hits,
`11000` centi-HP damage, `0` kills, hit/fire `0.0061111111`, and mean
nearest-visible aim error `2.0652` rad.

Stochastic replay: Team B won `35.10/0.00`; Team A produced `21` damage hits,
`21000` centi-HP damage, `1` kill, hit/fire `0.0116731518`, and mean
nearest-visible aim error `2.0855` rad.

## Verification

- Focused v3 preflight tests:
  `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q`
  -> `16 passed`.
- Import boundary:
  `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- Direct v3 teacher diagnostic:
  `cpp_basic` vs `weak_basic_v2`, `10W/0L/0D`, score `12.70/0.00`, Team A
  hit/fire `0.0917`, objective_on_point `0.8667`.
- Broader focused suite:
  `76 passed`.
- Replay dump smoke:
  `py -3.13 -m pytest tests\smoke\test_phase_checkpoint_replay_dump_smoke.py -q`
  -> `11 passed`.

## Decision

Decision: `NOT_REACHED`.

Stop this v3 privileged full-env rehearsal path before PPO. Do not retry by
only increasing rehearsal length, weakening the pre-PPO gate, or forcing PPO.

The result points back to the information surface: `cpp_basic` can produce
competent full-env behavior when run directly, but the unchanged flat actor
observation and policy do not retain that behavior after imitation. The next
work should be an explicit design decision about actor observation capacity or
an offline audit proving a different non-observation-changing teacher can be
represented by the current actor input.

## Completion Metadata

```json
{
  "changed_files": [
    "python/train/full_env_rehearsal.py",
    "python/scripts/diagnose_full_env_teacher.py",
    "python/tests/test_full_env_rehearsal.py",
    "python/tests/test_full_env_teacher_diagnostic.py",
    "python/tests/test_mappo_pretrain_hooks.py",
    "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml",
    "docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md"
  ],
  "verification": [
    "focused v3 tests: 16 passed",
    "broader focused suite: 76 passed",
    "import boundary: PASS",
    "direct cpp_basic teacher diagnostic: 10W/0L/0D",
    "replay dump smoke: 11 passed"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af plus dirty working-tree Phase 4 changes",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2",
  "replay_artifacts": [
    "data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay",
    "data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/test_full_env_teacher_diagnostic.py tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q",
    "py -3.13 -m pytest tests\\smoke\\test_phase_checkpoint_replay_dump_smoke.py -q"
  ],
  "behavior_changes": [
    "Added opt-in training-time cpp_basic teacher labels for full-env rehearsal only; actor inference remains unchanged."
  ],
  "reward_changes": [],
  "config_changes": [
    "Added v3 probe config with full_env_rehearsal.teacher=cpp_basic and target_selection_dim=0."
  ],
  "blocked_reason": null,
  "residual_risk": [
    "V3 used privileged training-time labels and did not change actor observation capacity.",
    "Dirty working-tree changes mean this result should be cited as commit plus explicit working-tree delta unless committed."
  ],
  "decision": "NOT_REACHED"
}
```
