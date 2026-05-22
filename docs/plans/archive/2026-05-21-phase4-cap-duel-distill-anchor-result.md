# Phase 4 cap_duel distillation anchor result

Date: 2026-05-21

## Summary

The PPO-time cap_duel v2 distillation anchor was implemented, tested, and run as
`phase4_mappo_cap_duel_distill_anchor_v1`. The hook was active during PPO and
logged finite `distill/*` metrics, but the run hit the configured early-stop
condition at update 50: Team A hit/fire remained below the `0.04` floor and Team
A score stayed at zero.

Formal phase-gate status: `NOT_CLEARED`.

## Evidence

- Config: `experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af`
- Seed: `1779134702`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/c8yyqsdd
- Run directory: `python/runs/phase4_mappo_cap_duel_distill_anchor_v1/`
- Evidence: `python/runs/phase4_mappo_cap_duel_distill_anchor_v1/evidence.json`
- Gate decision: `python/runs/phase4_mappo_cap_duel_distill_anchor_v1/gate_decision.json`
- Early stop: `python/runs/phase4_mappo_cap_duel_distill_anchor_v1/mappo/early_stop_decision.json`
- Replays:
  - `data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_stochastic.replay`
- Viewer command: `xushi2-viewer --replay data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay`

## Key metrics

- Update 25 eval: `0W/50L/0D`, score `0.00/23.33`, Team A hit/fire `0.0165`.
- Update 50 eval: `0W/50L/0D`, score `0.00/24.27`, Team A hit/fire `0.0168918919`.
- Early-stop rule: `team_a_hit_fire=0.0168918919 < 0.04` and `mean_score_a=0.0 <= 0.0`.
- Distillation final W&B summary:
  - `distill/loss=1.2836`
  - `distill/aim_loss=0.57077`
  - `distill/fire_loss=0.71284`
  - `distill/scaled_loss=0.06418`
  - `distill/active_samples=256`
  - `distill/teacher_fire_prob=0.80877`
  - `distill/student_fire_prob=0.48559`
  - `distill/fire_agreement=0.19141`
- Matrix transfer:
  - noop: `0W/0L/50D`, score `0.00/0.00`
  - weak_basic_v2: `0W/50L/0D`, score `0.00/23.33`
  - basic: `0W/50L/0D`, score `0.00/37.00`

## Verification

- `py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py tests/test_cap_duel_distill.py -q` -> `81 passed`
- `py -3.13 -m scripts.check_import_boundaries` -> pass
- `git diff --check` -> pass
- `py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q` -> `11 passed`
- `py -3.13 -m train.phase_gate.cli ...` -> `NOT_CLEARED`

## Decision

Stop this distillation-anchor path. The implementation is mechanically useful:
it proves a frozen cap_duel teacher can be sampled during PPO and logged through
new metrics without changing sim, reward, observation, action, replay, or gate
semantics. The result falsifies the current anchor hypothesis for Phase 4
transfer because it did not recover full-3v3 combat accuracy or scoring.

Recommended next move: Strategy 3 focus-fire target conditioning. Do not add a
new actor head or action-space-facing target machinery without explicit user
approval.

## Completion metadata

```json
{
  "changed_files": [
    "GOAL_INSTRUCTIONS.md",
    "experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml",
    "python/train/cap_duel_distill.py",
    "python/train/mappo_rollout_trainer.py",
    "python/train/mappo_eval_checkpoint.py",
    "python/train/mappo_training_hooks.py",
    "python/train/train.py",
    "python/tests/test_cap_duel_distill.py",
    "docs/journal/reinforcement_learning_journal.md",
    "docs/plans/archive/2026-05-21-phase4-cap-duel-distill-anchor-result.md"
  ],
  "verification": [
    "81 passed focused pytest suite",
    "check_import_boundaries PASS",
    "git diff --check PASS",
    "11 passed replay smoke suite",
    "phase_gate.cli status NOT_CLEARED"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml",
  "seeds": [1779134702],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/c8yyqsdd",
  "replay_artifacts": [
    "data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay",
    "data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py tests/test_cap_duel_distill.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "git diff --check",
    "py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q"
  ],
  "behavior_changes": [
    "Adds optional PPO-time cap_duel teacher distillation hook when run.cap_duel_distill.enabled is true",
    "Adds new distill/* training metrics",
    "Adds optional cap_duel_distill_early_stop rule for this probe config"
  ],
  "reward_changes": [],
  "config_changes": [
    "Adds phase4_mappo_cap_duel_distill_anchor_v1 probe config"
  ],
  "blocked_reason": null,
  "residual_risk": [
    "Only one seed was run",
    "The anchor did not improve full-3v3 hit/fire or scoring",
    "Human replay inspection is not needed because objective gate checks failed"
  ]
}
```
