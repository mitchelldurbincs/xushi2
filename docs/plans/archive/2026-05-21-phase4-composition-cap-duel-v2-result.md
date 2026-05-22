# Phase 4 Composition Rehearsal With Cap-Duel v2 Result - 2026-05-21

## Scope

Stage 1 from `GOAL_INSTRUCTIONS.md`: use the honest cap_duel v2 checkpoint as
the combat teacher for composition rehearsal, then allow full Phase 4 3v3 PPO
against `weak_basic_v2` only if the post-composition full-env hit/fire
kill-switch clears.

No C++ files, sim rules, reward formulas, observation/action spaces, replay
format, W&B metric schema, or MAPPO core logic were changed. No commit was
created.

## Configs

- `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml`
  - 2000 composition rehearsal steps.
  - Student warm-start and objective teacher:
    `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`.
  - Combat teacher:
    `runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt`.
  - `composition_combat_env.mini_game: cap_duel` with the cap_duel v2
    `mini_game_config` block mirrored field-for-field.
  - Full PPO was configured for 200 updates but did not run because the
    pre-PPO kill-switch fired.
- `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml`
  - The single allowed fallback: `composition_pretrain_steps: 4000`.
  - Separate output directory so iteration 1 evidence remains intact.
  - Same teacher checkpoints, mini-game config, PPO knobs, and phase-gate bar.

## Verification

- Pre-launch focused/full suite after initial config:
  `py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py -q`
  -> `87 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- After adding the fallback config, the same suite -> `88 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- The configured post-run command path
  `tests/test_phase4_checkpoint_replay_dump.py` does not exist in this
  checkout. Current equivalent:
  `py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q`
  -> `11 passed`.

## Iteration 1 - 2000 Steps

- Config:
  `../experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus working-tree
  config/test additions.
- Seed: `1779134702`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ao4hy6fa
- Output:
  `python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2/`
- Status: NOT_REACHED, pre-PPO kill-switch.
- Final composition loss: `0.0782`.
- Post-BC diagnostics:
  - objective on-point `0.131 < 0.250`
  - objective losses `50 > 0`
  - combat kills `4.26 < 12.00`
  - full Team A hit/fire `0.0047 < 0.0400`
  - full Team A aim error `1.387 < 1.550`
- Matrix artifact:
  `python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2/mappo/matrix_eval.json`
  - vs `noop`: 50 draws, Team A score `0.00`
  - vs `weak_basic_v2`: 50 losses, Team A score `0.00`
  - vs `basic`: 50 losses, Team A score `0.00`
- Phase gate: not invoked; PPO was skipped before any final gate evidence or
  subjective replay could be produced.

## Iteration 2 - 4000-Step Fallback

- Config:
  `../experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus working-tree
  config/test/doc additions.
- Seed: `1779134702`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ssmkgzua
- Output:
  `python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2_4000/`
- Status: NOT_REACHED, pre-PPO kill-switch.
- Final composition loss: `0.0550`.
- Post-BC diagnostics:
  - objective on-point `0.203 < 0.250`
  - objective losses `50 > 0`
  - combat kills `3.04 < 12.00`
  - full Team A hit/fire `0.0116 < 0.0400`
  - full Team A aim error `1.555 > 1.550`
- Matrix artifact:
  `python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2_4000/mappo/matrix_eval.json`
  - vs `noop`: 50 draws, Team A score `0.00`, Team A kills `1.0`
  - vs `weak_basic_v2`: 50 losses, Team A score `0.00`
  - vs `basic`: 50 losses, Team A score `0.00`
- Phase gate: not invoked; PPO was skipped before any final gate evidence or
  subjective replay could be produced.

## Decision

Stop condition reached. The required 2000-step run and its single allowed
4000-step fallback both failed the pre-PPO hit/fire kill-switch, so Stage 1 did
not reach full PPO, replay dumping, or the Phase 4 anchor-transfer gate.

This is a completed negative result, not a blocker. The cap_duel v2 teacher is
still valid in isolation, but the current composition-rehearsal loss does not
bind that skill into the full 3v3 weak_basic_v2 distribution.

## Hand-Off

Do not spend another config-only run on rehearsal length under the current
rules. The next useful decision is code-scope:

- Add a distillation anchor during PPO so cap_duel combat cannot be erased by
  the full-env gradient.
- Or escalate to Strategy 3 focus-fire target conditioning from the May 18
  proposal.

Both are outside this goal's authorized scope and need explicit user approval.

## Completion Metadata

```json
{
  "changed_files": [
    "experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml",
    "experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml",
    "python/tests/test_mappo_composition_rehearsal.py",
    "docs/journal/reinforcement_learning_journal.md",
    "docs/plans/archive/2026-05-21-phase4-composition-cap-duel-v2-result.md"
  ],
  "verification": [
    "py -3.13 -m pytest tests/test_mappo_composition_rehearsal.py -q -> 7 passed",
    "py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py -q -> 88 passed",
    "py -3.13 -m scripts.check_import_boundaries -> PASS",
    "py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q -> 11 passed"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml",
  "seeds": [1779134702],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ssmkgzua",
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [
    "tests/test_mappo_composition_rehearsal.py",
    "tests/test_phase4_cap_duel_mappo.py",
    "tests/test_phase4_combat_1v1_mappo.py",
    "tests/test_phase4_mappo_env.py",
    "tests/test_phase4_current_selfplay.py",
    "tests/test_mappo_matrix_eval.py",
    "tests/test_mappo_pretrain_hooks.py",
    "tests/test_mappo_team_spirit_ramp.py",
    "tests/smoke/test_phase_checkpoint_replay_dump_smoke.py",
    "scripts.check_import_boundaries"
  ],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [
    "Added cap_duel v2 composition rehearsal config with 2000 steps.",
    "Added the single allowed 4000-step fallback config after the pre-PPO kill-switch fired."
  ],
  "blocked_reason": null,
  "residual_risk": [
    "No subjective replay inspection was possible because PPO was skipped before producing meaningful full-3v3 replay artifacts.",
    "The configured replay-dump test path in GOAL_INSTRUCTIONS.md is stale in this checkout; the current smoke replay-dump suite passed instead."
  ]
}
```
