# Phase 4 cap_duel v2 focus-fire result

Date: 2026-05-22

## Summary

The config-only cap_duel v2 focus-fire probe was created and run as
`phase4_mappo_cap_duel_v2_focus_fire_v1`. It reused the existing
`team_focus_low_hp` target-conditioned focus-fire machinery and warm-started
from the honest cap_duel v2 checkpoint. No Python or C++ trainer/model code,
sim rules, rewards, observation/action spaces, replay format, or gate
thresholds were changed.

Formal phase-gate status: `NOT_CLEARED`.

## Evidence

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml`
- Git commit: `f776104eb95f64bea44975f0050af29f595f46af` plus dirty working
  tree Phase 4 changes documented by `git status --short`.
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/j90y7l1i
- Run directory: `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/`
- Evidence:
  `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/evidence.json`
- Gate decision:
  `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/gate_decision.json`
- Matrix eval:
  `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/mappo/matrix_eval.json`
- Replays:
  - `data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_stochastic.replay`
- Viewer command:
  `xushi2-viewer --replay data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay`

## Key metrics

- Update 50 eval: `0W/50L/0D`, score `0.00/35.73`, Team A hit/fire
  `0.0000`, Team A same-target fraction `0.804`, Team A target-selection
  entropy `0.374`, recent training `onpt=0.000`.
- Update 100 eval: `0W/50L/0D`, score `0.00/35.93`, Team A hit/fire
  `0.0017`, Team A same-target fraction `0.817`, Team A target-selection
  entropy `0.350`.
- Matrix transfer:
  - noop: `0W/0L/50D`, score `0.00/0.00`.
  - weak_basic_v2: `0W/50L/0D`, score `0.00/35.73`.
  - basic: `0W/50L/0D`, score `0.00/37.00`, Team B kills `30.0`.

The update-50 falsification shape occurred: zero Team A score, zero wins,
recent on-point contact below `0.25`, and Team A hit/fire below `0.02`. The
trainer naturally continued to the configured 100 updates; final metrics still
failed every objective check except the noop no-loss check.

## Verification

- `git status --short` -> dirty worktree with existing Phase 4 changes plus
  this config/result work.
- `git rev-parse HEAD` -> `f776104eb95f64bea44975f0050af29f595f46af`.
- `Get-Process | Where-Object { $_.ProcessName -like '*python*' }` -> no
  relevant training process before the clean relaunch.
- YAML/config sanity check confirmed v2 checkpoint, output dir,
  `target_conditioned_combat: true`, and
  `target_selection_aux_mode: team_focus_low_hp`.
- `py -3.13 -m pytest tests/test_mappo_focus_fire.py
  tests/test_mappo_pretrain_hooks.py tests/test_mappo_matrix_eval.py
  tests/test_phase4_mappo_env.py -q` -> `43 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- `py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q`
  -> `11 passed`.
- `phase_gate.cli` -> `NOT_CLEARED`.

## Operational note

The first worker-launched process wrote W&B run `pmbhalhz` and crashed at
update 30 with `OSError: [Errno 22] Invalid argument` while printing after the
subagent was closed. That partial run produced no eval and is not gate
evidence. Its log was preserved as
`python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/launch_failed_subagent.log`.
The canonical successful run is W&B `j90y7l1i` and the canonical `launch.log`
contains the clean relaunch.

## Decision

Stop this config-only cap_duel v2 focus-fire path. The exact combination of the
honest cap_duel v2 checkpoint and existing focus-fire target conditioning did
not recover full-3v3 combat, scoring, or objective conversion. This completes
the audit-recommended follow-up as a negative result.

Recommended next move: write a new active design plan for a different
structural intervention. Do not retry plain focus-fire, cap-duel v1 focus-fire,
cap-duel v2 focus-fire config-only variants, composition-rehearsal length
variants, or distillation coefficient-only variants.

## Completion metadata

```json
{
  "changed_files": [
    "GOAL_INSTRUCTIONS_MASTER.md",
    "GOAL_INSTRUCTIONS_WORKER.md",
    "experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml",
    "docs/journal/reinforcement_learning_journal.md",
    "docs/plans/archive/2026-05-22-phase4-cap-duel-v2-focus-fire-result.md"
  ],
  "verification": [
    "43 passed focused pytest suite",
    "check_import_boundaries PASS",
    "11 passed replay smoke suite",
    "phase_gate.cli status NOT_CLEARED"
  ],
  "commit": "f776104eb95f64bea44975f0050af29f595f46af",
  "config_path": "experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml",
  "seeds": [3519994490],
  "wandb_run_url": "https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/j90y7l1i",
  "replay_artifacts": [
    "data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay",
    "data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_stochastic.replay"
  ],
  "viewer_command": "xushi2-viewer --replay data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay",
  "tests_run": [
    "py -3.13 -m pytest tests/test_mappo_focus_fire.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_matrix_eval.py tests/test_phase4_mappo_env.py -q",
    "py -3.13 -m scripts.check_import_boundaries",
    "py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q"
  ],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [
    "Added phase4_mappo_cap_duel_v2_focus_fire_v1 config using the honest cap_duel v2 checkpoint plus existing focus-fire target conditioning."
  ],
  "blocked_reason": null,
  "residual_risk": [
    "Only one seed was run.",
    "The worktree was dirty, so this result is tied to commit plus explicit working-tree delta.",
    "The first subagent-launched process crashed before eval and is preserved only as non-gate operational context."
  ]
}
```
