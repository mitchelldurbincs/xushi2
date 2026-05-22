# Goal Instructions: Phase 4 Worker

Date: 2026-05-22

## Role

This file is for the `codex --yolo` worker. You are executing bounded tasks
assigned by the master Codex. The shared high-level goal is clearing Phase 4,
but this file by itself does not authorize arbitrary code changes or arbitrary
training runs.

If launched without a specific assignment from the master, do an evidence audit
only. Do not start a training run.

## Non-Negotiables

- The algorithm is MAPPO, not MAPO.
- Run only one experiment at a time.
- Do not launch a new training run while another relevant Python training
  process is in flight.
- Do not weaken phase gates.
- Do not fabricate W&B URLs, replay paths, seeds, metrics, or commit identity.
- Do not revert user/master changes.
- Do not change C++ sim rules, reward formulas, observation layout, action
  semantics, replay format, determinism behavior, or W&B existing metric schema
  unless the master assignment explicitly authorizes it.
- Do not add a new actor head, action-space-facing target field, or new
  load-bearing Phase 4 machinery unless the assignment explicitly authorizes it.
- If docs and code disagree on load-bearing behavior, stop and report the
  disagreement.

## Required First Reads

Before doing assigned work:

1. `GOAL_INSTRUCTIONS_MASTER.md`.
2. `docs/plans/README.md`.
3. Latest entries of `docs/journal/reinforcement_learning_journal.md`.
4. Relevant result docs in `docs/plans/archive/`.
5. Relevant config(s) in `experiments/configs/phase4/probe/`.
6. Relevant code/tests for the assigned file scope.

## Current State To Respect

Phase 4 is still not cleared.

Recent completed evidence:

- Cap-duel v2 solved the isolated combat/objective mini-game honestly.
  Config:
  `experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v2.yaml`.
  Seed `3519994490`.
  W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/l9890sl6`.
- Composition rehearsal using cap-duel v2 failed before PPO at both 2000 and
  4000 steps. Do not run another rehearsal-length variant.
- PPO-time cap-duel v2 distillation anchor ran and failed the unchanged gate.
  Config:
  `experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml`.
  Seed `1779134702`.
  W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/c8yyqsdd`.
  Status `NOT_CLEARED`; update-50 `team_a_hit_fire=0.0168918919 < 0.04`,
  `mean_score_a=0.0`.
- Existing focus-fire target-conditioning code already exists and already has
  prior results:
  - `phase4_mappo_focus_fire_v1` fixed target concentration but produced
    no score or wins.
  - `phase4_mappo_cap_duel_focus_fire_v1` using the older cap-duel v1
    checkpoint collapsed by update 50.
  Do not reimplement this machinery from scratch.
- Full-env rehearsal v1 failed before PPO with `NOT_REACHED`.
- Full-env rehearsal v2 fixed a real scripted-teacher aim convention bug
  (`atan2(x, y)` for actor `(sin, cos)` vectors) and reduced replay aim error,
  but still failed before PPO with `team_a_hit_fire=0.0`,
  `objective_on_point=0.01`, `50/50` losses, and score `0.00/37.00`.
- Direct teacher diagnostic found that the actor-observation-only teacher
  itself fails under contest: vs `weak_basic_v2`, `0W/0L/50D`, score
  `0.00/0.00`, Team A hit/fire `0.0`, objective_on_point `0.0`. The same
  teacher beats `noop` `10/10`, so movement/hold is present. Full-state
  `cpp_basic` beats `weak_basic_v2` `10/10`, score `12.70/0.00`,
  hit/fire `0.0917`, objective_on_point `0.8667`.
- V3 cpp_basic rehearsal preflight is implemented and focused tests passed.
  Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`.
- V3 cpp_basic rehearsal also ran and failed before PPO with `NOT_REACHED`.
  W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2`.
  Gate metrics: Team A hit/fire `0.0061111111`, objective_on_point
  `0.0166666667`, `50/50` losses, `0` wins, score `0.00/37.00`.
  Result doc:
  `docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md`.
- The bounded opt-in multi-enemy actor-observation implementation/preflight
  completed without launching W&B training. Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`.
  Direct `multi_enemy_visible` diagnostic vs `weak_basic_v2` produced
  `10W/0L/0D`, score `9.20/0.00`, Team A hit/fire `0.09`,
  objective_on_point `0.875`. Artifact:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json`.
  Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`.
- The first W&B training attempt for that config is blocked. W&B run
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/bw9jsxte` was
  created, but training crashed before usable metrics because the configured
  flat-actor checkpoint could not load strictly into the new
  `entity_attention_grid` actor topology. Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-blocked-result.md`.
  Do not retry the same config unchanged.
- The opt-in warm-start migration fix completed. The config now has
  `run.warm_start_migration: compatible_exact`; default warm-start remains
  strict. Import boundary, focused multi-enemy/diagnostic tests, warm-start
  tests, and a non-training config/model smoke passed. Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-warmstart-result.md`.
- The bounded W&B training run completed and returned `NOT_CLEARED`.
  W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw`.
  Best/final eval were `0W/50L/0D`, score `0.00/37.00`, Team A hit/fire
  `0.0`. Matrix eval was negative and replay analysis found greedy Team A did
  not fire. Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`.
  Do not retry this same config unchanged.
- User approval was received in-thread to proceed with one bounded opt-in
  Phase 4 multi-enemy supervised bridge using existing `multi_enemy_visible`
  teacher labels before PPO. This approval is narrow and does not authorize
  reward, sim, action, replay, phase-gate, existing W&B schema, or automatic
  W&B training changes.
- The bounded opt-in multi-enemy supervised bridge completed and returned
  `NOT_REACHED`. Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`.
  It learned labels mechanically, but the pre-PPO neural-policy gate failed:
  Team A hit/fire `0.0088888889`, objective_on_point `0.0116666667`,
  score `0.0`, `50/50` losses. Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-supervised-bridge-result.md`.
  Do not launch PPO from this result.
- The supervised bridge failure audit completed. Audit doc:
  `docs/plans/active/2026-05-22-phase4-multi-enemy-supervised-bridge-failure-audit.md`.
  It concluded the one-shot bridge learned labels on expert-style states but
  failed under closed-loop policy-state distribution shift. The proposed next
  implementation is a bounded opt-in closed-loop supervised bridge using
  policy-induced states and existing `multi_enemy_visible` labels. This is not
  authorized until the user explicitly approves it.
- User approval was received in-thread on 2026-05-22 for one bounded opt-in
  Phase 4 closed-loop supervised bridge using policy-induced states plus
  existing `multi_enemy_visible` teacher labels. This approval is narrow and
  does not authorize reward, sim, tick-pipeline, action semantics,
  action-space field, replay format, phase-gate threshold, existing W&B
  metric/schema, unbounded training, or automatic W&B/PPO changes.
- The bounded opt-in closed-loop supervised bridge completed and returned
  `NOT_REACHED`. Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`.
  It improved pre-PPO gate metrics to Team A hit/fire `0.0427777778`,
  objective_on_point `0.29`, and `0` losses, but still produced mean score A
  `0.0` and `0` wins. Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-closed-loop-supervised-bridge-result.md`.
  Do not launch PPO or W&B from this result.

## Completed Assignment: Strategy 3 Audit

The 2026-05-22 audit was accepted by the master. It found:

- `phase4_mappo_focus_fire_v1` already implemented the May 18 Strategy 3 core:
  `team_focus_low_hp` labels, no-target class, target-selection auxiliary loss,
  target-conditioned combat, and focus metrics. It fixed measured target
  concentration but still produced `0.00/0.00` score.
- `phase4_mappo_cap_duel_focus_fire_v1` used the older cap-duel v1 checkpoint
  (`runs/phase4_mappo_cap_duel_v1/mappo/ckpt_0075.pt`) and collapsed at update
  50. It does not directly falsify an honest cap-duel v2 focus-fire follow-up.
- Current checkout already contains the needed machinery for one config-only
  cap-duel v2 plus existing focus-fire probe.

## Completed Assignment: cap_duel v2 Focus-Fire Probe

The config-only cap_duel v2 focus-fire probe completed on 2026-05-22.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml`
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/j90y7l1i
- Output: `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/`
- Gate: `NOT_CLEARED`
- Final eval: `0W/50L/0D`, score `0.00/35.93`, Team A hit/fire `0.0017`.
- Matrix: draw-only vs `noop`, `50/50` losses vs `weak_basic_v2`, `50/50`
  losses vs `basic`.
- Replays:
  - `data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_stochastic.replay`

Do not retry this config-only focus-fire path.

## Current Assignment From Master

### Title

Audit Phase 4 multi-enemy closed-loop supervised bridge zero-score draw.

### Assignment Type

Offline audit/design only. Do not launch training and do not change code or
configs. Reconcile why the closed-loop bridge now passes visible fire,
hit/fire, objective_on_point, and loss-count floors but still produces zero
score and no wins. Choose a bounded next step and explicitly state whether
user approval is required.

### Config Path Under Audit

```text
experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml
```

### Seed

`3519994490`

### Relevant Output Directory

```text
python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/
```

### Required Preflight

Run from repo root:

```powershell
git status --short
git rev-parse HEAD
Get-Process | Where-Object { $_.ProcessName -like '*python*' }
```

Then from `python/`:

```powershell
py -3.13 -m scripts.check_import_boundaries
```

If a relevant Python training process is already running, stop and report it.

### Evidence To Respect

- `docs/plans/archive/2026-05-22-phase4-multi-enemy-closed-loop-supervised-bridge-result.md`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/multi_enemy_closed_loop_supervised_bridge_summary.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-supervised-bridge-result.md`

### Stop Criteria

- Stop if a relevant Python training process is already running.
- Stop if the audit cannot reconcile required artifact evidence.
- Stop if the next plausible step requires user approval; report the exact
  approval request instead of implementing it.
- Do not alter rewards, sim rules, tick pipeline, action semantics, replay
  format, phase-gate thresholds, or existing W&B metric schema.

### Completion Evidence

Return the audit doc path, concise findings, proposed next step, whether user
approval is required, and standard completion metadata.

## Previous Assignment From Master

### Title

Implement bounded opt-in Phase 4 multi-enemy actor-observation ablation.

### Assignment Type

Code/config/test implementation only, followed by focused preflight and a
direct diagnostic. Do not launch W&B training in this assignment. The training
run becomes a separate assignment only after the master verifies the
implementation, leak/shape/import tests, and direct diagnostic.

### Allowed Files Or Directories

Writable:

- `python/xushi2/`
- `python/envs/`
- `python/train/`
- `python/tests/`
- `python/scripts/`
- `experiments/configs/phase4/probe/`
- `docs/plans/active/2026-05-22-phase4-actor-information-decision.md`
- new run/diagnostic artifact directory only if needed for direct diagnostic

Read-only unless a narrow update is needed for completion metadata:

- `GOAL_INSTRUCTIONS_MASTER.md`
- `GOAL_INSTRUCTIONS_WORKER.md`
- `docs/plans/README.md`
- `docs/plans/active/2026-05-22-phase4-full-env-teacher-rehearsal-design.md`
- latest `docs/journal/reinforcement_learning_journal.md`
- `docs/plans/archive/`
- `python/runs/phase4_mappo_full_env_rehearsal_v1/`
- `python/runs/phase4_mappo_full_env_rehearsal_v2/`
- `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/`
- `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay`
- `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay`
- `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay`
- `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`
- `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay`
- `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay`
- `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`

### Completed V1 Config Path

```text
experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml
```

### Completed V1 Seed

`3519994490`

### Completed V1 Output Directory

```text
python/runs/phase4_mappo_full_env_rehearsal_v1/
```

### Completed V1 W&B

https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu

### Completed V2 Config Path

```text
experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml
```

### Completed V2 Seed

`3519994490`

### Completed V2 Output Directory

```text
python/runs/phase4_mappo_full_env_rehearsal_v2/
```

### Completed V2 W&B

https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr

### V1 Result To Respect

The v1 probe completed with `NOT_REACHED`. The supervised rehearsal loss
converged by step 2000 (`loss=0.0012`, `move=0.0007`, `aim=0.0002`,
`fire=0.0003`, `target=0.0001`), but the pre-PPO gate failed:
`team_a_hit_fire=0.0005556`, `objective_on_point=0.015`, `losses=50`,
`wins=0`, score `0.00/37.00`. PPO was intentionally skipped. Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-result.md`.

The v2 probe also completed with `NOT_REACHED`. It corrected the actor aim
convention bug found in v1 and added a regression test. V2 reduced Team A
greedy replay mean nearest-visible aim error from `2.5406` rad to `0.6073`
rad, but still produced `team_a_hit_fire=0.0`, `objective_on_point=0.01`,
`losses=50`, `wins=0`, and score `0.00/37.00`. Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v2-result.md`.

The direct teacher diagnostic completed after v2. Artifacts:
`python/runs/phase4_mappo_full_env_rehearsal_v2/scripted_teacher_diagnostic.json`
and
`python/runs/phase4_mappo_full_env_rehearsal_v2/cpp_basic_teacher_diagnostic.json`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-teacher-diagnostic-result.md`.

The v3 cpp_basic preflight implementation completed after the diagnostic.
Focused tests passed (`16 passed`), broader focused suite passed (`76 passed`),
import boundary passed, and direct teacher diagnostic on the v3 config passed:
`10W/0L/0D`, score `12.70/0.00`, hit/fire `0.0917`,
objective_on_point `0.8667`.

The v3 cpp_basic experiment completed after preflight. It returned
`NOT_REACHED` at the pre-PPO gate: Team A hit/fire `0.0061111111`,
objective_on_point `0.0166666667`, `50/50` losses, `0` wins, and score
`0.00/37.00`. PPO was intentionally skipped. Matrix eval was negative and
replays exist. Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md`.

### V3 Config Path

```text
experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml
```

### V3 Seed

`3519994490`

### Completed V3 Output Directory

From `python/` launch:

```text
python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/
```

### Completed V3 W&B

https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2

### Preflight

Run from repo root:

```powershell
git status --short
git rev-parse HEAD
Get-Process | Where-Object { $_.ProcessName -like '*python*' }
```

If a relevant training job is running, report it and do not start another job.

### Audit Questions

Already answered in:
`docs/plans/active/2026-05-22-phase4-actor-information-decision.md`.

Summary: the current flat Phase 4 actor observation cannot faithfully represent
the direct `cpp_basic` target-selection behavior because it exposes only the
counterpart enemy. The next implementation should wait for explicit user
approval for a bounded opt-in multi-enemy actor-observation ablation.

### User Approval

User approval was received in-thread on 2026-05-22 to change the master and
worker handoffs so work can continue. This approval is narrow and authorizes
only the bounded opt-in ablation described here.

### Authorized Scope

Allowed:

- Add one opt-in Phase 4 actor observation mode/config for a probe that exposes
  all visible enemy slots as masked actor-side entity tokens.
- Reuse existing Phase 7+/snapshot multi-enemy observation machinery when it is
  the safest path, but only through visibility-gated actor-visible data.
- Add actor/model/env shape plumbing required for the opt-in mode.
- Add or update tests proving:
  - hidden or non-visible enemies do not change actor-visible fields;
  - visible enemy slots do change;
  - masks are correct;
  - Team A/Team B frame conventions match existing actor obs;
  - import boundaries still hold;
  - the feature is off by default.
- Add one probe config under `experiments/configs/phase4/probe/`.
- Add one direct diagnostic command/artifact that proves the widened actor-side
  teacher/action path can act in the full `weak_basic_v2` distribution before
  any training run.

Disallowed:

- Reward changes.
- Sim-rule or tick-pipeline changes.
- Action semantics changes or new action-space-facing target fields.
- Replay format changes.
- Phase-gate threshold changes.
- Existing W&B metric/schema changes.
- Unbounded training or automatic W&B training launch.

### Required Preflight / Verification

Before returning completion, run and cite as many of these as are available in
the current checkout. If any listed binary is missing because the C++ build is
not present, report it explicitly and run the closest Makefile/CMake build
command only if needed for the verification.

```powershell
git status --short
git rev-parse HEAD
Get-Process | Where-Object { $_.ProcessName -like '*python*' }
```

From `python/`:

```powershell
py -3.13 -m scripts.check_import_boundaries
py -3.13 -m pytest tests/test_phase7_partial_obs.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py -q
py -3.13 -m pytest tests/test_phase4_mappo_env.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py -q
```

Run new focused tests for the ablation.

If C++ tests are built, run:

```powershell
.\build\tests\test_actor_leak.exe
.\build\tests\test_actor_obs.exe
.\build\tests\test_critic_obs.exe
.\build\tests\test_obs_dims.exe
```

If test binary names do not include `.exe` in this checkout, use the existing
Windows-valid names under `build/tests/`.

### Stop Criteria

- Do not start W&B training.
- Stop if implementing the ablation requires reward, sim-rule, action,
  replay-format, phase-gate, or existing W&B schema changes.
- Stop if hidden/non-visible enemy state leaks into actor-visible fields.
- Stop if the feature cannot be kept opt-in/off-by-default.
- Stop if the direct diagnostic cannot produce usable evidence after the
  implementation and focused tests pass; report `BLOCKED` with the reason.

### Completion

Implement the scoped ablation, run the required focused tests and direct
diagnostic, and report completion metadata. Do not launch the W&B training run.

### Completion Metadata Requirements

End with the standard metadata object:

```json
{
  "changed_files": [],
  "verification": [],
  "commit": null,
  "config_path": null,
  "seeds": [],
  "wandb_run_url": null,
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [],
  "blocked_reason": null,
  "residual_risk": [],
  "decision": "IMPLEMENTED_PRETRAINING_NOT_RUN"
}
```

Decision must be one of `NOT_REACHED`, `NOT_CLEARED`,
`HUMAN_INSPECTION_REQUIRED`, `CLEARED`, `EVIDENCE_INSUFFICIENT`, or `BLOCKED`.

## Preflight Checklist

Run from repo root unless the command says otherwise.

```powershell
git status --short
git rev-parse HEAD
Get-Process | Where-Object { $_.ProcessName -like '*python*' }
```

If a relevant training job is already running, stop before starting another
one and report it to the master.

For Python work, prefer:

```powershell
cd python
py -3.13 -m pytest <focused-tests> -q
py -3.13 -m scripts.check_import_boundaries
```

Use the exact focused tests named by the assignment. Do not run long training
jobs from a code-editing task unless the assignment explicitly asks for it.

## Experiment Run Protocol

For an assigned experiment:

1. Confirm the config path exists and record its contents in the summary.
2. Confirm seed(s), output directory, W&B group, phase gate, and stop criteria.
3. Run assigned preflight tests.
4. Launch from `python/` unless the assignment says otherwise:

```powershell
py -3.13 -m train.train --config ..\<config-path>
```

5. Capture or later recover the W&B URL. Do not guess it. If stdout does not
   print it, inspect the run's `wandb/latest-run/files/wandb-metadata.json`
   from the correct working directory.
6. Respect configured early-stop rules.
7. Run the configured matrix eval, gate CLI, replay dump, or smoke tests if the
   assignment requires them.
8. Write/update the assigned result docs only after artifacts exist.

## Stop Conditions

Treat bad metrics as a completed negative result, not a blocker.

Use `blocked_reason` only for actual inability to produce evidence, such as:

- build/import failure,
- W&B auth failure when W&B is required,
- missing config or checkpoint,
- process crash before usable metrics,
- machine sleep/disconnect,
- timeout before any usable evidence,
- human replay judgment required after objective checks pass,
- requested code change is outside the assignment.

If subjective replay judgment is required, report:

```text
HUMAN_INSPECTION_REQUIRED
```

and include W&B URL, replay path, viewer command, exact questions, and the
comment format needed to unblock.

## Completion Metadata

End every assigned task with machine-readable metadata:

```json
{
  "changed_files": [],
  "verification": [],
  "commit": null,
  "config_path": null,
  "seeds": [],
  "wandb_run_url": null,
  "replay_artifacts": [],
  "viewer_command": null,
  "tests_run": [],
  "behavior_changes": [],
  "reward_changes": [],
  "config_changes": [],
  "blocked_reason": null,
  "residual_risk": []
}
```

Also include a short human summary:

- what was assigned,
- what ran,
- key metrics,
- decision,
- what should not be retried.

## Default If No Assignment Is Provided

Do not train. Produce an audit report with:

- current dirty worktree,
- latest journal/result-doc status,
- latest relevant run directories,
- W&B/replay/gate artifacts that exist,
- conflicts or stale recommendations that require master decision.
