# Goal Instructions: Phase 4 Master

Date: 2026-05-22

## Role

This file is for the coordinating Codex instance. Its job is to supervise a
separate `codex --yolo` worker that runs experiments for the same high-level
goal: clearing Phase 4.

The master owns:

- reading the source-of-truth docs before assigning work,
- choosing the next bounded worker assignment,
- preventing duplicate or already-falsified runs,
- checking worker claims against local artifacts,
- deciding whether evidence is `CLEARED`, `NOT_CLEARED`, `NOT_REACHED`,
  `EVIDENCE_INSUFFICIENT`, or `HUMAN_INSPECTION_REQUIRED`,
- updating handoff instructions after each completed worker run.

The worker owns:

- executing exactly one assigned implementation or experiment task at a time,
- running the required preflight checks,
- capturing W&B, replay, matrix-eval, gate, and test evidence,
- reporting completion metadata without inventing missing evidence.

Do not let both agents edit the same file set at the same time. The safest
default is: master edits instruction/result docs; worker edits code/configs and
run artifacts only for the current assignment.

When a current worker assignment is present, the master should create or start
a separate worker Codex process/session for that assignment. The master should
not implement the worker assignment inline unless explicitly told to collapse
the roles.

## Required First Reads

Read these before issuing or accepting worker output:

1. `AGENTS.md` instructions from the session prompt.
2. `docs/plans/README.md`.
3. `docs/journal/reinforcement_learning_journal.md`, especially the latest
   Phase 4 entries.
4. Latest result files under `docs/plans/archive/`.
5. Candidate config under `experiments/configs/phase4/probe/`.
6. Run artifacts under `python/runs/<run_name>/` or `runs/<run_name>/`.

If docs disagree with code, config, or artifacts, stop and reconcile. Do not
silently pick the convenient source.

## Current Phase 4 Audit

Phase 4 is not cleared.

The current evidence chain says:

- `phase4_mappo_basic_v6_5` produced a useful objective/cap checkpoint, but
  direct transfer to combat opponents collapsed.
- Escape Protocol isolated probes were already tried and falsified as complete
  fixes: auxiliary aim head, per-action entropy, invalid-fire mask,
  aim-only transfer, aim-freeze/aim-target variants, target conditioning, and
  mode gating. Do not rerun exact variants.
- `phase4_mappo_cap_duel_selfplay_v2` is a valid isolated combat/objective
  teacher. It removed v1 quirks: spawn-on-point, knockback displacement, and
  respawn-on-point. It trained with seed `3519994490` and W&B
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/l9890sl6`.
- Cap-duel v2 inspection found honest kill-then-hold behavior:
  greedy `8W/0L/2D`, stochastic `10W/0L/0D`, combined
  `200/212 = 94.3%` kill-then-hold score attribution.
- Composition rehearsal with the cap-duel v2 teacher failed before PPO:
  - 2000-step config:
    `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml`.
    W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ao4hy6fa`.
    Full Team A hit/fire `0.0047 < 0.0400`.
  - 4000-step fallback:
    `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml`.
    W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ssmkgzua`.
    Full Team A hit/fire `0.0116 < 0.0400`.
  - Status: `NOT_REACHED`; no useful full-3v3 replay was produced.
- PPO-time cap-duel v2 distillation anchor was implemented and run:
  `experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml`.
  Seed `1779134702`, W&B
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/c8yyqsdd`.
  It early-stopped at update 50 with `team_a_hit_fire=0.0168918919 < 0.04`
  and `mean_score_a=0.0`.
  Phase gate status: `NOT_CLEARED`.
- Matrix eval for the distillation anchor:
  - vs `noop`: draw-only, score `0.00/0.00`.
  - vs `weak_basic_v2`: `50/50` losses, score `0.00/23.33`.
  - vs `basic`: `50/50` losses, score `0.00/37.00`.
- Distillation replays exist:
  - `data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_stochastic.replay`
  Viewer command:
  `xushi2-viewer --replay data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay`.

Important reconciliation note: the latest cap-duel distillation result
recommends "Strategy 3 focus-fire target conditioning", but target-conditioned
focus-fire machinery already exists in this checkout and was previously run on
2026-05-18. `phase4_mappo_focus_fire_v1` fixed measured target concentration
but did not score; `phase4_mappo_cap_duel_focus_fire_v1` using the older
cap-duel v1 checkpoint collapsed by update 50. Therefore "Strategy 3" must not
mean reimplementing the same actor head from scratch. It can only mean a
deliberately scoped follow-up that reuses or revises the existing machinery,
with the prior May 18 results called out.

2026-05-22 audit result: accepted. The worker reconciled Strategy 3 against
current code/results and found that a single config-only cap-duel v2 plus
existing focus-fire probe is defensible because the exact combination of the
honest cap-duel v2 checkpoint and the existing `team_focus_low_hp` machinery
has not been run. This does not authorize reimplementing target conditioning or
adding action-space-facing target machinery.

2026-05-22 cap-duel v2 focus-fire probe result: `NOT_CLEARED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml`.
Seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/j90y7l1i`.
Final eval: `0W/50L/0D`, score `0.00/35.93`, Team A hit/fire `0.0017`.
Matrix eval: draw-only vs `noop`, `50/50` losses vs `weak_basic_v2`, `50/50`
losses vs `basic`. Gate decision:
`python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/gate_decision.json`.
Status: stop this config-only focus-fire path.

New active design plan:
`docs/plans/active/2026-05-22-phase4-full-env-teacher-rehearsal-design.md`.
The next move is implementation-design/preflight for full-env teacher
rehearsal, not another experiment launch.

2026-05-22 full-env teacher rehearsal implementation status: implemented and
focused tests passed. Added:

- `python/train/full_env_rehearsal.py`
- `python/tests/test_full_env_rehearsal.py`
- `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`

Wired the rehearsal hook before composition/BC in `mappo_eval_checkpoint.py`.
No C++, reward, observation/action schema, replay, existing W&B schema, or gate
threshold changes. Verification: `9 passed` new/pretrain tests, `31 passed`
focus/leak-adjacent tests, `70 passed` broader focused suite, import boundary
PASS.

2026-05-22 full-env teacher rehearsal probe result: `NOT_REACHED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`.
Seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu`.
The supervised rehearsal labels were mechanically learned by step 2000
(`loss=0.0012`, `move=0.0007`, `aim=0.0002`, `fire=0.0003`,
`target=0.0001`), but the pre-PPO gate failed:
`team_a_hit_fire=0.0005556 < 0.04`, `objective_on_point=0.015 < 0.25`,
`losses=50 > 49`, `wins=0`, score `0.00/37.00`. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/full_env_rehearsal_gate.json`.
Because the gate returned `NOT_REACHED`, PPO was intentionally skipped and
there is no post-PPO phase-gate decision. Replays exist:
`data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay`.
Status: stop this full-env scripted rehearsal v1 path before PPO.

2026-05-22 full-env teacher rehearsal v2 result: `NOT_REACHED`.
Audit of the v1 replay found a real aim-label bug: actor aim vectors are
documented as `(sin theta, cos theta)`, but v1 computed target angle with
`atan2(y, x)`. The helper now uses `atan2(x, y)` for target vectors and has a
regression test. Corrected v2 config:
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`.
Seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr`.
The corrected labels converged by step 2000 (`loss=0.0012`, `aim=0.0003`),
but the pre-PPO gate still failed: `team_a_hit_fire=0.0 < 0.04`,
`objective_on_point=0.01 < 0.25`, `losses=50 > 49`, `wins=0`, score
`0.00/37.00`. V2 replay analysis reduced Team A mean nearest-visible aim error
from v1 greedy `2.5406` rad to v2 greedy `0.6073` rad, but Team A still dealt
`0` damage while firing continuously. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/full_env_rehearsal_gate.json`.
Replays:
`data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay` and
`data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`.
Status: stop corrected full-env scripted rehearsal v2 before PPO.

2026-05-22 direct full-env teacher diagnostic result: accepted.
Added `python/scripts/diagnose_full_env_teacher.py` and tests. With the v2
config and seed `3519994490`, the actor-observation-only teacher itself failed
against `weak_basic_v2`: `0W/0L/50D`, score `0.00/0.00`, Team A hit/fire
`0.0`, visible-fire rate `1.0`, objective_on_point `0.0`. Against `noop`, the
same teacher won `10/10`, score `37.00/0.00`, objective_on_point `0.9333`.
The full-state `cpp_basic` teacher against `weak_basic_v2` won `10/10`, score
`12.70/0.00`, Team A hit/fire `0.0917`, objective_on_point `0.8667`.
Artifacts:
`python/runs/phase4_mappo_full_env_rehearsal_v2/scripted_teacher_diagnostic.json`
and
`python/runs/phase4_mappo_full_env_rehearsal_v2/cpp_basic_teacher_diagnostic.json`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-teacher-diagnostic-result.md`.
Interpretation: actor-observation-only v1/v2 is not worth more compute. The
flat Phase 4 actor obs exposes only the counterpart enemy through
`visible_enemy_1v1`; `cpp_basic` succeeds by choosing among all visible enemies.

2026-05-22 full-env rehearsal v3 cpp_basic result: `NOT_REACHED`.
Added opt-in `teacher: cpp_basic` support in
`python/train/full_env_rehearsal.py`, focused tests, and config
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`.
The C++ `basic` bot supplies privileged training-time movement/aim/fire labels;
actor inference remains unchanged and actor-observation-only. V3 disables the
target-selection head to avoid mixing prior focus-fire conditioning with
full-state labels. Direct teacher diagnostic on the v3 config passed:
`10W/0L/0D`, score `12.70/0.00`, Team A hit/fire `0.0917`,
objective_on_point `0.8667`. Diagnostic artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/cpp_basic_teacher_diagnostic.json`.
Verification: `16 passed` focused full-env/pretrain tests, `76 passed` broader
focused suite, import boundary PASS, config smoke `obs_dim=31`, `action_dim=6`,
`target_selection_dim=0`, replay smoke `11 passed`.

The W&B run was
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2`. The
pre-PPO gate failed with Team A hit/fire `0.0061111111 < 0.04`,
objective_on_point `0.0166666667 < 0.25`, `50/50` losses, `0` wins, and score
`0.00/37.00`. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/mappo/full_env_rehearsal_gate.json`.
PPO was intentionally skipped. Matrix eval remained negative: draw-only vs
`noop`, `50/50` losses vs `weak_basic_v2`, and `50/50` losses vs `basic`.
Replays:
`data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay`
and
`data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md`.
Status: stop the privileged full-env rehearsal path before PPO.

2026-05-22 actor information audit result: implementation now requires user
approval. Audit doc:
`docs/plans/active/2026-05-22-phase4-actor-information-decision.md`.
Conclusion: the current flat Phase 4 actor observation exposes only one
counterpart enemy through `visible_enemy_1v1`, while direct `cpp_basic`
succeeds by choosing among all visible enemies. Another no-observation-change
teacher is not justified unless it first wins or scores as direct actions in
the full `weak_basic_v2` distribution. Recommended next step is to ask the user
to approve one bounded opt-in Phase 4 multi-enemy actor-observation ablation
with strict leak/shape/import tests and no reward, sim, action, replay, or gate
changes.

2026-05-22 user approval received in thread: continue by updating the master
and worker handoffs so the next worker can implement the bounded opt-in Phase 4
multi-enemy actor-observation ablation. This approval is narrow. It authorizes
only the implementation assignment described below; it does not authorize
reward changes, sim-rule changes, action semantics changes, replay format
changes, phase-gate threshold changes, existing W&B schema changes, or
unbounded training.

2026-05-22 multi-enemy actor-observation preflight result:
`IMPLEMENTED_PRETRAINING_NOT_RUN`. Config:
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`.
Seed `3519994490`. W&B: none; no training launched. Artifact:
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json`.
The opt-in wrapper exposes currently visible enemy slots as masked
multi-enemy entity-grid actor observations, zeroes hidden enemy token payloads,
and leaves the default Phase 4 flat actor path unchanged. Direct
`multi_enemy_visible` teacher diagnostic vs `weak_basic_v2` produced
`10W/0L/0D`, score `9.20/0.00`, Team A hit/fire `0.09`, and
objective_on_point `0.875`. Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`.
Status: ready for one separate bounded W&B training assignment; do not treat
the direct diagnostic as gate evidence.

2026-05-22 multi-enemy actor-observation training attempt result: `BLOCKED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`.
Seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/bw9jsxte`.
The worker preflight passed, W&B authenticated, and the run started, but it
crashed before usable metrics during strict warm-start. The configured
checkpoint `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` is a flat-actor
model and cannot load strictly into the new `entity_attention_grid` actor
topology; missing keys included `actor_entity_encoder`, `actor_grid_encoder`,
and `actor_fusion`, with unexpected `actor_embed` keys. No gate, matrix,
checkpoint, replay, or training metrics were produced. Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-blocked-result.md`.
Status: do not retry this config unchanged. The next move is a bounded
implementation/preflight fix for opt-in warm-start migration across the actor
observation topology change.

2026-05-22 multi-enemy actor-observation warm-start fix result:
`IMPLEMENTED_PRETRAINING_NOT_RUN`. Added explicit
`run.warm_start_migration: compatible_exact`; default warm-start remains
strict. The opt-in migration loads only same-name, same-shape tensors and
reports missing, unexpected, and shape-mismatched keys. The probe config now
opts into this mode. Master verification passed: import boundary PASS,
multi-enemy actor-observation/diagnostic tests `11 passed`, warm-start hook
tests `8 passed`, and a non-training smoke loaded the flat Phase 4 checkpoint
into the `entity_attention_grid` model with `obs_dim=3167`, `action_dim=6`,
`target_selection_dim=0`, and 17 compatible tensors loaded. Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-warmstart-result.md`.
Status: ready for one separate bounded W&B training assignment using the same
config.

2026-05-22 multi-enemy actor-observation training result: `NOT_CLEARED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`.
Seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw`.
The run completed 100/100 updates. Best and final eval were both
`0W/50L/0D`, score `0.00/37.00`, mean reward `-11.0`, Team A hit/fire `0.0`.
Matrix eval was draw-only vs `noop`, `50/50` losses vs `weak_basic_v2`, and
`50/50` losses vs `basic`; transfer summary gate status was
`evidence_insufficient`. Replays exist:
`data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`.
Replay analysis found greedy Team A issued no fire commands and stochastic
Team A fired continuously but barely hit. Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`.
Status: do not retry this same config unchanged. Objective checks did not
pass, so no human replay inspection is required for clearance. The next move is
an offline post-run audit/design decision, not another automatic training
launch.

2026-05-22 multi-enemy training failure audit result:
`IMPLEMENTED_PRETRAINING_NOT_RUN`. Audit doc:
`docs/plans/active/2026-05-22-phase4-multi-enemy-training-failure-audit.md`.
The audit reconciled the positive direct `multi_enemy_visible` teacher with
the failed PPO run: widened actor observations are sufficient for scripted
direct action selection, but the neural policy did not inherit a usable actor
mapping because the compatible warm-start skipped the old flat actor encoder
and left `actor_entity_encoder`, `actor_grid_encoder`, and `actor_fusion`
effectively random. Recommendation: do not retry the same config unchanged and
do not run a longer/coefficient-only PPO variant next. Proposed next step is
one bounded opt-in supervised bridge using existing `multi_enemy_visible`
teacher labels before PPO, with a pre-PPO neural-policy gate. This requires
explicit user approval before implementation.

2026-05-22 user approval received in thread: proceed with the bounded opt-in
Phase 4 multi-enemy supervised bridge recommended by
`docs/plans/active/2026-05-22-phase4-multi-enemy-training-failure-audit.md`.
This approval is narrow. It authorizes one implementation/preflight assignment
for an opt-in supervised bridge using existing `multi_enemy_visible` teacher
labels before PPO. It does not authorize reward changes, sim-rule changes,
tick-pipeline changes, action semantics changes, action-space-facing target
fields, replay format changes, phase-gate threshold changes, existing W&B
metric/schema changes, unbounded training, or automatic W&B training launch.

2026-05-22 multi-enemy supervised bridge result: `NOT_REACHED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`.
Seed `3519994490`. W&B: none; no training launched. The opt-in bridge used
existing `multi_enemy_visible` teacher labels for movement, aim, and fire, and
wrote a no-W&B pre-PPO gate artifact. Labels converged after 2000 supervised
steps (`loss=0.0005422`, `move=0.0000411`, `aim=0.0003568`,
`fire=0.0001443`), but the neural-policy gate failed:
Team A visible fire rate `1.0`, Team A hit/fire `0.0088888889 < 0.04`,
objective_on_point `0.0116666667 < 0.25`, mean score A `0.0 < 1.0`,
`50/50` losses, and `0` wins. Artifacts:
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/multi_enemy_supervised_bridge_summary.json`,
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/multi_enemy_supervised_bridge_gate.json`,
and
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-supervised-bridge-result.md`.
Status: stop before PPO. Do not retry this same bridge config unchanged, do
not increase bridge length as the next move, and do not force PPO past the
failed pre-PPO gate.

2026-05-22 multi-enemy supervised bridge failure audit result:
`IMPLEMENTED_PRETRAINING_NOT_RUN`. Audit doc:
`docs/plans/active/2026-05-22-phase4-multi-enemy-supervised-bridge-failure-audit.md`.
The audit reconciled the direct `multi_enemy_visible` teacher success with the
failed PPO and one-shot supervised bridge results. The actor-visible
information surface is sufficient for direct scripted action selection, but
the compatible warm start left the widened neural actor front end effectively
random, and the one-shot bridge learned labels on expert-style states without
surviving closed-loop policy-state rollout. Proposed next step: one bounded
opt-in closed-loop supervised bridge using policy-induced states and existing
`multi_enemy_visible` teacher labels, with movement/aim/fire agreement
diagnostics and the same pre-PPO gate. This requires explicit user approval
before implementation. Do not launch another worker until that approval is
received.

2026-05-22 user approval received in thread: proceed with one bounded opt-in
Phase 4 closed-loop supervised bridge assignment using policy-induced states
plus existing `multi_enemy_visible` teacher labels. This approval is narrow.
It does not authorize reward changes, sim-rule changes, tick-pipeline changes,
action semantics changes, action-space field changes, replay format changes,
phase-gate threshold changes, existing W&B metric/schema changes, unbounded
training, or automatic W&B/PPO launch.

2026-05-22 multi-enemy closed-loop supervised bridge result: `NOT_REACHED`.
Config:
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`.
Seed `3519994490`. W&B: none; PPO was not launched. The closed-loop bridge
used policy-induced states and existing `multi_enemy_visible` labels. Final
policy-state agreement was strong for fire (`fire_accuracy=1.0`,
`fire_positive_recall=1.0`) and moderate for aim (`aim_abs_error=0.2027`).
The pre-PPO gate improved over the one-shot bridge but failed score:
Team A visible-fire rate `1.0`, Team A hit/fire `0.0427777778`,
objective_on_point `0.29`, losses `0`, mean score B `0.0`, but mean score A
`0.0 < 1.0` and wins `0`. Artifacts:
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/multi_enemy_closed_loop_supervised_bridge_summary.json`,
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`,
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`, and
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-closed-loop-supervised-bridge-result.md`.
Status: stop before PPO. Do not retry this same config unchanged and do not
force PPO past the failed score check. The next move is an offline zero-score
draw audit, not training.

## Worktree Caution

At audit time, the worktree already had modified/untracked Phase 4 files from
other work, including `GOAL_INSTRUCTIONS.md`,
`docs/journal/reinforcement_learning_journal.md`, trainer files, tests, and new
Phase 4 probe configs. Treat those as existing user/worker changes. Do not
revert them.

Before assigning a worker task, capture:

```powershell
git status --short
git rev-parse HEAD
```

For gate evidence, a dirty worktree must be described as:

```text
git_commit + explicit working-tree delta
```

or the changes must be committed before using the result as strong gate
evidence.

## Assignment Rules

Issue one worker assignment at a time. A valid assignment has:

- title,
- allowed files or directories,
- config path,
- seed(s),
- expected output directory,
- W&B requirements,
- preflight tests,
- stop criteria,
- completion metadata requirements.

Do not assign:

- another composition-rehearsal length variant from the cap-duel v2 path,
- another PPO-time distillation coefficient-only variant,
- exact repeats of May 18 focus-fire, cap-duel focus-fire, mode-gated,
  auxiliary aim, per-action entropy, invalid-fire mask, or aim-transfer probes,
- load-bearing sim/reward/observation/action/replay changes without explicit
  user approval.

## Candidate Next Decisions

The next move is not an automatic training command. The master should pick one
of these and make it explicit:

1. Do not launch another full-env rehearsal length variant, coefficient-only
   variant, or forced-PPO variant from v1/v2/v3. All three failed the pre-PPO
   gate.
2. The offline design/audit step is complete:
   `docs/plans/active/2026-05-22-phase4-actor-information-decision.md`.
   The next move is a user approval request for a bounded opt-in multi-enemy
   actor-observation ablation, not implementation.
3. User approval has been received for exactly one bounded opt-in Phase 4
   multi-enemy actor-observation ablation. The next worker assignment may
   implement that scoped ablation and focused tests. It must not launch a
   training run until implementation preflight and direct diagnostic pass.
4. If a no-observation-change teacher is proposed, require an offline
   diagnostic showing that the teacher itself wins or scores in the full
   `weak_basic_v2` distribution before training the policy against it.

## Completed Worker Assignment

The bounded opt-in Phase 4 multi-enemy actor-observation ablation
implementation/preflight assignment completed. The master verified the local
diagnostic artifact and updated the result docs/journal.

### Title

Implement bounded opt-in Phase 4 multi-enemy actor-observation ablation.

### Assignment Type

Code/config/test implementation only, followed by focused preflight and a
direct diagnostic. Do not launch W&B training in this assignment. The training
run becomes a separate assignment only after the master verifies the
implementation, leak/shape/import tests, and direct diagnostic.

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
py -3.13 -m pytest tests/test_phase7_partial_obs.py tests/test_phase4_multi_enemy_actor_obs.py -q
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

- Stop if implementing the ablation requires reward, sim-rule, action,
  replay-format, phase-gate, or existing W&B schema changes.
- Stop if hidden/non-visible enemy state leaks into actor-visible fields.
- Stop if the feature cannot be kept opt-in/off-by-default.
- Stop if the direct diagnostic cannot produce usable evidence after the
  implementation and focused tests pass; report `BLOCKED` with the reason.

### Completion Metadata

Worker completion must include:

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

If an assignment requires subjective replay judgment and the objective checks
pass, return `HUMAN_INSPECTION_REQUIRED` with W&B URL, replay path, viewer
command, exact questions, and the comment format needed to unblock.

## Next Worker Assignment

Master action: create/start a separate worker Codex process/session and give it
`GOAL_INSTRUCTIONS_WORKER.md` as the active assignment. The master should then
wait for the worker completion report, verify artifacts locally, and update
handoff/result docs before choosing the next assignment.

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

### Audit Scope

Allowed:

- Read result docs, configs, run artifacts, logs, and relevant replay/gate
  tooling.
- Add one audit/design note under `docs/plans/active/` if evidence supports a
  specific next step.
- Report whether any proposed next implementation or W&B/PPO run requires
  explicit user approval.

Disallowed:

- W&B training, PPO launch, or local training.
- Source or config changes.
- Reward changes.
- Sim-rule or tick-pipeline changes.
- Action semantics changes or new action-space-facing target fields.
- Replay format changes.
- Phase-gate threshold changes.
- Existing W&B metric/schema changes.

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

If a relevant Python training job is already running, stop and report it.

### Evidence To Respect

- `docs/plans/archive/2026-05-22-phase4-multi-enemy-closed-loop-supervised-bridge-result.md`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/multi_enemy_closed_loop_supervised_bridge_summary.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`
- `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-warmstart-result.md`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`
- `docs/plans/archive/2026-05-22-phase4-multi-enemy-supervised-bridge-result.md`
- `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/multi_enemy_supervised_bridge_summary.json`
- `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/multi_enemy_supervised_bridge_gate.json`
- `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`
- `python/wandb/run-20260522_141504-ud4c09jw/files/output.log`
- `python/wandb/run-20260522_141504-ud4c09jw/files/wandb-summary.json`
- `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/matrix_eval.json`
- `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.json`
- `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay`
- `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`
- `docs/plans/active/2026-05-22-phase4-multi-enemy-supervised-bridge-failure-audit.md`

### Stop Criteria

- Stop if a relevant Python training process is already running.
- Stop if the audit cannot reconcile required artifact evidence.
- Stop if the next plausible step requires user approval; report the exact
  approval request instead of implementing it.
- Do not alter rewards, sim rules, tick pipeline, action semantics, replay
  format, phase-gate thresholds, or existing W&B metric schema.

### Completion Evidence

Return the audit doc path, concise findings, proposed next step, whether user
approval is required, and standard completion metadata. Decision should be
`EVIDENCE_INSUFFICIENT`, `BLOCKED`, or `IMPLEMENTED_PRETRAINING_NOT_RUN`
(meaning audit/design complete, no training run).

## Worker Completion Acceptance

Do not accept a worker result unless it includes:

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

The master should verify local artifacts before updating the journal or result
docs. W&B URL, seed, config path, and replay paths are mandatory for any
experiment result that could influence a Phase 4 gate decision.
