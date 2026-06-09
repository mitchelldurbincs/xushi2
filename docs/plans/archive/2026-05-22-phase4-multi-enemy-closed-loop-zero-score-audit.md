# Phase 4 Multi-Enemy Closed-Loop Zero-Score Audit

Date: 2026-05-22

## Assignment

Audit why
`phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml` passes the
visible-fire, hit/fire, objective-on-point, and loss-count pre-PPO floors but
still produces zero score and no wins.

Scope was offline audit/design only. No training was launched and no source or
config files were changed.

## Inputs

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`
- Seed: `3519994490`
- Output:
  `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/`
- Result doc:
  `docs/plans/archive/2026-05-22-phase4-multi-enemy-closed-loop-supervised-bridge-result.md`
- Required artifacts:
  - `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/multi_enemy_closed_loop_supervised_bridge_summary.json`
  - `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`
  - `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`
  - `python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`

## Evidence

The 50-episode pre-PPO gate artifact is internally consistent:

- status: `NOT_REACHED`
- Team A visible-fire rate: `1.0`
- Team A hit/fire: `0.0427777778`
- objective_on_point: `0.29`
- losses: `0`
- wins: `0`
- mean score A/B: `0.0/0.0`

The final policy-state agreement artifact also supports that the bridge learned
the closed-loop teacher labels it was asked to learn:

- movement MSE: `0.0154950926`
- aim absolute error: `0.2026730925`
- fire accuracy: `1.0`
- fire positive recall: `1.0`
- policy fire rate: `1.0`
- teacher fire rate: `1.0`

A read-only 5-episode checkpoint evaluation against `weak_basic_v2` was run to
inspect objective counters that are not written into the gate JSON. It
reproduced draw-only, zero-score behavior and exposed the missing conversion
step:

- `5/5` draws, score `0.00/0.00`
- Team A kills `13.4`, Team B kills `0.0`
- Team A hit/fire `0.047`, visible-fire rate `0.98`
- Team A majority-on-point seconds `24.34`
- Team A uncontested-on-point seconds `4.90`
- Team A cap-progress gain ticks `238.2`
- Team A cap-progress loss ticks `197.4`
- first Team A alive-edge-to-score seconds: `-1.0`
- objective unlock seconds `15.0`, capture seconds `8.0`

The game design requires the objective to unlock first, then a team must finish
`240` capture ticks before scoring starts. The sampled policy averages just
under one full capture worth of gross gain but loses most of it, and its
uncontested point time is well below the 8 seconds needed to finish a neutral
capture. `objective_on_point` therefore measures majority presence, not
completed ownership or scoring readiness.

## Reconciliation

The closed-loop bridge fixed the prior failure modes of no firing, poor
hit/fire, immediate losses, and total objective absence. It did not teach the
policy the last control-point conversion behavior: clearing or zoning the point
long enough after unlock to complete capture progress and then remain
uncontested for scoring ticks.

This is why the result can have useful hit/fire, no losses, and
`objective_on_point=0.29` while still scoring `0.00/0.00`. Team A often has a
temporary on-point majority and combat advantage, but the windows are
fragmented and contested. The gate's `objective_on_point` floor is necessary
but not sufficient for a control-point score gate.

The result should not be interpreted as clearance to force PPO. The configured
pre-PPO gate correctly rejected the checkpoint on `mean_score_a < 1.0`.

## Decision

Do not launch PPO or W&B training from
`phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`. Do not retry
the same closed-loop bridge unchanged. Do not weaken the score gate.

The bounded next step should be one opt-in closed-loop objective-conversion
bridge, still using the existing `multi_enemy_visible` teacher and
`actor_obs: multi_enemy_entity_grid`, but adding training-time labels and
diagnostics for the score-conversion gap:

- preserve rewards, sim rules, tick pipeline, action semantics,
  action-space fields, replay format, phase-gate thresholds, and existing W&B
  metric schema;
- keep the feature opt-in and off by default;
- collect policy-induced states against `weak_basic_v2`;
- continue movement/aim/fire imitation from existing `multi_enemy_visible`
  labels;
- add an objective-conversion diagnostic/gate based on existing env metrics:
  uncontested-on-point seconds, cap-progress gain/loss, first alive-edge to
  score, and mean score;
- if adding an auxiliary supervised objective-mode/action label, keep it
  internal to training and do not add an action-space-facing target field;
- stop before PPO unless the pre-PPO neural policy produces nonzero score, not
  just majority-on-point time.

This is a new supervised training path and requires explicit user approval
before implementation or any W&B/PPO assignment.

## Approval Request

Approve one bounded opt-in Phase 4 closed-loop objective-conversion bridge:

- no reward, sim-rule, tick-pipeline, action-semantics, replay-format,
  phase-gate-threshold, existing W&B-schema, or action-space-facing field
  changes;
- use the existing multi-enemy actor observation and existing
  `multi_enemy_visible` teacher;
- train only on policy-induced states against `weak_basic_v2`;
- add conversion diagnostics from existing objective metrics, especially
  uncontested-on-point seconds, cap-progress gain/loss, and first score after
  alive edge;
- require the same pre-PPO score check to pass before any later PPO/W&B run.

## Decision Status

`IMPLEMENTED_PRETRAINING_NOT_RUN`

Audit/design is complete. No training was launched. The proposed next step
requires explicit user approval.
