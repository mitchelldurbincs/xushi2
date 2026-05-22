# Phase 4 Multi-Enemy Supervised Bridge Failure Audit

Date: 2026-05-22

## Assignment

Audit why the direct `multi_enemy_visible` teacher wins in the full
`weak_basic_v2` distribution and the neural supervised bridge learned labels
mechanically, but the pre-PPO neural-policy gate still failed to produce useful
hit/fire, objective pressure, score, or wins.

Scope was offline audit/design only. No training was launched and no source or
config files were changed.

## Inputs

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`
- Seed: `3519994490`
- Output:
  `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/`
- Result docs:
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-warmstart-result.md`
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-supervised-bridge-result.md`
- Required local artifacts:
  - `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/multi_enemy_supervised_bridge_summary.json`
  - `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/multi_enemy_supervised_bridge_gate.json`
  - `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`
  - `python/wandb/run-20260522_141504-ud4c09jw/files/output.log`
  - `python/wandb/run-20260522_141504-ud4c09jw/files/wandb-summary.json`
  - `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/matrix_eval.json`
  - `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.json`
  - `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`

## Evidence

The direct widened-observation teacher remains positive evidence that the
actor-visible information surface is sufficient. The
`multi_enemy_visible` diagnostic against `weak_basic_v2` produced `10W/0L/0D`,
mean score `9.20/0.00`, Team A hit/fire `0.09`, visible-fire rate `1.0`, and
objective_on_point `0.875`.

The plain PPO run on the widened actor observation failed before this bridge.
It completed `100/100` updates but final eval was `0W/50L/0D`, score
`0.00/37.00`, Team A hit/fire `0.0`, objective focus fraction `0.0`, and
matrix eval was draw-only vs `noop` plus `50/50` losses vs `weak_basic_v2` and
`basic`. The run log showed `warm_start_migration=compatible_exact loaded=17`,
with the new `actor_entity_encoder`, `actor_grid_encoder`, and `actor_fusion`
keys missing from the warm start, so the widened actor front end began
effectively random.

The supervised bridge fixed the "no firing" failure mode but not the closed-loop
policy. The bridge ran `2000` supervised steps and converged to low label loss:
`loss=0.0005421519745141268`, `move_loss=0.000041060040530283004`,
`aim_loss=0.00035681703593581915`, and
`fire_loss=0.00014427490532398224`. The neural gate then failed:
Team A visible-fire rate was `1.0`, but Team A hit/fire was only
`0.0088888889`, objective_on_point was `0.0116666667`, mean score A was `0.0`,
and the policy lost `50/50` episodes with mean score B `31.9666666667`.

## Reconciliation

The apparent contradiction resolves as a closed-loop imitation failure:

- The actor information bottleneck is not the current blocker for direct
  scripted action selection. The same visible enemy tokens let the direct
  teacher fire, hit, win fights, and hold the point.
- The compatible warm start explains why plain PPO did not discover that
  mapping: the new actor front end was not initialized from the old flat actor
  encoder.
- The supervised bridge did learn the label mapping on its training
  distribution, but the gate result shows that low aggregate movement/aim/fire
  loss on teacher-collected states did not transfer to policy-induced states.
  The policy copied visible firing but fell off the teacher trajectory enough
  that it did not stay on point, did not create combat advantage, and hit at
  roughly one tenth of the direct teacher's hit/fire rate.

This is not evidence that the multi-enemy actor observation is useless, and it
is not evidence that a longer version of the same bridge should be run next.
The result specifically falsifies this one-shot expert-state bridge as a
pre-PPO launch condition.

## Decision

Do not launch PPO or W&B training from
`phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`. Do not retry the same
bridge config unchanged, do not increase bridge length as the next move, and
do not force PPO past the failed pre-PPO gate.

The next bounded design should be a closed-loop supervised bridge diagnostic,
not another open-loop behavior-cloning pass. It should keep the existing
multi-enemy actor observation and `multi_enemy_visible` teacher, but collect
and train on states induced by the current neural policy against
`weak_basic_v2`, with teacher labels queried on those states. It should report
per-component action agreement on policy states, then run the same pre-PPO gate
before any PPO update. If the gate still fails, stop with `NOT_REACHED`.

This is a new opt-in supervised training path and requires explicit user
approval before implementation.

## Approval Request

Approve one bounded opt-in Phase 4 closed-loop supervised bridge assignment:

- Keep rewards, sim rules, tick pipeline, action semantics, action-space fields,
  replay format, phase-gate thresholds, and existing W&B metric schema
  unchanged.
- Keep the feature opt-in and off by default.
- Use `actor_obs: multi_enemy_entity_grid` and the existing
  `multi_enemy_visible` teacher labels.
- Add a policy-state supervised bridge loop: roll out the current neural policy
  against `weak_basic_v2`, query teacher movement/aim/fire labels on those
  visited states, update the policy, and repeat for a bounded number of rounds.
- Add diagnostics for teacher-vs-policy movement, aim, and fire agreement on
  policy-induced states, plus the existing pre-PPO neural-policy gate.
- Stop before PPO unless the gate shows nonzero Team A firing, useful hit/fire,
  objective pressure, nonzero score, and not `50/50` losses.
- If the pre-PPO gate passes, allow exactly one later bounded W&B training
  assignment to collect normal Phase 4 evidence.

## Decision Status

`IMPLEMENTED_PRETRAINING_NOT_RUN`

Audit/design is complete. No training was launched. The proposed next step
requires explicit user approval.
