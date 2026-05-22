# Phase 4 Multi-Enemy Actor Observation Training Failure Audit

Date: 2026-05-22

## Assignment

Audit why the direct `multi_enemy_visible` teacher wins in the full
`weak_basic_v2` distribution, while the neural PPO run using the same widened
actor-observation surface produced no Team A firing, objective pressure, or
score.

Scope was offline audit/design only. No training was launched and no source or
config files were changed.

## Inputs

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
- Seed: `3519994490`
- W&B:
  https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw
- Output:
  `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/`
- Result docs:
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-warmstart-result.md`
  - `docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-training-result.md`
- Required local artifacts:
  - `python/wandb/run-20260522_141504-ud4c09jw/files/output.log`
  - `python/wandb/run-20260522_141504-ud4c09jw/files/wandb-summary.json`
  - `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/matrix_eval.json`
  - `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.json`
  - `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay`
  - `data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`

## Evidence

The direct widened-observation teacher is positive evidence for information
sufficiency, not for PPO discoverability. Its diagnostic used scripted direct
actions from the multi-enemy actor tokens and beat `weak_basic_v2` `10W/0L/0D`,
with mean score `9.20/0.00`, Team A hit/fire `0.09`, visible-fire rate `1.0`,
and objective_on_point `0.875`.

The W&B run completed all `100/100` PPO updates but failed every objective
check. Best eval at update 50 and final eval at update 100 were both
`0W/50L/0D`, score `0.00/37.00`, mean reward `-11.0`, and Team A hit/fire
`0.0`. Matrix eval was draw-only vs `noop`, `50/50` losses vs `weak_basic_v2`,
and `50/50` losses vs `basic`. Transfer summary was
`evidence_insufficient`.

The warm start did not initialize the new actor encoder path. The run log shows
`warm_start_migration=compatible_exact loaded=17`, while the missing keys
included the new `actor_entity_encoder`, `actor_grid_encoder`, and
`actor_fusion` parameters. The skipped unexpected keys were the old flat
`actor_embed.*` tensors. This means the old Phase 4 checkpoint preserved only
compatible same-shape weights outside the new actor observation encoder, while
the load-bearing widened actor front end began effectively random.

The PPO traces are consistent with a policy that did not form a useful action
mapping from the new tokens. Entropy stayed near `3.935` throughout, policy
loss was effectively zero, and training rollout binary-action mean stayed near
random around `0.49`. In greedy eval, Team A visible-fire rate and intentional
fire fraction were both `0.0`; Team B had visible-fire rate `1.0` and scored
`37.0`. The replay analyzer from the result doc matches this: greedy Team A
issued no fire commands and did no damage, while stochastic Team A fired
continuously but produced only `1000` centi-HP damage over five detected
episodes.

## Reconciliation

The apparent contradiction resolves as follows:

- The actor information bottleneck found in the previous flat-observation audit
  is fixed for direct scripted action selection. The currently visible enemy
  slots are present and sufficient for a hand-coded teacher to aim, fire, win
  fights, and hold the point.
- The PPO run did not inherit a usable actor representation for that widened
  observation. The compatible warm-start intentionally skipped the incompatible
  flat actor encoder, leaving the new entity/grid actor path to be learned from
  sparse contested 3v3 PPO reward at `1.0e-6` learning rate and only 100
  updates.
- The direct teacher therefore proves that the observation surface can support
  the behavior, but the neural policy was never bridged to that behavior. This
  is a trainability and representation-initialization failure, not evidence
  that multi-enemy actor information is useless.

## Decision

Do not retry `phase4_mappo_multi_enemy_actor_obs_v1.yaml` unchanged. Also do
not run a longer or coefficient-only PPO variant as the next step. The negative
run already shows that widened observation plus random new actor front end plus
plain PPO is not enough under this budget.

The next bounded design should be an opt-in supervised bridge for the widened
actor observation, using the already validated `multi_enemy_visible` direct
teacher as labels before PPO. The bridge should have its own pre-PPO gate and
should stop before PPO if the cloned neural policy cannot reproduce contested
hit/fire and objective pressure. This is a new supervised training path, so it
requires user approval before implementation.

## Approval Request

Approve one bounded opt-in Phase 4 multi-enemy supervised bridge assignment:

- Add a config-only or narrowly implemented pre-PPO behavior-cloning/rehearsal
  path for `actor_obs: multi_enemy_entity_grid` using the existing
  `multi_enemy_visible` teacher labels for movement, aim, and fire.
- Keep rewards, sim rules, tick pipeline, action semantics, replay format,
  phase-gate thresholds, and existing W&B metric schema unchanged.
- Do not add action-space-facing target fields.
- Keep the feature opt-in and off by default.
- Add or reuse focused tests for shape compatibility, import boundaries, and
  no hidden-enemy actor leakage.
- Before any PPO updates, run a direct neural-policy gate against
  `weak_basic_v2`; require nonzero Team A firing, useful hit/fire, objective
  pressure, and nonzero score, otherwise stop with `NOT_REACHED`.
- If that pre-PPO gate passes, allow exactly one separate bounded W&B training
  run assignment to collect normal Phase 4 W&B, matrix, replay, and gate
  evidence.

## Decision Status

`IMPLEMENTED_PRETRAINING_NOT_RUN`

Audit/design is complete. No training was launched. The proposed next step
requires explicit user approval.
