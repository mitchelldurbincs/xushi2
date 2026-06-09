# Phase 4 Auxiliary Aim Head Plan

## Context

Phase 4 is in the Escape Protocol draw basin. The latest opponent-design test,
`weak_basic_v1`, reached its configured stop point at update 500 with 50/50
draws, score 0/0, and kills 4.0/4.0. The stochastic replay shows agents fire
and move while firing, but they still do not convert combat into score.

The opponent and hyperparameter space is exhausted. Do not create another
Phase 4 hyperparameter or opponent variant from this plan.

## Goal

Implement Escape Protocol Section 5.1 as one isolated architecture change:
an auxiliary aim prediction head trained to predict the angle to a visible
enemy from the actor path.

## Scope

- Add an opt-in MAPPO config flag and coefficient for the auxiliary aim loss.
- Add a small prediction head to `MappoActorCritic` without changing action,
  observation, reward, replay, or game-rule schemas.
- Train the head during BC pretrain and PPO updates using existing actor
  observations as supervision.
- Log auxiliary aim loss and RMSE so a short diagnostic can verify whether
  the head learns below the Escape Protocol target.
- Add focused tests for config parsing, target extraction, and loss behavior.

## Non-Goals

- No new Phase 4 training config until the code path has a cheap diagnostic.
- No changes to C++ sim rules, reward shaping, observation fields, action
  fields, determinism, or replay format.
- No combined architecture changes such as per-action entropy or action masks.

## Initial Verification

- `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py tests/test_phase_registry.py::test_phase4_smoke_config_builds_mappo_config`
- A short BC-only diagnostic proving auxiliary RMSE is logged and can decrease.

## Follow-Up Gate

Only after the implementation diagnostic succeeds should a new Phase 4 config
be created. That config must include `metadata.hypothesis`,
`metadata.falsification_criteria`, and a cheap diagnostic result in the journal.

## Implementation Note

Implemented in commit draft following this plan:

- Aux head is opt-in via `ppo.aim_aux_coef`.
- Existing checkpoints without the head can warm-start aux-enabled models; only
  `actor_aim_aux_head.*` keys may be missing.
- Cheap BC-only diagnostic on the Phase 4 smoke env reduced fixed-batch
  auxiliary RMSE from `1.8184` to `1.3880` over 40 BC steps.
- Focused verification:
  `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py tests/test_phase_registry.py::test_phase4_smoke_config_builds_mappo_config tests/test_mappo_warm_start.py::test_mappo_warm_starts_from_init_checkpoint tests/test_mappo_loss_mask.py -q`
  and
  `cd python && .venv/bin/ruff check train/mappo_model.py train/mappo_rollout_trainer.py train/mappo_bc_pretrain.py train/mappo_eval_checkpoint.py tests/test_mappo_aux_aim.py`.

## Probe Result

`phase4_mappo_aux_aim_v1` ran to the configured 500-update stop point. The
auxiliary head learned the supervised target (`aim_aux_rmse` `1.7855` ->
`0.0076` during BC pretrain), but final eval remained 0/50 wins, 50/50 draws,
score 0/0, kills 1.0/6.0. Stochastic replay was dumped to
`data/replays/phase4_aux_aim_v1_ckpt0500_stochastic.replay`.

Conclusion: the isolated Escape Protocol 5.1 auxiliary aim head is falsified as
a Phase 4 fix. Do not continue with coefficient variants; use a different
Section 5 architecture intervention or escalate for review.
