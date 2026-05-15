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
