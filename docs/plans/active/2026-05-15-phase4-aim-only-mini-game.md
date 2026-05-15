# Phase 4 Aim-Only Mini-Game Plan

## Context

Escape Protocol Sections 5.1, 5.2, and 5.3 have each been falsified as
isolated fixes. The latest invalid-fire mask probe showed
`fire_valid_fraction = 0.9994`, so wasted fire on invisible enemies is not the
dominant bottleneck.

Do not create another Phase 4 hyperparameter, opponent, entropy, auxiliary
aim, or mask variant from this plan.

## Goal

Implement Escape Protocol Section 5.4 support: a Phase 4-compatible aim-only
mini-game that keeps the actor observation/action/checkpoint interface stable
while giving direct reward for mapping visible enemy relative position to
`aim_delta` plus `primary_fire`.

## Scope

- Add a synthetic `Phase4AimOnlyMappoEnv` with `(3, 31)` actor obs, `(3, 6)`
  actions, and a `(135,)` critic obs writer.
- Route Phase 4 configs with `env.mini_game: aim_only` to the synthetic env.
- Preserve checkpoint compatibility with full Phase 4 MAPPO actors.
- Add focused tests for observation shape, hit/miss reward, critic obs, and
  registry routing.

## Non-Goals

- No full 3v3 config in this implementation task.
- No long PPO benchmark in this implementation task.
- No C++ sim, reward, action-schema, replay-format, opponent, or observation
  manifest changes.

## Verification

- `cd python && .venv/bin/pytest tests/test_phase4_aim_only_mappo.py -q`
- `cd python && .venv/bin/pytest tests/test_phase_registry.py::test_phase4_smoke_config_builds_mappo_config tests/test_mappo_warm_start.py::test_mappo_warm_starts_from_init_checkpoint -q`
- `cd python && .venv/bin/ruff check envs/phase4_aim_only_mappo.py envs/__init__.py train/phases.py tests/test_phase4_aim_only_mappo.py`
- `git diff --check`

## Implementation Result

Implemented code support only:

- `Phase4AimOnlyMappoEnv` emits the Phase 4 flat actor shape `(3, 31)`, accepts
  `(3, 6)` actions, and writes a `(135,)` critic observation.
- The mini-game gives direct per-agent reward for matching `aim_delta` to a
  visible target angle and holding `primary_fire`.
- Phase 4 configs can opt in with `env.mini_game: aim_only` and optional
  `env.mini_game_config`.
- Focused tests cover hit/miss reward, observation shape, critic obs, and
  registry routing.

No Phase 4 experiment config or long PPO run was created in this
implementation task.
