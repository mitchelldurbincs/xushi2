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

## Probe Result

`phase4_mappo_aim_only_v1` ran to the configured 200-update stop point and
succeeded. The success threshold was `mean_team_a_kills >= 48` by update 200,
equivalent to at least 50% greedy hit rate across `3 agents x 32 decisions`.

The run crossed that threshold at update 80 with `64.66` hits and reached
`94.96 / 96` possible hits by update 200. W&B:
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/d6qgug61`.
Checkpoint: `runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt`.

Conclusion: Escape Protocol 5.4 is a positive diagnostic. The actor can learn
visible-target aim in isolation. The next isolated test is full 3v3 weak_basic
warm-started from the aim-only checkpoint, without combining auxiliary aim,
per-action entropy, or invalid-fire masking.

## Transfer Result

`phase4_mappo_aim_transfer_v1` tested that transfer step in the full weak_basic
3v3 objective environment. It ran to update 500 and failed: final eval was
`0/50` wins, `50/50` draws, score `0/0`, and kills `0.0/3.0`.

Replay: `data/replays/phase4_aim_transfer_v1_ckpt0500_stochastic.replay`.
W&B: `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/9n07ntl5`.

Autopsy: Team A still fired almost continuously (`0.9996`) and moved while
firing (`0.980`), but the synthetic aim skill did not transfer into full-env
kill or score conversion. Do not queue aim-transfer variants without a new
diagnostic that directly measures where the aim mapping is lost.

## Retention Diagnostic

Loaded the aim-only checkpoint, evaluated synthetic aim before and after the
same 500-step `walk_and_shoot` BC pass used by `aim_transfer_v1`.

- Before BC: `94.84/96` synthetic hits.
- Before BC in full weak_basic: `0/50` wins, score `0/7`, kills `0/0`.
- After BC: `0.02/96` synthetic hits.
- After BC in full weak_basic: `50/50` draws, score `0/0`, kills `6/5`.

Conclusion: standard full-env BC erases the synthetic aim mapping before PPO.
The next transfer design must protect the aim skill or change the BC target;
rerunning the same aim-transfer shape is invalid.
