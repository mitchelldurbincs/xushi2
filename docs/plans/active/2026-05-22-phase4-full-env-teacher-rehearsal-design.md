# Phase 4 Full-Env Teacher Rehearsal Design

Date: 2026-05-22

## Status

V1, v2, and v3 have all been implemented and run. Each returned `NOT_REACHED`
at the pre-PPO gate, so this document now serves as the design/result trail and
negative evidence record. Do not start another training run from this document
without a bounded assignment.

## Problem

Phase 4 still fails at the same composition point after the latest cap_duel v2
focus-fire probe:

- Objective-only/warm-start behavior can reach or contest the point.
- Isolated cap_duel v2 can learn honest kill-then-hold behavior.
- Existing focus-fire machinery can concentrate target labels.
- PPO-time cap_duel distillation can run cleanly.

But none of these mechanisms bind combat into full weak_basic_v2 3v3. The
latest negative result, `phase4_mappo_cap_duel_v2_focus_fire_v1`, reached
update 100 with `0W/50L/0D`, Team A score `0.00`, Team A hit/fire `0.0017`,
and no objective conversion.

## Non-Starters

Do not spend more runs on:

- composition-rehearsal length variants;
- cap_duel distillation coefficient-only variants;
- plain focus-fire or cap_duel v1 focus-fire repeats;
- cap_duel v2 focus-fire config-only variants;
- exact repeats of auxiliary aim, per-action entropy, invalid-fire mask,
  aim-transfer, mode-gated, or target-conditioned probes already falsified.

Do not change C++ sim rules, reward formulas, observation/action spaces,
determinism behavior, replay format, W&B existing metric schema, or phase-gate
thresholds as part of this plan.

## Hypothesis

The missing rung is not another isolated teacher. It is full-env decision
alignment: the learner needs supervised examples in the actual 3v3
weak_basic_v2 distribution where movement, aim, fire, and objective timing are
trained together from actor-visible observations.

The next structural intervention should use scripted or checkpoint teachers in
the full Phase 4 environment to produce joint action targets for the learner,
then gate the result before PPO starts.

## Proposed Mechanism

Add an opt-in full-env teacher rehearsal stage:

1. Run full Phase 4 weak_basic_v2 episodes from the student checkpoint.
2. For each learner slot, generate actor-visible teacher targets using an
   explicit scripted policy or a hybrid teacher:
   - movement: approach/hold objective when alive;
   - target choice: shared low-HP or objective-relevant visible enemy;
   - aim: turn toward selected visible target;
   - fire: fire only when selected target is visible and roughly aligned.
3. Train the existing actor heads on those labels before PPO:
   - MSE for `move_x`, `move_y`, and `aim_delta`;
   - BCE/KL for `primary_fire`;
   - reuse existing target-selection auxiliary loss if target-conditioned
     combat is enabled.
4. Gate the rehearsed checkpoint in full weak_basic_v2 before PPO:
   - Team A hit/fire must exceed `0.04`;
   - Team A objective contact/on-point must exceed `0.25`;
   - Team A must avoid `50/50` losses;
   - replay must show approach + aim/fire + point pressure rather than
     off-point spraying.

This differs from prior mini-game composition attempts because the supervised
data is collected in the real full 3v3 distribution, not in cap_duel or
combat_1v1. It differs from focus-fire because target choice is not the only
supervised behavior; movement and fire timing are trained in the same samples.

## Candidate Implementation Scope

Allowed in a future implementation card:

- new Python trainer helper, e.g. `python/train/full_env_rehearsal.py`;
- focused tests under `python/tests/`;
- one probe config under `experiments/configs/phase4/probe/`;
- new W&B metrics under a new prefix such as `full_rehearsal/*`;
- replay/eval diagnostics for the pre-PPO gate.

Disallowed without explicit approval:

- C++ changes;
- reward changes;
- observation/action schema changes;
- new action-space-facing target fields;
- modifying phase-gate thresholds.

## Suggested Config Shape

```yaml
run:
  full_env_rehearsal:
    enabled: true
    steps: 2000
    batch_size: 256
    teacher: scripted_objective_focus_fire
    target_conditioned_combat: true
    movement_coef: 1.0
    aim_coef: 1.0
    fire_coef: 1.0
    target_selection_coef: 0.5
    gate:
      episodes: 50
      min_team_a_hit_fire: 0.04
      min_on_point: 0.25
      max_losses: 49
```

First probe config name:

```text
experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml
```

## Required Tests

Future implementation must include focused tests proving:

- the feature is off by default;
- the scripted teacher emits finite, bounded actions with the existing Phase 4
  action shape;
- target labels use only actor-visible enemy fields;
- no hidden-enemy or critic-only state is read by actor-side rehearsal labels;
- one rehearsal update produces finite movement, aim, fire, and optional
  target-selection losses;
- the pre-PPO gate blocks PPO when full-env hit/fire or objective contact is
  below threshold.

Because the implementation would touch actor-side label generation and
possibly target-conditioned combat paths, run and cite the existing leak and
focus-fire tests in completion metadata.

## Decision Rules

- If the rehearsal gate fails before PPO, status is `NOT_REACHED`; do not
  launch PPO.
- If the rehearsal gate passes, run one bounded 100-update PPO probe against
  `weak_basic_v2` with the unchanged Phase 4 anchor-transfer gate.
- If update 50 has zero Team A score, zero wins, Team A hit/fire below `0.04`,
  and objective contact collapses below `0.25`, stop or let the single
  configured 100-update run finish once. Do not extend compute.
- If objective checks pass, require human replay inspection before claiming any
  Phase 4 progress.

## Open Questions

- Should the first teacher be purely scripted or a hybrid of scripted movement
  plus existing checkpoint logits?
- Should target-conditioned combat be enabled in v1, or should v1 avoid the
  existing target-conditioning head to reduce confounds?
- What full-env replay question best distinguishes honest objective pressure
  from off-point spray?

## Next Assignment

Audit and implement only the rehearsal teacher/gate helper and focused tests.
Do not run the long PPO probe until the pre-PPO rehearsal gate can be produced
and inspected.

## Implementation Status

2026-05-22: first implementation landed in the working tree.

- Added `python/train/full_env_rehearsal.py` with actor-observation-only
  scripted movement/aim/fire labels, supervised rehearsal loss/pretrain, and a
  pre-PPO gate writer.
- Added `maybe_run_full_env_rehearsal(...)` to `mappo_pretrain_hooks.py` and
  inserted it before composition/BC pretrain in `mappo_eval_checkpoint.py`.
- Added focused tests in `python/tests/test_full_env_rehearsal.py` and extended
  `python/tests/test_mappo_pretrain_hooks.py`.
- Added probe config
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`.

Verification:

- `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py -q`
  -> `9 passed`.
- `py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py -q`
  -> `31 passed`.
- `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py tests/test_phase7_partial_obs.py tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py -q`
  -> `70 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.

## V1 Probe Result

2026-05-22: first probe completed as `phase4_mappo_full_env_rehearsal_v1`.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`.
- Seed: `3519994490`.
- W&B:
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu`.
- Gate:
  `python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/full_env_rehearsal_gate.json`.
- Replays:
  `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay` and
  `data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay`.

The supervised labels converged mechanically by step 2000
(`loss=0.0012`, `move=0.0007`, `aim=0.0002`, `fire=0.0003`,
`target=0.0001`), but the pre-PPO gate returned `NOT_REACHED`:
`team_a_hit_fire=0.0005556`, `objective_on_point=0.015`, `losses=50`,
`wins=0`, score `0.00/37.00`. PPO was intentionally skipped.

Decision: stop v1 before PPO. The next assignment should audit replay/label
quality and propose a bounded v2 design. Do not rerun v1, extend rehearsal
length only, weaken the gate, or force PPO after a failed pre-PPO gate.

## V2 Probe Result

2026-05-22: v1 audit found a concrete aim convention bug. Actor aim vectors
are documented as `(sin theta, cos theta)`, but the v1 teacher computed target
angle as `atan2(y, x)`. The helper now uses `atan2(x, y)` for the target vector
and has a regression test.

The corrected probe completed as `phase4_mappo_full_env_rehearsal_v2`.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`.
- Seed: `3519994490`.
- W&B:
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr`.
- Gate:
  `python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/full_env_rehearsal_gate.json`.
- Replays:
  `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay` and
  `data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`.

V2 also returned `NOT_REACHED`: `team_a_hit_fire=0.0`,
`objective_on_point=0.01`, `losses=50`, `wins=0`, score `0.00/37.00`. Replay
analysis shows the fix helped aim alignment: Team A greedy mean
nearest-visible aim error dropped from v1 `2.5406` rad to v2 `0.6073` rad.
However, Team A still fired continuously and dealt `0` damage in both v2
replays.

Decision: stop corrected full-env scripted rehearsal v2 before PPO. The next
design needs a higher-fidelity full-env teacher that accounts for shooting
geometry and objective timing, or a scripted-action diagnostic proving that the
teacher can hit and hold in the same full-env distribution before training a
neural policy against those labels. Do not extend v2 length or force PPO.

## Direct Teacher Diagnostic

2026-05-22: added `python/scripts/diagnose_full_env_teacher.py` to run teacher
action streams directly in `Phase4MappoEnv`.

Results from the v2 config and seed `3519994490`:

- `actor_obs_scripted` vs `weak_basic_v2`, 50 episodes:
  `0W/0L/50D`, score `0.00/0.00`, Team A hit/fire `0.0`,
  visible-fire rate `1.0`, objective_on_point `0.0`.
- `actor_obs_scripted` vs `noop`, 10 episodes:
  `10W/0L/0D`, score `37.00/0.00`, objective_on_point `0.9333`.
- `cpp_basic` vs `weak_basic_v2`, 10 episodes:
  `10W/0L/0D`, score `12.70/0.00`, Team A hit/fire `0.0917`,
  objective_on_point `0.8667`.

Interpretation: the actor-observation-only teacher is not a viable training
target under contest. It can move/hold against no pressure, but cannot damage
or produce majority pressure against `weak_basic_v2`. The full-state
`cpp_basic` baseline proves the wrapper can support the desired behavior, but
it uses information outside the current flat actor observation. Current Phase 4
flat actor obs exposes only the counterpart enemy via `visible_enemy_1v1`; the
working scripted baseline chooses among all visible enemies.

Next: a bounded v3 should start from a higher-fidelity teacher and a preflight
diagnostic. Either use privileged training-time imitation from `cpp_basic` with
explicit no-inference-leak tests, or request approval to move Phase 4 to an
existing wider multi-enemy actor observation path. Do not alter sim rules,
reward formulas, action semantics, replay format, or phase-gate thresholds.

## V3 Preflight Implementation

2026-05-22: implemented an opt-in `teacher: cpp_basic` path in
`python/train/full_env_rehearsal.py`.

- The C++ `basic` bot is used only for training-time movement/aim/fire labels.
- Actor inference remains unchanged and consumes actor observations only.
- No C++, sim rule, reward, observation layout, action semantics, replay, or
  phase-gate threshold changes were made.
- Added config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`.
- Target-selection conditioning is disabled in v3 to avoid coupling the prior
  focus-fire head to privileged full-state labels.

Preflight diagnostic on the v3 config:

- Command:
  `py -3.13 -m scripts.diagnose_full_env_teacher --config ../experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml --episodes 10 --seed 3519994490 --teacher cpp_basic --output runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/cpp_basic_teacher_diagnostic.json`.
- Result: `10W/0L/0D`, score `12.70/0.00`, Team A hit/fire `0.0917`,
  objective_on_point `0.8667`.

Verification:

- `py -3.13 -m pytest tests/test_full_env_rehearsal.py tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q`
  -> `16 passed`.
- Broader focused suite including focus/fire/leak-adjacent, Phase 4 env, and
  matrix eval tests -> `76 passed`.
- `py -3.13 -m scripts.check_import_boundaries` -> PASS.
- Config smoke: `obs_dim=31`, `action_dim=6`, `target_selection_dim=0`.

## V3 Probe Result

2026-05-22: v3 completed as
`phase4_mappo_full_env_rehearsal_v3_cpp_basic`.

- Config:
  `experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`.
- Seed: `3519994490`.
- W&B:
  `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2`.
- Gate:
  `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/mappo/full_env_rehearsal_gate.json`.
- Replays:
  `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay`
  and
  `data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay`.

V3 returned `NOT_REACHED`: Team A hit/fire `0.0061111111 < 0.04`,
objective_on_point `0.0166666667 < 0.25`, `50/50` losses, `0` wins, and score
`0.00/37.00`. PPO was intentionally skipped.

Matrix eval for `ckpt_final.pt` was negative: draw-only vs `noop`, `50/50`
losses vs `weak_basic_v2`, and `50/50` losses vs `basic`.

Replay analysis shows a small amount of damage but no useful objective
conversion: greedy replay Team A hit/fire `0.0061111111`, `0` kills, score
`0.00/37.00`; stochastic replay Team A hit/fire `0.0116731518`, `1` kill,
score `0.00/35.10`.

Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md`.

Decision: stop this v3 privileged full-env rehearsal path before PPO. The next
step should not be another rehearsal-length or forced-PPO variant. The evidence
now points to an actor information/capacity decision: either approve a
load-bearing move to an existing wider multi-enemy actor observation path, or
run an offline audit proving a different no-observation-change teacher can be
represented by the current flat actor input.
