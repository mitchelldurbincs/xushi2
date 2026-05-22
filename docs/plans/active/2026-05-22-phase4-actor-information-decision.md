# Phase 4 Actor Information Decision

Date: 2026-05-22

## Status

Audit complete. User approval was received in-thread for one bounded opt-in
Phase 4 multi-enemy actor-observation ablation. The implementation/preflight
assignment completed without launching W&B training.

## Evidence

The recent full-env teacher path produced three negative pre-PPO results:

- V1 scripted actor-observation rehearsal:
  `docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-result.md`
- V2 corrected scripted actor-observation rehearsal:
  `docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v2-result.md`
- V3 privileged `cpp_basic` rehearsal:
  `docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v3-cpp-basic-result.md`

The direct teacher diagnostic is the key separator:

- `actor_obs_scripted` vs `weak_basic_v2`: `0W/0L/50D`, score `0.00/0.00`,
  Team A hit/fire `0.0`, objective_on_point `0.0`.
- `cpp_basic` vs `weak_basic_v2`: `10W/0L/0D`, score `12.70/0.00`, Team A
  hit/fire `0.0917`, objective_on_point `0.8667`.

Diagnostic result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-teacher-diagnostic-result.md`.

## Observation Audit

`docs/observation_spec.md` says Phase 4 is CTDE MAPPO with shared actor
weights, but the actor obs remains the 31-float Phase-1 layout. The Phase-4
critic is widened to 135 floats; the actor is not.

The C++ actor builder enforces this directly:

- `src/sim/src/actor_obs.cpp` writes the enemy block only through
  `obs_utils::visible_enemy_1v1(...)`.
- `src/sim/src/obs_utils.cpp` maps 3v3 actor enemy observation to the
  counterpart slot with `counterpart_enemy_slot(viewer_slot)`.

So, in 3v3 Phase 4, each actor can condition on only one enemy slot even when
other enemies are visible and strategically relevant.

Existing wider actor-side machinery exists, but it is not the current Phase 4
actor surface:

- `python/xushi2/multi_enemy_obs.py` can build self, three enemy tokens, and an
  entity grid from flat actor obs plus visibility-gated enemy state.
- `python/xushi2/snapshot_policy.py` uses that path for Phase 7+ entity-grid
  snapshot conversion when `entity_token_count > 3`.

Using this in Phase 4 would be a load-bearing observation/model change and
must not be slipped in as another training variant.

## Answer To Audit Questions

Can the current flat Phase 4 actor observation represent direct `cpp_basic`
behavior?

Not faithfully. Direct `cpp_basic` wins by selecting among all visible enemy
slots. The current actor input exposes only the counterpart enemy slot. A
policy cannot imitate target switches to non-counterpart enemies from features
it never receives.

Is there a no-observation-change teacher worth trying next?

Not without a direct diagnostic first. The only no-observation teacher tried
directly, `actor_obs_scripted`, failed to damage or pressure `weak_basic_v2`.
Any future no-observation teacher must first run as direct actions in the full
environment and show wins or score against `weak_basic_v2`; otherwise training
against it is not justified.

What is the smallest load-bearing change to request approval for?

An opt-in Phase 4 actor-observation ablation that exposes all visible enemy
slots as masked entity tokens, without changing sim rules, rewards, action
semantics, replay format, or phase-gate thresholds.

The safer implementation direction is a visibility-gated actor observation path
whose tests prove hidden enemies cannot affect actor-visible values. Reusing
the existing multi-enemy Python adapter may be acceptable for a bounded probe
only if the approval explicitly allows it and leak tests cover the critic-based
conversion boundary.

## Required Tests If Approved

Any implementation that changes actor observation or model input must run and
cite:

- C++ actor leak and obs tests:
  `./build/tests/test_actor_leak`,
  `./build/tests/test_actor_obs`,
  `./build/tests/test_critic_obs`,
  `./build/tests/test_obs_dims`.
- Python observation/import/focus tests:
  `py -3.13 -m scripts.check_import_boundaries`,
  `py -3.13 -m pytest tests/test_phase7_partial_obs.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py -q`.
- Phase 4 env/model shape tests:
  `py -3.13 -m pytest tests/test_phase4_mappo_env.py tests/test_mappo_pretrain_hooks.py tests/test_mappo_focus_fire.py -q`.
- New tests proving hidden or non-visible enemies do not change actor-visible
  multi-enemy fields, visible enemy slots do change, masks are correct, and
  Team A/Team B frame conventions match existing actor obs.

## Recommended Next Assignment

Launch one separate bounded W&B training run for:

`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`

Preflight result:
`docs/plans/archive/2026-05-22-phase4-multi-enemy-actor-obs-preflight-result.md`.

The run must collect W&B URL, seed, gate/matrix artifacts, and replay paths.
Do not change rewards, phase gates, sim rules, action semantics, replay format,
or existing W&B metric schema as part of that run.
