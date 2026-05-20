# Goal: Remove Hardcoded Phase Runtime Architecture

## Purpose

Refactor the Python runtime architecture so experiment phases remain useful
metadata, documentation, and config organization, but environments, learners,
dispatch, and reusable runtime helpers are not hardcoded around `phaseX`.

The current project uses phase labels in two different ways:

1. **Good use:** experiment identity, config paths, W&B tags, phase-gate history,
   docs, and journal lineage.
2. **Problem use:** runtime architecture, env class names, trainer routing,
   observation/action dimensions, feature toggles, and env wrapper inheritance.

Preserve the first use. Unwind the second use.

## First Reads

Before changing code, read these in order:

1. `docs/architecture/python_layers.md`
2. `docs/rl_design.md`
3. `docs/observation_spec.md`
4. `docs/action_spec.md`
5. `docs/coding_philosophy.md`
6. `docs/standards/python_training_checklist.md`
7. `python/train/phases.py`
8. `python/train/train.py`
9. `python/train/mappo_rollout_trainer.py`
10. `python/train/mappo_eval_checkpoint.py`
11. `python/envs/__init__.py`

Also inspect the current worktree before relying on this file. There may be
unrelated local changes.

## Current Scope From Initial Audit

The main hardcoded runtime phase coupling is in:

- `python/train/phases.py`
  - central `PHASE_REGISTRY`;
  - `_make_phaseN_env` factories;
  - `_phaseN_env_bundle` config extractors;
  - `resolve_phase(config)` returning learner/env dimensions from numeric phase.
- `python/train/train.py`
  - dispatches phase 0, phases 2-3, and phases 4-11 explicitly;
  - calls `train_phase4_from_config` for all MAPPO phases;
  - formats user-facing banners by numeric phase groups.
- `python/train/mappo_rollout_trainer.py`
  - `make_mappo_config` gets `obs_dim`, `critic_obs_dim`, `n_agents`, action
    dimensions, and target-action support from `resolve_phase`.
- `python/train/mappo_eval_checkpoint.py`
  - `build_runtime_context` depends on `resolve_phase`;
  - exported function name `train_phase4_from_config` now handles more than
    Phase 4;
  - composition/BC paths still contain Phase 4-specific helper names.
- `python/train/composition_rehearsal.py`
  - imports private `_make_phase4_env` from `train.phases`.
- `python/envs/`
  - env classes and modules are phase-named;
  - later envs wrap/import earlier phase envs, especially `Phase4MappoEnv`;
  - examples include Phase 5/6/7/9/11 using `Phase4MappoEnv`, Phase 8 using
    `Phase7FogMappoEnv`, and Phase 10 using `Phase8RandomMapMappoEnv`.
- `python/scripts/` and `python/eval/`
  - replay/eval helper names and modes still refer to phase-specific contracts.

Some phase references in docs, config filenames, W&B tags, tests, and journal
entries are acceptable and should remain unless they block the architecture.

## Important Existing Issue

At the time this goal was written, running the import-boundary checker failed
before it could evaluate boundaries because `python/tests/test_reward.py`
contained conflict markers around line 639:

the Git conflict marker for `HEAD`.

Before relying on `python -m scripts.check_import_boundaries`, inspect and fix
that file if the conflict markers are still present. Do not silently discard
user changes while resolving it.

## Architectural Direction

Move from a phase registry to an explicit runtime task/environment registry.

Target config concepts:

```yaml
experiment:
  phase: phase4              # optional metadata for docs/W&B/gates
  tags: [...]

learner:
  kind: mappo                # mappo, ppo_recurrent, scripted_determinism, etc.

env:
  kind: mappo_match          # mappo_match, memory_toy, aim_only, cap_duel, etc.
  actor_obs: flat            # flat, entity, entity_grid, etc.
  critic_obs: team_global
  team_size: 3
  learner_team: A
  opponent:
    kind: basic              # basic, noop, weak_basic_v2, snapshot, current
  features:
    fog: none                # none, team_shared, per_agent
    map_randomization: false
    target_slot: false
    current_selfplay: false
```

Exact field names may differ if the codebase has a better local pattern. The
important property is that runtime behavior is selected by explicit
capabilities, not `phase == N`.

Backward compatibility is required. Existing phase configs should keep working
through a compatibility adapter while new configs can use the task/env shape.

## Non-Negotiables

- Keep MAPPO spelled MAPPO.
- Do not change game rules, reward functions, observation layouts, action
  semantics, replay format, deterministic sim behavior, or W&B metric schemas
  unless the change is explicitly required and called out.
- Keep phase labels in experiment docs/config organization where useful.
- Do not remove existing phase configs or break old checkpoints without an
  explicit migration plan.
- Do not perform a broad rename-only churn pass before the runtime abstraction
  exists.
- Preserve actor/critic separation. No actor-observation path may call helpers
  that expose hidden enemies or full state.
- Keep diffs narrow and staged. This should be an incremental migration, not a
  one-shot rewrite.

## Recommended Migration Plan

### Step 1: Create Runtime Specs

Introduce a small explicit spec layer, likely under `python/train/` or
`python/xushi2/`, with dataclasses or typed dictionaries for:

- experiment metadata,
- learner spec,
- env spec,
- observation spec,
- action spec,
- opponent/self-play spec,
- map/fog/snapshot feature spec.

The spec should be constructible from:

1. existing phase configs,
2. new explicit task/env configs.

Keep `resolve_phase` temporarily as a compatibility adapter, but stop spreading
new call sites that depend on numeric phases.

### Step 2: Replace Trainer Dependence On Phase Numbers

Refactor learner entrypoints so they dispatch on `learner.kind` and env/task
capabilities:

- PPO recurrent path should receive an explicit `TaskSpec`.
- MAPPO path should receive explicit dimensions and env factory from the runtime
  spec.
- `make_mappo_config` should not need `phase in (4, 5, 6, 7, 8, 9, 10, 11)`.
- Rename or alias `train_phase4_from_config` to a neutral name such as
  `train_mappo_from_config`, while preserving old imports if tests or scripts
  depend on them.

### Step 3: Move Env Construction Out Of `train.phases`

Create an env registry/factory that maps explicit env specs to constructors.

The training layer should ask for "build env from spec", not import or call
`_make_phase4_env`. Composition rehearsal should also use the public factory.

Avoid adding new direct imports from `train` into phase-private env modules.

### Step 4: Rename Or Wrap Reusable Env Concepts

After the spec/factory exists, start moving phase-named envs toward capability
names. Suggested target names:

- `Phase4MappoEnv` -> `RangerMappoMatchEnv` or `FlatRangerMappoMatchEnv`
- `Phase4CurrentSelfplayMappoEnv` -> `CurrentSelfplayMappoMatchEnv`
- `Phase5EntityMappoEnv` -> `EntityObsMappoWrapper`
- `Phase6GridMappoEnv` -> `EntityGridObsMappoWrapper`
- `Phase7FogMappoEnv` -> `FogMappoMatchEnv` or `FogObsMappoWrapper`
- `Phase8RandomMapMappoEnv` -> `RandomizedMapMappoEnv`
- `Phase9SnapshotMappoEnv` -> `SnapshotOpponentMappoEnv`
- `Phase10TargetSlotMappoEnv` -> `TargetSlotMappoEnv`
- `Phase11CurrentSelfplayMappoEnv` -> `SixAgentCurrentSelfplayMappoEnv`

Use compatibility aliases during the migration so existing tests, checkpoints,
and scripts do not all have to change in one commit.

### Step 5: Keep Experiment Identity Separate

Add or preserve experiment metadata fields used for:

- W&B run names/tags,
- journal entries,
- config listing,
- gate decisions,
- phase-specific docs.

The phase label can still be logged. It just should not determine learner
shape or env behavior by itself after migration.

### Step 6: Update Tests And Boundaries

Add focused tests proving:

- legacy phase configs still resolve and train/import correctly;
- new explicit runtime configs resolve to the same env/learner specs;
- MAPPO dimensions come from runtime specs/env spaces rather than hardcoded
  phase numbers;
- `train` does not import phase-private env modules directly where the boundary
  checker forbids it;
- actor/critic leak tests still pass after any env/observation changes.

## Suggested First Slice

A good first implementation slice is:

1. Add a neutral runtime spec module.
2. Add a compatibility adapter from the current `PHASE_REGISTRY` entries into
   that runtime spec.
3. Change `make_mappo_config`, `build_runtime_context`, and benchmark scripts
   to consume the runtime spec instead of raw phase spec dictionaries.
4. Add one explicit non-phase config fixture in tests that is equivalent to the
   Phase 4 smoke config.
5. Keep all existing phase configs working.

This creates the architectural direction without forcing a full env rename in
the same patch.

## Verification Commands

Use PowerShell-friendly commands on native Windows.

Minimum focused checks after the first slice:

```powershell
cd python
py -3.13 -m pytest tests/test_train_dispatch.py tests/test_phase_registry.py tests/test_mappo_public_api.py tests/test_mappo_phase_env_parity.py tests/test_vector_env.py -q
py -3.13 -m scripts.check_import_boundaries
```

If observation builders or actor/critic paths are touched, also run:

```powershell
cd python
py -3.13 -m pytest tests/test_bindings_obs.py tests/test_obs_manifest.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py tests/test_phase7_partial_obs.py tests/test_phase10_target_slot.py -q
```

If C++ observation or sim code is touched, also run the relevant C++ tests:

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release -R "ActorLeak|ActorObs|CriticObs|ObsDims|ObsUtils|Determinism|GoldenReplay" --output-on-failure
```

Do not run long training jobs for this architectural goal unless a later card
explicitly asks for behavioral experiment evidence.

## Completion Criteria

This goal is complete when:

- runtime env/learner behavior can be selected without hardcoding a numeric
  `phase`;
- existing phase configs still work through compatibility;
- at least one explicit non-phase runtime config/test path works;
- MAPPO config construction no longer requires membership in a hardcoded
  phase range;
- public env construction goes through a neutral factory/spec layer;
- direct trainer imports of phase-private env modules are removed or isolated
  behind compatibility shims;
- docs explain the distinction between experiment phase metadata and runtime
  capability specs;
- focused tests and import-boundary checks pass, or any remaining failure is
  clearly documented as unrelated pre-existing worktree state.

## Completion Metadata

When reporting completion, include:

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

For this architecture goal, `wandb_run_url`, `replay_artifacts`, and
`viewer_command` are normally null unless the work unexpectedly includes
training/eval runs.

## Good `/goal` Prompt

Use this prompt:

```text
Use GOAL_INSTRUCTIONS.md as the active goal. Refactor xushi2 so phase labels
remain experiment metadata, but runtime env and learner behavior are selected
from explicit task/env/learner specs rather than hardcoded phase numbers. Work
incrementally, preserve existing phase configs through compatibility, avoid
changing game/reward/observation semantics, and verify with focused Python tests
plus the import-boundary checker.
```
