# Goal: Simplify xushi2 Project Surface Area

## Purpose

Simplify xushi2 so the current RL work is easier to understand, run, test, and
extend without weakening the deterministic sim, replay, reward, observation/action
contracts, or actor/critic separation.

Bias toward removing accidental complexity in Python training/runtime, configs,
tests, docs, and local workflow. Do not simplify by deleting active research
capability, changing game semantics, or hiding important experiment metadata.

Phases still matter for RL progress tracking. Runtime behavior should continue
to be selected by explicit runtime/task/env/learner specs.

## First Reads

Before changing code, inspect the current state:

1. `docs/architecture/python_layers.md`
2. `docs/architecture/runtime_specs.md`
3. `docs/rl_design.md`
4. `docs/observation_spec.md`
5. `docs/action_spec.md`
6. `docs/coding_philosophy.md`
7. `docs/standards/python_training_checklist.md`
8. `python/train/mappo_eval_checkpoint.py`
9. `python/train/runtime_specs.py`
10. `python/train/checkpoint_runtime.py`
11. `python/train/phases.py`
12. `python/envs/__init__.py`
13. `experiments/configs/`

Also inspect `git status` first. There may be unrelated local edits, especially
`GOAL_INSTRUCTIONS.md`.

## Simplification Priorities

Work in small, reviewable slices. Prefer simplifications that reduce the number
of concepts a future agent must hold in memory.

High-value targets:

1. **Split large orchestration modules.**
   - `python/train/mappo_eval_checkpoint.py` currently mixes runtime context,
     pretrain hooks, training loop orchestration, checkpoint writing, eval, and
     matrix hooks.
   - Split by responsibility without changing training behavior or metric names.
2. **Finish moving runtime behavior behind public runtime APIs.**
   - Replay, eval, benchmark, checkpoint, and training helpers should consume
     `resolve_runtime_spec`, `checkpoint_runtime`, or neutral public env
     factories.
   - Avoid new direct use of `train.phases` or numeric phase branches outside
     legacy compatibility.
3. **Reduce phase-named production concepts.**
   - Keep phases as experiment metadata.
   - Prefer capability names for reusable env concepts, helpers, docs, and tests.
   - Keep aliases only when they support current configs/tests/checkpoints.
4. **Consolidate config surface.**
   - Make `experiments/configs/runtime/` the preferred home for new runnable
     configs.
   - Keep only clearly active smoke/baseline/probe configs outside archive.
   - Move stale or superseded configs to archive/legacy with notes.
5. **Split mixed fast/slow tests.**
   - Separate config/unit tests from one-update training, BC, replay, and eval
     smoke tests.
   - Keep representative slow tests, but make them explicit and easy to run.
   - Avoid hiding slow behavior inside broad unit-test files.
6. **Make current checkpoint schema explicit.**
   - New checkpoints should save a clear current schema using `experiment`,
     `learner`, `env`, and model config.
   - Old checkpoint support can remain thin and best-effort. Do not over-optimize
     for historical compatibility.
7. **Clean local/generated workflow noise.**
   - Ensure caches, build outputs, runs, W&B output, egg-info, benchmark output,
     and replay output are ignored or have a documented cleanup path.
   - Add or improve a PowerShell-friendly cleanup command only if it avoids
     deleting tracked or user-authored artifacts.

## Non-Negotiables

- Keep MAPPO spelled MAPPO.
- Do not change game rules, reward functions, observation layouts, action
  semantics, replay format, deterministic sim behavior, or W&B metric schemas
  unless the change is explicitly required and called out.
- Preserve actor/critic separation. No actor-observation path may call
  hidden-enemy/full-state helpers.
- Keep phase labels for experiment progress, docs, W&B tags, config
  organization, and journal lineage.
- Prefer compatibility with current configs and current checkpoints. Do not
  spend large effort preserving every older experimental shape unless it is
  still actively used.
- Keep diffs narrow. Do not do broad rename churn unless the renamed surface is
  actively being simplified.

## Suggested First Slice

A good first simplification slice:

1. Split `python/train/mappo_eval_checkpoint.py` into smaller modules:
   - runtime context / config normalization;
   - pretrain and composition hooks;
   - checkpoint save/load payload helpers;
   - post-training eval/matrix hooks;
   - top-level `train_mappo_from_config`.
2. Keep public imports stable through `train.mappo`.
3. Add focused tests for the extracted helpers.
4. Run existing MAPPO public API, runtime spec, matrix eval, replay dump, and
   train dispatch tests.

## Suggested Follow-On Slices

1. Split `tests/test_phase_registry.py` into:
   - legacy config compatibility;
   - config compactness;
   - slow MAPPO train smoke;
   - slow BC/pretrain smoke.
2. Move legacy phase-to-runtime mapping out of `train.phases` if practical, or
   document it as a residual compatibility adapter.
3. Convert the most-used active configs to explicit runtime YAMLs.
4. Archive stale probe/legacy configs that are not part of current Phase 4 work.
5. Rename neutral public helpers where aliases already exist, then update
   production call sites.
6. Add or update `.gitignore` and cleanup tooling for generated local outputs.

## Verification Commands

Use PowerShell-friendly commands on native Windows.

Focused checks for Python simplification slices:

```powershell
cd python
py -3.13 -m pytest tests/test_runtime_specs.py tests/test_train_dispatch.py tests/test_mappo_public_api.py -q
py -3.13 -m pytest tests/test_benchmark_run.py tests/test_mappo_matrix_eval.py tests/test_phase4_checkpoint_replay_dump.py -q
py -3.13 -m scripts.check_import_boundaries
```

If observation builders or actor/critic paths are touched:

```powershell
cd python
py -3.13 -m pytest tests/test_bindings_obs.py tests/test_obs_manifest.py tests/test_phase5_entity_obs.py tests/test_phase6_grid_obs.py tests/test_phase7_partial_obs.py tests/test_phase10_target_slot.py -q
```

If sim or C++ observation code is touched:

```powershell
cmake --build build --config Release
ctest --test-dir build -C Release -R "ActorLeak|ActorObs|CriticObs|ObsDims|ObsUtils|Determinism|GoldenReplay" --output-on-failure
```

For slow-test restructuring, run both the fast replacement tests and at least
one representative slow smoke before claiming completion.

## Completion Criteria

This simplification goal is complete when:

- the chosen simplification slice reduces active code/test/config complexity in
  a measurable way;
- current runtime config paths still work;
- current phase configs used for Phase 4 still work unless explicitly migrated;
- public imports used by current tests/scripts remain stable or have clear
  aliases;
- no game/reward/observation/action/replay semantics changed unexpectedly;
- slow tests are either split or clearly documented;
- import-boundary checks pass;
- focused tests and representative smoke tests pass;
- remaining legacy complexity is documented as residual risk rather than hidden.

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

## Good `/goal` Prompt

Use this prompt:

```text
Use GOAL_INSTRUCTIONS.md as the active goal. Simplify xushi2's active project
surface without changing game, reward, observation, action, replay,
determinism, or W&B metric semantics. Focus on reducing Python
training/runtime, config, test, and workflow complexity now that runtime
behavior is selected by explicit specs. Prefer current config/checkpoint
compatibility over broad historical compatibility, keep phases as experiment
progress metadata, split large orchestration and mixed slow-test files where
practical, archive or clarify stale configs, keep public runtime APIs neutral,
and do not mark the goal complete until focused tests, representative smoke
tests, and the import-boundary checker pass or any remaining failure is clearly
unrelated.
```
