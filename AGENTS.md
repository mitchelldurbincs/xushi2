# xushi2 Agent Instructions

## Project identity

xushi2 is a deterministic 2D 3v3 control-point hero-shooter simulator for multi-agent reinforcement learning.

The algorithm is **MAPPO**, not MAPO.

The project already has:
- C++20 simulation core (`src/sim/`)
- pybind11 Python bindings producing `xushi2_cpp` (`src/python_bindings/`)
- Python env wrappers (`python/xushi2/`), trainers (`python/train/`), per-phase envs (`python/envs/`)
- raylib viewer (`src/viewer/`) and scripted bots (`src/bots/`)
- phase-driven experiment configs under `experiments/configs/`
- W&B integration, TensorBoard support, replay tooling
- pytest + GoogleTest, CI, determinism tests, golden-replay CI

Do not reinvent experiment tracking, baseline config structure, phase metadata, runner scaffolding, or replay format unless a card explicitly asks for it.

## Source of truth for design

Before changing anything load-bearing, read the relevant doc:
- `docs/game_design.md` — game rules, tick pipeline, hero kits
- `docs/rl_design.md` — algorithm, actor/critic, reward, curriculum
- `docs/observation_spec.md`, `docs/action_spec.md`, `docs/replay_format.md`, `docs/determinism_rules.md` — interface contracts
- `docs/coding_philosophy.md` + `docs/standards/{cpp_determinism_checklist,python_training_checklist}.md` — review gates
- `docs/plans/active/` — currently driving plans; `docs/plans/README.md` lists the canonical ones
- `docs/journal/reinforcement_learning_journal.md` — running experiment log

If a doc and the code disagree, ask. Do not silently pick one.

## Experiment identity

Every experiment result must be tied to:

```text
git_commit + phase_config_path + seeds + W&B run URL + replay/artifact paths
```

Do not treat a result as gate evidence unless commit, config path, and seeds are all known.

## Standard commands

Use the Makefile wrappers when possible. Raw forms are listed in the README.

```bash
# Build the C++ sim and Python extension
make build-cpp
make py-install

# Tests
make test-cpp                    # ctest under build/
cd python && pytest              # Python tests; pytest is configured to run from python/

# Benchmarks
make bench-cpp                   # build C++ benchmarks
make run-bench                   # run all C++ benchmark binaries
make bench-viewer                # viewer regression (15% tolerance, see README §"Viewer benchmark")
xushi2-bench --target mappo --config experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml \
    --warmup-iterations 2 --measured-iterations 10 --output json

# Quality
make format
make lint
```

These commands assume a POSIX-like shell (Linux/macOS/WSL). On native Windows,
prefer the configured Python environment and PowerShell-friendly commands
(`py -3.13 -m pytest`, CMake directly, or the project-specific approved test
command) rather than fighting shell activation syntax.

## Standard training run

Two equivalent forms. Pick one and stick to it:

```bash
# Form A (from repo root, requires `make py-install` to expose the console script)
xushi2-train --config experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml

# Form B (from python/, module entrypoint — note the relative path)
cd python && python -m train.train --config ../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml
```

Real Phase 4 config layout (do not invent paths):

```text
experiments/configs/phase4/
    smoke/      phase4_mappo_smoke.yaml, phase4_mappo_async_smoke.yaml
    baseline/   phase4_mappo_basic.yaml, phase4_mappo_basic_3x_shaping.yaml
    probe/      phase4_mappo_noop_probe.yaml, phase4_mappo_objective_probe.yaml
    legacy/     archived; do not use for new runs
```

Other current config roots live alongside Phase 4: `experiments/configs/phase3/{smoke,baseline,probe,legacy}/`, `experiments/configs/phase11/probe/`, `experiments/configs/runtime/`, and top-level smoke/probe YAMLs such as `phase0_determinism.yaml`, `phase1b_env_smoke.yaml`, `phase2_memory_toy.yaml`, `phase5_entity_attention_probe.yaml`, `phase6_entity_grid_probe.yaml`, `phase7_*_fog_probe.yaml`, `phase8_random_map_probe.yaml`, `phase9_snapshot_probe.yaml`, and `phase10_target_slot_probe.yaml`.

## W&B

W&B is the source of truth for metrics and curves. Completion summaries should include the W&B run URL when available.

The trainer does not currently print the run URL to stdout in a parseable way. Until that lands, either (a) read it from `wandb.run.url` after `wandb.init` if you are in-process, or (b) parse the run working directory's `wandb/latest-run/files/wandb-metadata.json` after the run starts. The path is cwd-dependent: runs launched from `python/` write under `python/wandb/`, while runs launched from the repo root write under `wandb/`. Do not fabricate or guess URLs.

Do not build a parallel experiment tracker.

## Replays and viewer

Replays are first-class artifacts. Every benchmark should capture replay paths when available or explicitly state that no replay was produced.

Viewer fixtures and baseline: `data/benchmarks/viewer/`. See `README.md` §"Viewer benchmark" for the 15% tolerance band rule.

Use the viewer/replay path for behavior inspection, reward-hacking checks, and human judgment before claiming a phase gate is cleared.

## Phase gates

Phases are gates. Do not proceed past a phase until it produces stable, interpretable behavior.

For a phase-gate decision, collect:
- commit
- config path
- seeds
- W&B run URL
- replay path
- relevant metrics (anchored vs scripted-bot baselines, not just self-play win rate)
- human/viewer inspection notes when subjective judgment is required
- decision: cleared, not cleared, blocked, or evidence insufficient

Self-play win rate trends to 50% by construction and is not a gate metric on its own (`rl_design.md` §11).

## Default current focus: Phase 4

Unless a Kanban card says otherwise, the current focus is Phase 4.

Phase 4 is recurrent MAPPO, 3v3 Ranger, centralized critic, fixed map. All six slots run Ranger so Phase 4 isolates the multi-agent / CTDE delta from hero diversity. As-built (2026-07+), Phase-4 training uses `multi_enemy_entity_grid` observations, snapshot opponents, and the redefined 600t gate — see docs/rl_design.md's as-built divergence note and the RL journal.

Phase 4 work is split across several active sub-plans in `docs/plans/active/`. The canonical current driver is listed in `docs/plans/README.md` — read that first; do not pick a sub-plan by guessing.

## Done vs blocked

A bad metric result is **done**, not blocked.

Done examples:
- run completed but reward is poor
- run completed but phase gate is not cleared
- replay shows bad behavior
- experiment disproves the hypothesis

Blocked examples:
- build failed
- `xushi2_cpp` import failed
- config missing or path wrong
- W&B auth failed
- process crashed before any usable metrics emitted
- executor host slept/disconnected
- timed out before usable evidence
- requires human design or replay judgment
- requires unauthorized code change

## Human inspection

When a phase decision requires subjective replay/viewer judgment, do not silently complete. Block with a reason starting:

```text
HUMAN_INSPECTION_REQUIRED
```

Include:
- W&B run URL
- replay artifact path
- viewer command if known
- exact questions for the user
- what kind of comment the user should leave (e.g. `approved: ...` / `rejected: ...`)
- instruction to unblock after commenting

This is a normal lifecycle outcome, not a failure.

## Code-change rules

Keep diffs narrow. Do **not** silently change:
- game rules or tick pipeline ordering
- reward functions or shaping caps
- observation or action spaces
- phase configs
- MAPPO/PPO training logic
- recurrent hidden-state handling
- determinism behavior (RNG, ordering, tie-breaks, integer-tick math)
- replay format
- W&B metric names or schema

If any of these change, call it out explicitly in the card summary and flag it for review.

### Actor/critic separation (load-bearing)

`rl_design.md` §3 calls this "the single most important piece of infrastructure in the project."

> No function that iterates over hidden enemies or full state may be called by `actor_obs_builder`.

A leak silently invalidates the research contribution. Any change to `python/xushi2/{entity_obs,partial_obs,multi_enemy_obs,obs_manifest}.py`, `python/train/mappo_model.py`, or the C++ observation builders requires the existing leak tests to be run and the test list to be cited in completion metadata.

## Testing priorities

Prefer focused tests for:
- deterministic sim behavior (same seed → same trajectory)
- config loading
- env reset/step
- reward calculation and shaping clip
- observation/action shapes and dtypes
- **actor/critic leak prevention**
- replay round-trip
- W&B-disabled mode
- runner / completion behavior

Do not run long training jobs from code-editing cards unless the card explicitly assigns that.

## Completion metadata convention

Workers complete with `kanban_complete(summary=..., metadata=...)`. Use a machine-readable shape:

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

Workers may add task-specific fields (executor, wall time, exit code, etc.). Keep secrets, raw logs, API keys, tokens, and large transcripts out of metadata — store paths and one-line summaries instead.

## Tiered strictness (from `docs/coding_philosophy.md`)

- **Tier 0** — sim core and Python boundary (`src/sim/`, `src/python_bindings/`, `python/xushi2/env.py`, obs builders, replay): strict determinism, bounded loops, no post-init allocation, `Result<T>`, explicit validation, leak tests must pass.
- **Tier 1** — viewer, tools, bots: practical ergonomics, never at the cost of sim correctness.
- **Tier 2** — Python trainer/eval: practical ML, but shape/dtype/NaN checks, seed logging, replay export, no information leakage.

Apply the right tier when reviewing a change.

## Plans workflow

- Active plans live in `docs/plans/active/`; results and superseded plans in `docs/plans/archive/` as `YYYY-MM-DD-topic-result.md`.
- New cards that drive multi-step work should reference an active plan or create one.
- When a plan is superseded or completed, move it to `archive/` and link the replacement.
