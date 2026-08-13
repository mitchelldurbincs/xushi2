# Simplicity, Readability & Code Organization Audit — xushi2

Scope: full repo (~45k lines of source: ~12k C++ across 101 files, ~10k `python/train`, ~7.3k `python/xushi2` + `python/envs`, ~3.5k scripts/eval, ~11k Python tests). No changes made — audit only. Churn data from git history (Apr–Aug 2026) used to weight findings toward frequently modified code.

---

## 1. Overall Assessment

**Verdict: Mixed — with an unusually wide spread.** The C++ sim core on its own would rate Excellent; the Python training stack's experiment-accretion layer rates Poor in places. Most code sits in between.

- **Readability:** Generally good at the statement level. The comment culture is exceptional — non-obvious code routinely cites the run, date, or journal entry that motivated it (`reward.py:348`, `mappo_rollout_trainer.py:105`, `tests/replay/test_golden_replay.cpp:25`). The "why" is unusually well preserved. The problem is almost never a confusing line; it is the 300–770-line function bodies and the number of near-copies of the same idea.
- **Simplicity:** The core abstractions are sound and mostly earn their keep (tick pipeline, obs manifest, orchestration protocol, reward-feature view). The complexity that hurts is *accreted duplication*: the same rollout loop written 5 times, the same supervised-pretrain driver written 4–5 times, PPO loss written twice (the shared one dead in production), sim validation written twice (C++ sim vs bindings), the observation builder written twice (C++ and Python, pinned by parity tests), `clamp01` written 4 times, `arena_center` 5 times.
- **Navigation:** Good. The three-layer Python contract (`xushi2` → `envs` → `train`) and the `src/sim/src` vs `src/sim/src/internal` split make "where would this live?" answerable. But the *config → env* path crosses 18 modules and contains **two parallel resolution pipelines**, one of which (the documented-canonical one) is used by 1 of 78 configs.
- **Unnecessary complexity:** Moderate. Mostly dead or vestigial abstractions (`Result<T>`, `FixedVector`, `mappo_phase_common.py`, `benchmark_writer.cpp` — all zero-use) and redundant layers (three pure pass-through stage wrappers, hand-written setter forwarding beside a working generic dispatcher).
- **Boundary health:** The declared boundaries are good; enforcement is decaying. `xushi2/vector_env.py:728` imports `envs.runtime_factory` in violation of the layer contract; the `check_import_boundaries.py` enforcement script is **not wired into CI**; `bots/src/internal/behavior_primitives.cpp` includes sim's private internals via `../../../` path traversal.
- **Biggest sources of cognitive load:** (1) the dual config→env resolution pipeline; (2) the supervised pretrain/rehearsal/distill file family (~2,600 lines, 4–5 structural clones); (3) `_update_full_rollout` (298 lines) and `evaluate_mappo` (374 lines, 10 levels deep); (4) `MappoEvalStats`'s 62 fields restated in 5+ places; (5) `module.cpp`'s single 769-line function with duplicated validation.

A root cause worth naming: **AGENTS.md's "keep diffs narrow / never silently change behavior" norm, combined with heavy experimentation, has made copy-paste the locally rational move.** Copying `build_critic_obs` into a new phase env is a narrower diff than refactoring the env that trains today. Each copy was individually defensible; collectively they are the main tax on the codebase.

---

## 2. Biggest Simplicity Problems

1. **Two parallel config→env resolution pipelines, with the canonical one at ~1% adoption.** `runtime_specs.py` supports an explicit spec (`env.kind` + `learner.kind`) and a legacy phase path (`phase: 4` → `phases.PHASE_REGISTRY`). 76/78 configs use legacy; 1 uses explicit. The two paths duplicate helpers (`_resolve_seed_base` byte-identical in both files; `_extract_base_env_cfg` ≈ `_base_env_cfg`), build different `functools.partial` shapes from the same YAML, and diverge observably: `vector_env: sim_pool` gates on `fn.func is make_mappo_match_env` and **fails for every `phase:`-style config**. On top of this, `resolve_runtime_env_factory` runs **three times per training start** (train.py, mappo_eval_checkpoint.py, mappo_runtime_context.py) with two results discarded.

2. **The supervised pretrain family is a copy-paste lineage.** `full_env_rehearsal.py` (758), `mappo_bc_pretrain.py` (541), `composition_rehearsal.py` (435), `cap_duel_distill.py` (399), `mappo_pretrain_hooks.py` (468): the Adam/step/clip/log driver loop is written 4–5×; the move/aim/fire supervised loss is composed 3× (each redeclaring `_MOVE_ACTION_INDICES` etc.); `_assert_teacher_compatible` and `_teacher_policy_targets` exist twice (`cap_duel_distill` *already imports* from `composition_rehearsal` but redefines these); the two BC functions (124 and 168 lines, 9 and 12 params) differ by a 144-line diff that is mostly indentation shift; `_masked_mean` is defined twice.

3. **PPO loss exists twice; the shared one is dead.** `losses.compute_ppo_loss` (with `PpoLossResult`) is imported only by its own test (`test_ppo_shared_loss.py`, tests literally named `*_matches_reference_*`). `_update_full_rollout` re-implements the identical math inline. The "extract and verify" step happened; the "switch over" step never did — the worst of both worlds: an extra module *and* the inline copy.

4. **`MappoEvalStats` (62 fields) is restated at least 5 times:** the dataclass, the 62-kwarg constructor call at the end of `evaluate_mappo`, the 107-line hand-written `eval_stats_dict` (with renames like `mean_team_a_score` → `mean_score_a`), the ~34-interpolation `on_eval` print, `_log_canonical_eval`, and the rehearsal gate's metrics dict. Adding one eval metric requires 3–5 coordinated edits with silent-miss failure modes.

5. **Config sprawl across four parallel schemas.** 144 allow-listed key names (`config_schema.py`), 59 `MappoConfig` fields, 33 `RuntimeContext` fields, 41 `run.*` keys consumed as raw untyped `.get()` calls scattered over 5 modules. Defaults are written in 2–3 places per field (`MappoConfig` default, `make_mappo_config`'s `.get(x, default)`, the YAML). Nested sub-dicts (`full_env_rehearsal`, `cap_duel_distill`, `matrix_eval`, …) have no key validation at all — precisely the failure `config_schema.py` was built to prevent at the top level.

6. **`module.cpp` is a 769-line single function wrapping two hand-maintained mirrors.** ~100 lines of validation re-implement `sim.cpp`'s checks (deliberately, to convert aborts into Python exceptions — but textually independent and free to drift); `kValidBotNames` hand-mirrors the factory registry in `runner.cpp` (and a third prose copy in `runner.h` is already stale, missing 2 of 6 names). Inside the block: byte-identical `step`/`step_decision` lambdas, ~8 verbatim env-index-bounds blocks, repeated ndim/length check pairs.

7. **Phase envs reuse each other through three different mechanisms, none clean, while the purpose-built base class sits dead.** `phase4_selfplay` subclasses `Phase4MappoEnv` but calls `gym.Env.__init__` directly, bypassing `super()`; `phase11` doesn't subclass but reaches in statically (`Phase4MappoEnv._action_to_cpp_for_slot(...)`); `phase4_multi_enemy` composes but proxies four *private* attributes through to the wrapped env. Meanwhile `mappo_phase_common.py` (`BaseMappoPhaseEnv` + `RandomizedMapMixin`, written for exactly this) has **zero importers**. Byte-identical 15-line `build_critic_obs`, 15-line info blocks, and a thrice-copied scripted-opponent loop whose three copies normalize `aim_delta` three arithmetically-equal-but-different ways.

8. **The curriculum-setter surface is hand-triplicated beside its own generic dispatcher.** `env_capabilities.py` exists specifically so vector wrappers can "dispatch generically off this tuple rather than carrying one hand-written method per knob" (its own docstring), and the generic paths exist (`_apply_setter`, `_broadcast_setter`, the worker's `elif cmd in CURRICULUM_SETTERS`) — yet all 7 setters are still hand-written in `XushiVectorEnv`, `XushiAsyncVectorEnv`, *and* `SimPoolVectorEnv`, and the three mini-games define silent no-op `set_team_spirit` methods next to the declaration mechanism built to outlaw silent no-ops.

9. **Boundary enforcement has rotted.** The layer checker isn't in CI (`ci.yml` runs pytest only); `vector_env.py:728` violates the layer rule; `behavior_primitives.cpp` includes `../../../sim/src/internal/sim_combat.h`; AGENTS.md and `experiments/configs/README.md` document a config inventory (phase3/5/6/7/8/9/10 dirs and files) that **does not exist**; `docs/architecture/python_layers.md`'s module inventory is stale.

10. **Small-helper duplication in C++ that invites divergence.** `clamp01` ×4 (while `common::clampf` exists), `arena_center`/objective-membership ×5 (with a header comment falsely claiming one "wraps" another), two angle-wrap functions with different boundary conventions, two incompatible `observable_enemy` (the bots one ignores fog and takes an unused parameter), `ray_circle_hit_t` verbatim ×2, two contradictory aim-noise determinism policies (GLSL sin-hash in `sim_pool.cpp` vs splitmix64 in `bot.cpp` with a comment rejecting the sin-hash).

---

## 3. File and Module Hotspots

### `python/train/mappo_rollout_trainer.py` — 905 lines · churn #1 (23 changes)
- **Responsibilities:** rollout tensor bag, `MappoTrainer` (resume/LR, 11 curriculum pass-through setters, update), the PPO+aux+anchor+distill update body, config parsing (`make_mappo_config`), hyperparameter validation.
- **Problem:** `_update_full_rollout` (298 lines) is the concentrated hot spot — see §4. `make_mappo_config` is a 100-line field-by-field transcription restating every default.
- **Why hard:** the update body interleaves 6 concerns (unroll, anchor forward, 3 aux losses, PPO loss, optimization, 30-key metrics) through 13 accumulator lists; 8 config flags materially change its control flow (`value_per_agent` alone forks 3 branches).
- **Direction:** decompose the update body by concern; delete the inline PPO math in favor of `losses.compute_ppo_loss`. **Split? No** (file stays), but the function: yes.

### `python/train/mappo_evaluate.py` — 611 lines
- **Responsibilities:** episode-loop evaluation, combat/objective metric merging, stats serialization.
- **Problem:** `evaluate_mappo` is 374 lines, 9 params, 10 levels of nesting — the deepest code in the repo — ending in a 62-kwarg constructor call; `eval_stats_dict` restates all 62 fields by hand.
- **Direction:** an accumulator object for the per-episode state (collapses the ~15 loose locals), and derive the dict from the dataclass (`dataclasses.asdict` + a small rename map). **Split? No** — simplify in place.

### `python/train/mappo_model.py` — 1,094 lines · churn #3
- **Responsibilities:** four stacked concerns — `MappoConfig` (59 fields), `MappoEvalStats` (62 fields), **8 stateless curriculum schedule functions** (nothing to do with the model), aux-head loss/metric functions (~370 lines).
- **Why hard:** a reader opening "the model file" wades through config prose, eval-stat mirrors, and curriculum ramps to find the network. The constructor stacks 5 validation blocks and 6 conditionally-`None` heads whose `is not None` re-checks ripple into the trainer.
- **Direction:** **Split? Yes** — by concept, not size: schedules → a `curriculum` module (their only consumer is `mappo_training_hooks`); `MappoEvalStats` → the eval module it describes; aux-head losses → beside the other loss code. The model class itself stays put.

### The pretrain family — `full_env_rehearsal.py` (758) + `mappo_bc_pretrain.py` (541) + `composition_rehearsal.py` (435) + `cap_duel_distill.py` (399) + `mappo_pretrain_hooks.py` (468)
- **Problem:** ~2,600 lines that are 4–5 structural clones of one concept: "collect a batch from a teacher, compute a supervised move/aim/fire loss, run an Adam loop, print gates, write a checkpoint." Two raw `torch.save` calls bypass the atomic checkpoint writer. Generic teacher/env utilities live in `composition_rehearsal.py` by historical accident.
- **Direction:** **Consolidate? Yes** — one pretrain toolkit (driver loop, teacher-target extraction, compat assert, loss composition, gate reporting), with each stage becoming its declarative delta. `CapDuelDistillAnchor` is the shape to generalize from — it's the only member with a real interface.

### `src/python_bindings/module.cpp` — 950 lines · churn #16
- **Problem:** one 769-line function; duplicated validation layer; hand-mirrored bot registry; repeated bounds/shape-check blocks; `SimPool.step` lambda 5 levels deep with 6 array args.
- **Direction:** **Split? Yes** — into `register_sim/register_obs/register_pool/...` functions; have C++ export the bot-name list (one `bot_names()` in `runner.cpp` consumed by both `make_bot_by_name` and the binding); single validation source (sim-side validators that return error strings, wrapped once for Python).

### `python/envs/` phase family — 3,120 lines, 7 env classes
- **Problem:** three inconsistent reuse mechanisms + dead purpose-built base class + byte-identical method copies (quantified: 107 matched lines between the two selfplay envs, 41% of the smaller file). Three pure-NumPy mini-games impersonating the tensor contract share a directory and naming scheme with three real sim-backed envs — a genuinely different kind of object, undifferentiated.
- **Direction:** **Partially.** Pick one reuse mechanism (composition with explicit hooks — what `BaseMappoPhaseEnv` intended) and delete the dead file either way. Move shared blocks (`build_critic_obs`, info assembly, scripted-opponent loop, space construction) into one place. Separating mini-games into their own sub-package or naming convention would remove a real trap for newcomers.

### `python/train/runtime_specs.py` (304) + `phases.py` (231) + `runtime_adapter.py` + `envs/runtime_factory.py` (292)
- **Problem:** the dual pipeline (§2.1). `runtime_factory`'s dispatch itself is thin and fine; `_check_ignored_params` is load-bearing safety, keep it.
- **Direction:** **Collapse? Yes** — one resolution path, resolved once per run. Either finish the explicit-spec migration (mechanical YAML rewrite of 77 configs + make `phase:` an alias) or declare the phase registry canonical and delete the explicit path. Finishing the migration is the better end state since `sim_pool` already requires it.

### `python/xushi2/vector_env.py` — 771 lines · churn #4
- **Responsibilities:** sync backend, async backend (workers, timeouts, death diagnostics), sim_pool adapter, factory.
- **Assessment:** mostly justified size — sync vs async share little real code, and the async failure engineering is excellent. Problems: triplicated setter surface (§2.8) and the layer-violating reverse-engineering of a higher layer's `partial` in `_make_sim_pool_vector_env`. **Split? No** — fix the two specific warts.

### `src/viewer/` + `src/sim/src/internal/sim_combat.cpp` + `replay_loader.cpp`
- `sim_combat.cpp` (326): two files stuck together — 80 lines of pure geometry (with `ray_circle_hit_t`/`cross` duplicated elsewhere) + game rules; twin 44-line fire resolvers. **Split? Maybe** — move geometry primitives beside `sim_movement_geometry` and merge the resolvers' shared body.
- `replay_loader.cpp` (267): 116-line `load_replay`, 28 hand-parsed header keys, three replay generations branched on `values.size()`, substring-matching key lookup (`seed` would match inside `random_seed=`), format-version field written by Python but never checked. Also: 1 writer (Python) and **2 independent readers** (this file + `analyze_replay_combat.py`) with no shared schema. **Restructure? Yes** — table-driven key parsing and a version check; longer-term, one schema definition.
- Viewer generally: 8 shared sources compiled into two binaries; `main.cpp --benchmark` duplicates `bench_main.cpp`; 24-field `PanelViewModel` initializer duplicated verbatim in both mains; two benchmark JSON writers, one with zero callers.

### `python/train/mappo_training_hooks.py` — 555 lines · churn #10
- **Problem:** four jobs (curriculum push-down, metric decoration, eval policy, gate artifacts); ~120 of 555 lines are three near-parallel giant f-string log lines; temporal coupling — `collect_rollout` stashes 6 `_last_*` fields that `update_step` later reads.
- **Direction:** keep the class (it's the one `OrchestrationHooks` impl and its caller is legible); extract the log-line builders and make the curriculum state an explicit small object passed between the two hook methods. **Split? No.**

---

## 4. Functions and Local Complexity

| Function | Size/shape | What makes it hard | Fix kind |
|---|---|---|---|
| `MappoTrainer._update_full_rollout` (`mappo_rollout_trainer.py:415`) | 298 lines, 13 accumulator lists, nesting 4 | Interleaves recurrent unroll, a *second* full anchor-model unroll, 3 aux losses, inline PPO math (duplicate of dead shared fn), optimizer step, and a 30-key metrics dict; 8 config flags fork its control flow | Decompose by concern: unroll → per-step outputs; loss assembly; metrics. Adopt `compute_ppo_loss`. |
| `evaluate_mappo` (`mappo_evaluate.py:129`) | 374 lines, 9 params, nesting 10 | ~15 loose accumulators mutated across a 200-line `try` body; ends in a 62-kwarg constructor | Accumulator object + per-episode helper; nesting collapses with it |
| `PYBIND11_MODULE` (`module.cpp:181`) | 769 lines | Every binding concern in one scope; repeated bounds/shape blocks | Decompose into `register_*` units; helper for env-index/shape checks |
| `bc_pretrain_walk_and_shoot_to_objective` (`mappo_bc_pretrain.py:374`) | 168 lines, 12 params, nesting 8 | Near-clone of its 124-line sibling; the diff is mostly an indentation shift from one extra loop | Merge into one parameterized function (or the shared pretrain driver) |
| `build_runtime_context` (`mappo_runtime_context.py:49`) | 191 lines | Flat transcription of config dicts into 33 fields; third redundant call to `resolve_runtime_env_factory` | Resolve once, pass down; group fields into a few sub-structs |
| `make_mappo_config` (`mappo_rollout_trainer.py:758`) | 100 lines | Every default restated (`MappoConfig` already has them) | Drive from the dataclass: fill only keys present, let defaults come from one place |
| `load_replay` (`replay_loader.cpp:152`) | 116 lines | ~35 sequential `parse_kv_*` one-liners; 3-generation branch on `values.size()`; substring key matching | Table-driven key→field map; anchored key match; explicit version handling |
| `MappoTrainingHooks.collect_rollout` (`mappo_training_hooks.py:120`) | 96 lines | Computes 6 schedules, calls 8 `trainer.set_*`, stashes 6 fields read later by a different method | Extract a `CurriculumState` computed in one step and applied/logged explicitly |
| `resolve_revolver_fire` / `resolve_mender_sidearm_fire` (`sim_combat.cpp:174/223`) | 44-line twins, 6 params (one redundant) | 12-line duplicated barrier-damage block; drift risk between weapon rules | Shared hitscan core parameterized by damage/radius; drop the redundant `mechanics` param |
| `train_mappo_from_config` (`mappo_eval_checkpoint.py:22`) | 148 lines | Four sequential gates expressed as accumulating boolean guards, with failure encoded as `total_updates = 0` | An explicit ordered stage list: run → if failed, record & skip PPO. Makes "gate failed" a state, not a sentinel |

---

## 5. Unnecessary Abstractions

- **`losses.compute_ppo_loss` + `PpoLossResult` as currently used** — dead in production (only its own test imports it). Either becomes *the* implementation (preferred) or is deleted; the status quo is pure overhead.
- **`common/result.hpp` (`Result<T>`, 15 `ErrorCode`s), `X2_CHECK_RET`, `FixedVector`** — zero uses. The `X2_REQUIRE(cond, error_code)` macro **discards the error code** at ~150 call sites. Delete or make real.
- **3 of 5 static stage wrappers in `sim_tick_pipeline.cpp`** — pure pass-throughs with identical signatures, while two other stages are called directly. Either wrap all (uniform naming) or none.
- **The bot class hierarchy** (6 `final` classes averaging 15 lines) — variation is one float and one bool; three bots share identical 3-line bodies. `IBot` itself is fine (real polymorphism at the pool boundary); the leaf-class ceremony could shrink to parameterized factories. Low priority.
- **`mappo_phase_common.py`** — the right idea, but dead; in its current state it's a trap (a reader may assume it's the base of the phase envs). Adopt or delete.
- **`benchmark_writer.cpp`'s `write_bench_json`** — zero callers; its twin in `viewer_bench_output.cpp` is the live one. Delete.
- **The hand-written setter forwarding across all three vector-env backends** — the generic `CURRICULUM_SETTERS` dispatch already exists; the per-method forwarding is the redundant layer.
- **Hand-mirrored registries in `module.cpp`** (bot names, validation) — mirrors, not abstractions; each is a drift channel.

## 6. Missing Abstractions or Boundaries

- **A single supervised-pretrain driver** (optimizer loop + logging cadence + gate reporting + checkpoint IO) — the concept exists 4–5 times; it deserves one owner.
- **One teacher-policy-targets + teacher-compat module** — currently duplicated between `composition_rehearsal` and `cap_duel_distill`, and homeless (generic utilities living in a phase-specific file).
- **One eval-stats serialization derived from the dataclass** — the 62-field restatements need a generated path (asdict + rename map), and ideally the three parallel log lines become one formatter.
- **One gate/decision-JSON writer** — currently 5 local inventions of the same `{status, reason, metrics, thresholds}` shape; `mappo_eval_gate_io.write_json_artifact` already exists with one caller.
- **One geometry-primitives home in the sim** (`ray_circle_hit_t`, `cross`, segment distances) and one `clamp01`/`arena_center`/objective-membership — these are real shared concepts currently re-derived per file.
- **A replay-format schema with a version check** — one Python writer, two independent hand-rolled readers (C++ viewer, Python analyzer); the `format:` field exists but nothing validates it.
- **A typed carrier for `run.*` sub-configs** — nested pretrain/gate dicts are consumed raw with zero key validation; `MatrixEvalConfig.from_dict` is the in-repo pattern to replicate.
- **A single "canonical mechanics constants" source** — the `7500/15/0.75F/240` tuple is retyped in ~5 C++ and ~9 Python locations.

## 7. Navigation and Organization

**Works well:** the top-level layout (`src/{common,sim,bots,pool,python_bindings,viewer}` with a clean dependency DAG; `python/{xushi2,envs,train,eval,scripts}` with a written layer contract); the `src/internal` convention; `tests/` mirroring source areas; `experiments/configs/<phase>/{smoke,baseline,probe,legacy}` with metadata blocks and a README honestly labeling 64 probe configs as historical records; `python/tests` with `contracts/`/`smoke/` markers.

**Needs attention:**
- **Enforcement:** wire `check_import_boundaries.py` into CI; fix the `vector_env.py:728` violation it would catch; fix the `behavior_primitives.cpp` cross-module include by exporting what bots need through a proper header.
- **`python/scripts/` holds three undifferentiated populations:** load-bearing tooling (matrix eval, replay_dump, checkers — some invoked by Makefile and imported by tests), dated one-off analyses (six files, ~1,600 lines, each tied to a specific report), and explicit benchmark scratch. The README's own rule ("nothing long-lived lives here") is contradicted by its own workflow tables. Promote the load-bearing pieces; archive or delete the dated ones (two are accidentally load-bearing via test imports of private names — untangle those first).
- **Docs drift:** AGENTS.md and `experiments/configs/README.md` describe phase3/5/6/7/8/9/10 config trees that don't exist; `python_layers.md`'s inventory is stale; `runner.h`'s prose bot list misses 2 of 6 names; a comment in `obs_utils.cpp` points at a constant's old home. Stale maps are worse than no maps — a newcomer following AGENTS.md's config section hits phantom paths immediately.
- **`scripts` isn't a declared package** in pyproject yet is imported by tests and invoked as `python -m scripts.X`.

## 8. Cognitive Load Ranking

**Easy:** `src/sim` internal pipeline (17-stage flat list; best entry point in the repo), `sim_weapon_ranger`, `entity_obs.cpp` (large but architecturally coherent — the fog leak barrier is type-enforced and test-pinned), `common_orchestration.py`, `obs_manifest.py`, `env_capabilities.py`, `reward_features.py`, `mappo_checkpoint_outputs.py`, `config_schema.py`, `python/eval/`, `sim_pool.{h,cpp}`.

**Moderate:** `reward.py` (19 coefficients but well-guarded; the scalar/per-agent dual path is its one real duplication), `vector_env.py`, `sim_pool_env.py` (parity-pinned, honest docstring), `mappo_matrix_eval.py`, `runner.py`, bots, viewer rendering.

**Difficult:** the phase env family (three reuse mechanisms + copies — you must diff files to know what's shared); `module.cpp` (one scope, two mirrors); `mappo_model.py` (four concerns stacked); `mappo_training_hooks.py` + `mappo_pretrain_hooks.py` + `train_mappo_from_config` (gate flow via boolean accumulation and a `total_updates = 0` sentinel); the pretrain family (which clone is authoritative?).

**Very difficult:** `_update_full_rollout` (the single highest-risk edit surface: churn #1 file, 298 lines, 8 behavior-forking flags, silent-wrongness failure modes); `evaluate_mappo` (10 levels deep, 62-field fan-out); the config→env resolution (18 modules, two pipelines, triple resolution — understanding "what env does this YAML produce" is the repo's worst archaeology).

## 9. Simplification Opportunities (highest value)

1. **One config→env pipeline, resolved once** — removes a whole parallel concept, ~4 duplicated helpers, two discarded resolutions, and the sim_pool/legacy incompatibility trap.
2. **One pretrain toolkit** — collapses 4–5 driver clones, 3 loss compositions, 2 teacher-target/compat copies, 2 raw `torch.save` bypasses; each experiment becomes a readable delta instead of a 400–750-line sibling.
3. **One PPO loss, one decomposed update** — turns the repo's most-edited function from a 298-line interleave into named stages; kills a dead module.
4. **Dataclass-derived eval serialization + one log formatter** — 5 restatements of 62 fields → 1 definition + 1 rename map.
5. **`register_*` decomposition of `module.cpp` + C++-exported bot registry + single validation source** — removes both drift channels and makes the binding navigable.
6. **One reuse mechanism for phase envs + shared blocks extracted + mini-games visibly separated** — the family becomes "one real env, N deltas, M toys" instead of 7 files needing pairwise diffing.
7. **Generic curriculum-setter dispatch actually used** — deletes ~120 lines of triplicated forwarding and closes the silent no-op hole in the mini-games.
8. **Dead-code sweep** — `Result<T>`/`FixedVector`/`X2_CHECK_RET`/error-code args, `mappo_phase_common.py`, `benchmark_writer` twin, 5 dead limits constants, `compute_ppo_loss`-as-dead (resolved by #3).

## 10. Prioritized Cleanup Plan

### P0 — Major readability/ownership problems (actively risky)
1. **Collapse the dual runtime resolution** (`runtime_specs.py`, `phases.py`, `runtime_adapter.py`, `runtime_factory.py`; callers in `train.py`, `mappo_eval_checkpoint.py`, `mappo_runtime_context.py`). Now: two pipelines, triple resolution, sim_pool incompatible with 76/78 configs. Change: finish the explicit-spec migration (mechanical config rewrite + `phase:` alias) or crown the phase registry and delete the explicit path; resolve once. End state: one answer to "what env does this YAML build," one seed/env-cfg extraction.
2. **Restore boundary enforcement.** Wire `python -m scripts.check_import_boundaries` into `.github/workflows/ci.yml`; fix `vector_env.py:728` (invert the dependency: `envs`-layer registers the sim_pool builder, or move `_make_sim_pool_vector_env` up a layer); replace `behavior_primitives.cpp`'s `../../../` include with a proper exported header. Also fix the stale AGENTS.md/config-README inventories — they are the onboarding path.
3. **De-risk `_update_full_rollout`** (churn #1): switch to `losses.compute_ppo_loss` (behavior-identical, already reference-tested), then decompose into named stages. End state: the most-edited function in the repo reads as a sequence of verifiable steps.
4. **Close the checkpoint bypass**: route the two raw `torch.save` calls in `mappo_pretrain_hooks.py` (:275, :349) through `save_mappo_checkpoint` (atomicity + single payload builder).

### P1 — High-value simplifications
5. **Pretrain toolkit consolidation** (files in §3): one driver, one loss composition, one teacher-target/compat module, one gate reporter; rebuild the five stages on it. Fold the two BC near-clones into one function.
6. **`evaluate_mappo` + eval-stats serialization**: accumulator object, dataclass-derived dict, single log formatter shared by `on_eval`/`_log_canonical_eval`/gate prints.
7. **`module.cpp` decomposition**: `register_*` functions; export bot names from `runner.cpp` (deletes `kValidBotNames` and fixes the stale `runner.h` list); shared helpers for env-index and array-shape checks; single validation source shared with sim (validators returning error strings; sim asserts on them, bindings throw).
8. **`mappo_model.py` split by concept**: schedules → curriculum module; `MappoEvalStats` → eval; aux losses → loss code. Model class + `MappoConfig` remain.
9. **Phase-env unification**: adopt one composition-with-hooks base (per the dead `BaseMappoPhaseEnv` design), extract the byte-identical blocks, delete `mappo_phase_common.py` if not adopted; distinguish mini-games (sub-package or naming).
10. **Config defaults single-sourced**: `make_mappo_config` driven by the `MappoConfig` dataclass; add key validation for the nested `run.*` sub-dicts (pattern: `MatrixEvalConfig.from_dict`).

### P2 — Local readability improvements
11. Curriculum setters via generic dispatch in all three backends; remove the mini-games' silent no-op setters in favor of `UNSUPPORTED_CURRICULUM_SETTERS`.
12. `sim_combat.cpp`: move geometry primitives to one home (dedup `ray_circle_hit_t`, `cross`); merge the twin fire resolvers' shared body; drop the redundant `mechanics` param. Unify `clamp01`/`arena_center`/objective-membership on `common`/`obs_utils`; reconcile the two angle-wrap and two `observable_enemy` variants (the bots one silently ignores fog — decide and document).
13. `replay_loader.cpp`: table-driven header parsing, anchored key matching, validate the format-version field; share the default-config with `main.cpp`.
14. `mappo_training_hooks`: extract `CurriculumState` (kills the `_last_*` temporal coupling) and the log-line builders; make the pretrain gate flow in `train_mappo_from_config` an explicit stage list instead of boolean accumulation + `total_updates = 0` sentinel.
15. Unify gate-JSON writing on `write_json_artifact`.

### P3 — Optional cleanup
16. Dead-code sweep: `Result<T>`/`ErrorCode` plumbing (or make `X2_REQUIRE`'s code arg real), `FixedVector`, `X2_CHECK_RET`, dead `limits.hpp` constants, `benchmark_writer::write_bench_json`, the 3 pass-through stage wrappers (or wrap all 5 uniformly).
17. `python/scripts` triage: promote load-bearing tools, archive dated one-offs (first untangle the two test imports of private script names), declare the package properly in pyproject.
18. Viewer: single benchmark path (fold `main.cpp --benchmark` and `bench_main.cpp`), shared `PanelViewModel` construction, static library for the 8 twice-compiled sources.
19. Style convergence in `sim_stage_mender/vanguard` (casts, braces, naming) to match siblings; fix the three "delegates to shared geometry helper" comments that point at a call that doesn't happen; unify the two aim-noise hash policies or document why both exist.
20. Bot leaf classes → parameterized factories (keep `IBot`).

---

## Final Question — the 3–5 changes I would make first

1. **Collapse config→env resolution to one pipeline and resolve it once** (P0.1). It's the repo's worst archaeology, a live incompatibility (sim_pool vs `phase:` configs), and every future env/config feature currently pays the two-pipeline tax.
2. **Build the shared supervised-pretrain toolkit and rebuild the five pretrain/rehearsal/distill stages on it** (P1.5). Biggest single reduction in duplicated concepts (~2,600 lines → one driver + small deltas), and it removes the "which clone is authoritative?" question from the most experiment-heavy area.
3. **Adopt the shared PPO loss and decompose `_update_full_rollout`; simplify `evaluate_mappo` and derive eval serialization from the dataclass** (P0.3 + P1.6). These two functions are where correctness risk, churn, and unreadability coincide.
4. **Decompose `module.cpp` and delete its two hand-maintained mirrors** (P1.7). The C++/Python boundary becomes navigable and stops being a silent drift channel.
5. **Re-arm the boundaries: import checker in CI, fix the two layering violations, refresh the stale AGENTS.md/config docs** (P0.2). Cheap, and it stops the currently-good architecture from continuing to erode — every other cleanup holds its value only if the boundaries are enforced.

## Verification (for any future implementation of this plan)
- Behavior-preserving refactors are well-covered by existing suites: `pytest python/tests` (65 files; includes `test_ppo_shared_loss.py` reference tests, `test_sim_pool_env_parity.py`, `test_mappo_phase_env_parity.py`, entity-obs golden/parity/leak tests), `ctest --test-dir build` (20 binaries incl. golden replay hashes), `make test-cpp` / `make train-smoke`.
- Determinism-sensitive edits (anything under `src/sim`, reward, obs) must keep golden replay hashes and parity fixtures byte-stable — per `docs/determinism_rules.md` and AGENTS.md.
- After the runtime-resolution collapse: run every config under `experiments/configs/` through `list_configs.py` + a dry resolve to prove identical env construction.
