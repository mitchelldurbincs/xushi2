# Phase 3 → Phase 4 Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Sync documentation to the post-Phase-2 reality, tighten CI and lint coverage, reclaim repo hygiene after the big trainer refactor, and produce a Phase-4 readiness audit + design memo — so that MAPPO work starts against a clean, accurate foundation.

**Architecture:** Four independent tracks — **B** (docs sync), **C** (repo/artifact hygiene), **D** (test/CI health), **E** (Phase-4 architecture prep). Each track is self-contained; tracks can be executed in any order or in parallel. Track E is **read-only / design-doc only** — no runtime code changes — to stay out of the in-flight Phase 3 training run's way.

**Tech Stack:** CMake/C++20, pybind11, Python 3.10+, pytest, GoogleTest, ruff, mypy, GitHub Actions.

---

## Phase 3 run safety

Phase 3 is currently training on the user's machine. Nothing in this plan touches files the running trainer reads or writes:

- **Do not** modify `python/train/ppo_recurrent/`, `python/envs/phase3_ranger.py`, `python/xushi2/env.py`, `python/xushi2/runner.py`, `python/xushi2/reward.py`, or `experiments/configs/phase3_ranger_*.yaml` during execution.
- **Do not** delete anything under `runs/phase3_ranger/`.
- Track E is audit-only and produces only `docs/plans/...md` — safe.
- If the executing agent needs to rebuild the C++ module for any reason, use a new `build-cleanup/` directory to avoid touching the trainer's `build/` or `build-py313/`.

## Per-task commit policy

The user commits the whole delta at the end. **Do not commit per task.** The final step of the plan is a single staged commit.

---

## Track B — Documentation Sync

The README still says "Phase 1b." Phase 2 cleared and Phase 3 is running. Specs haven't been walked against code since the big trainer refactor (`af4003c`).

### Task B1: README Status line

**Files:**
- Modify: `README.md:9-18`

**Step 1:** Replace the "Status" paragraph. Current text says "Phase 1b." Rewrite to reflect:
- Phase 2 memory-toy gate cleared (cite `docs/plans/2026-04-23-phase2-result.md`: recurrent −0.107, feedforward −0.995, gap 0.889).
- Phase 3 (C++ sim wired into the recurrent PPO trainer) is in progress.
- The viewer is still a scaffold (unchanged).

**Step 2:** Verify. Read `README.md:9-18` and confirm the text matches the Phase 2 result doc's numbers verbatim.

### Task B2: README "Current state" section

**Files:**
- Modify: `README.md:88-133`

**Step 1:** Rewrite the three sublists:

- **What works today:** add recurrent PPO trainer package (`python/train/ppo_recurrent/`, 7 files — config, trainer, losses, evaluate, orchestration, lr_schedule); value normalization; cosine LR schedule; Phase-2 memory-toy env + eval (`python/envs/memory_toy.py`, `python/eval/eval_memory_toy.py`); Phase-3 ranger env + eval (`python/envs/phase3_ranger.py`, `python/eval/eval_phase3.py`); rollout buffer (`python/train/rollout_buffer.py`); actor/critic models (`python/train/models.py`); per-head gradient instrumentation.
- **What's a scaffold:** viewer (unchanged wording). **Remove** the "Python trainer — Phase-0 only" bullet (stale).
- **What's not there yet:** remove "Feedforward PPO (Phase 2), recurrent PPO (Phase 3)" — those exist. Keep "multi-agent MAPPO (Phase 4+)," "Batched / vectorized env," "Fog of war (Phase 7+)," "Second heroes (Phase 10+)."

**Step 2:** Verify. Grep for lingering "Phase-0 only" or "arrives in Phase 2" strings: `Grep pattern="Phase-0|arrives in Phase 2|scaffold only" path="README.md"`. Expected: no matches.

### Task B3: README "Training" section

**Files:**
- Modify: `README.md:134-139`

**Step 1:** Replace the "nothing to train against yet" sentence with accurate commands:

```
# Phase 2 memory-toy (recurrent vs feedforward reference run):
python -m train.train --config experiments/configs/phase2_memory_toy.yaml

# Phase 3 C++ sim + recurrent PPO (smoke):
python -m train.train --config experiments/configs/phase3_ranger_smoke.yaml
```

Keep the pointer to `docs/rl_design.md` §6.

### Task B4: Spec drift audit — `rl_design.md`

**Files:**
- Read-only: `docs/rl_design.md`
- Read-only: `python/train/ppo_recurrent/{trainer,losses,config}.py`, `python/train/rollout_buffer.py`, `python/xushi2/reward.py`
- Create: `docs/plans/2026-04-24-spec-drift-audit.md` (working memo)

**Step 1:** For each of §5 (reward), §6 (curriculum), and whatever section defines PPO hyperparameters, read the spec and then the corresponding code. Record deltas in the memo as a three-column table: *what the spec says / what the code does / resolution (edit spec | edit code | intentional drift, explain)*.

**Step 2:** Do not apply any resolutions in this task. This task produces only the memo; Task B7 will apply the spec-side edits.

**Step 3:** Verify: the memo exists and contains at minimum one row per `rl_design.md` section touched by Phase 2/3 work.

### Task B5: Spec drift audit — `observation_spec.md`

**Files:**
- Read-only: `docs/observation_spec.md`
- Read-only: `src/sim/src/actor_obs.cpp`, `src/sim/src/critic_obs.cpp`, `src/sim/src/obs_utils.cpp`, `python/xushi2/obs_manifest.py`
- Append to: `docs/plans/2026-04-24-spec-drift-audit.md`

**Step 1:** Walk the spec's obs layout definition against the actual field ordering in `actor_obs.cpp` and `critic_obs.cpp`. Confirm `obs_manifest.py` constants match both.

**Step 2:** Record any deltas in the memo under a `## observation_spec.md` heading.

### Task B6: Spec drift audit — `action_spec.md`

**Files:**
- Read-only: `docs/action_spec.md`
- Read-only: `src/common/include/xushi2/common/types.h` (`Action` struct), `src/common/include/xushi2/common/action_canon.hpp`, `python/xushi2/env.py` action space
- Append to: `docs/plans/2026-04-24-spec-drift-audit.md`

**Step 1:** Walk each field in the spec against the code. Confirm edge-triggered vs held semantics.

**Step 2:** Record deltas under `## action_spec.md`.

### Task B7: Apply spec-side edits from the drift audit

**Files:**
- Modify: `docs/rl_design.md`, `docs/observation_spec.md`, `docs/action_spec.md` as called for by the audit memo.

**Step 1:** For every row in the memo marked "edit spec," apply the edit.

**Step 2:** For every row marked "edit code" or "intentional drift," do **not** touch code in this task — those are surfaced to the user at the end of the plan as a follow-up list (see Final Review).

**Step 3:** Verify: re-read each edited section and confirm it now matches the code. No grep check here — this is eyes-on.

### Task B8: Relocate completed plan docs

**Files:**
- Read: `docs/plans/2026-04-21-memory-toy-plan.md`, `docs/plans/2026-04-22-sim-cpp-modularization.md`, `docs/plans/2026-04-22-ppo-recurrent-split.md`
- Potentially modify: each of the above (add a one-line status header)

**Step 1:** For each plan doc, determine completion state:
- `2026-04-21-memory-toy-plan.md` → completed (memory-toy ran, Phase 2 cleared).
- `2026-04-22-sim-cpp-modularization.md` → completed (commit `d0e6634 refactor(sim): split sim.cpp`).
- `2026-04-22-ppo-recurrent-split.md` → completed (commit `af4003c Big trainer refactor`).

**Step 2:** Prepend each file with a one-line status block:
```
> **Status:** Completed YYYY-MM-DD (commit <short-sha>).
```

Do not move files; keeping history co-located is fine. Date = date of the landing commit.

---

## Track C — Repo / Artifact Hygiene

`.gitignore` is already solid (`build/`, `build-*/`, `runs/`, `.pytest_cache/` all ignored; `git status --ignored` confirms nothing accidentally tracked). Hygiene here is about disk reclaim and eliminating *import ambiguity*, not about untracking files.

### Task C1: Inventory and reconcile build directories

**Files:**
- None modified directly; this task produces a decision.

**Step 1:** Report sizes: `du -sh build build-full build-py313`. Current: 455M / 404M / 26M (885M total).

**Step 2:** Check which is the trainer's active build. Phase 3 is running — inspect `experiments/configs/phase3_ranger_*.yaml` and any README/CI hint for which build dir the live run depends on. The Python module lives at `python/xushi2/xushi2_cpp.cp31{2,3}-win_amd64.pyd`, *not* in `build*/`, so the Phase 3 trainer does not need `build*/` at runtime.

**Step 3:** Present to the user (via the executing agent's checkpoint): "which build dirs are still useful?" with sizes. **Do not delete without confirmation** — build dirs are recreatable but recreation costs minutes.

**Step 4:** After user confirms, delete the abandoned ones.

### Task C2: Remove stale MSBuild scratch dirs

**Files:**
- Delete: `python/xushi2/Debug/`, `python/xushi2/Release/` (if present and empty / only contain old intermediates).

**Step 1:** `ls python/xushi2/Debug python/xushi2/Release`. Report contents.

**Step 2:** If contents are only `.obj`, `.pdb`, or empty: delete. If they contain a current `.pyd` that's not in the parent: stop and flag to the user.

**Step 3:** Verify: directories gone.

### Task C3: Prune stale Python-version `.pyd`

**Files:**
- Delete: whichever of `python/xushi2/xushi2_cpp.cp312-win_amd64.pyd` and `python/xushi2/xushi2_cpp.cp313-win_amd64.pyd` is not for the active venv.

**Step 1:** Detect active Python: check `python/.venv/pyvenv.cfg` if present, else ask the user. (Phase 3 is running under *one* of these — do not delete the active one.)

**Step 2:** Delete the stale `.pyd` only. Keep whichever matches the active interpreter.

**Step 3:** Verify import still works: `python -c "import xushi2; print(xushi2.__file__)"` from `python/`. Expected: prints path to `python/xushi2/__init__.py` with no error.

### Task C4: Consolidate `runs/` locations

**Files:**
- Possibly move: `python/runs/*` → `runs/`

**Step 1:** `ls python/runs runs`. Current: `python/runs/` has 10+ Phase-2 experiment dirs; top-level `runs/` has only `phase3_ranger/`.

**Step 2:** Decision point — which is canonical? The trainer code should be checked: `Grep pattern="runs/" path="python/train/ppo_recurrent" -n`. Whatever the code writes to is canonical.

**Step 3:** Move the *non-canonical* experiment dirs to the canonical location with `git mv`-style `mv` (these are gitignored so git doesn't care — plain `mv`). Do not move `phase3_ranger/` if Phase 3 is actively writing to it.

**Step 4:** Verify: `ls <canonical>/runs/` shows all Phase 2 experiments plus `phase3_ranger/`. The other location is empty.

### Task C5: `.gitignore` tightening

**Files:**
- Modify: `.gitignore`

**Step 1:** Add (if not already implied):
```
# MSBuild scratch inside the package dir
python/xushi2/Debug/
python/xushi2/Release/
# Secondary runs path (belt-and-suspenders; top-level runs/ already ignored)
python/runs/
```

**Step 2:** Verify: `git check-ignore -v python/xushi2/Debug python/runs` shows a matching rule for each. `git status` stays clean.

---

## Track D — Test / CI Health

`pyproject.toml` has ruff and mypy configured. CI runs neither. CI tests on Ubuntu + Python 3.10 only; user develops on Windows with 3.12/3.13.

### Task D1: Add ruff lint job

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1:** Add a new job after `python-tests`:

```yaml
  python-lint:
    name: Python Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: python -m pip install --upgrade pip ruff
      - run: ruff check python/
```

**Step 2:** Run `ruff check python/` locally first. If it fails, fix the findings in the same task before adding the CI job — do **not** land a red CI step.

**Step 3:** Verify: `ruff check python/` exits 0 locally.

### Task D2: Add mypy job (optional if it cascades)

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1:** Run `mypy python/xushi2 python/train python/envs python/eval` locally first.

**Step 2:** If mypy finds >20 issues, **skip this task** and add a follow-up item. A red mypy job is worse than no mypy job. If it finds ≤20 or all are ignorable, fix and add the job.

**Step 3:** If adding: mirror the ruff job structure; `python -m pip install -e 'python[dev]'` (need torch stubs) then `mypy python/`.

### Task D3: Python version matrix

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1:** Change `python-tests.steps[Set up Python]` to use a matrix:

```yaml
  python-tests:
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    ...
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
```

**Step 2:** Verify locally only that `pip install -e 'python[dev]'` succeeds on 3.10 (already passing in CI). The 3.11 and 3.12 runs will get exercised when CI fires on the cleanup commit — that's fine.

### Task D4: Warnings-as-errors C++ build variant

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1:** Add a second C++ job (not a replacement — keep the lenient one) with `-DXUSHI2_WARNINGS_AS_ERRORS=ON`. Only runs on Ubuntu. It may currently fail; if it does, **skip the job and add a follow-up item**. Same reason as D2.

**Step 2:** If adding: verify locally:
```
cmake -S . -B build-wae -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_VIEWER=OFF -DXUSHI2_WARNINGS_AS_ERRORS=ON
cmake --build build-wae -j
```
Expected: build completes with zero warnings treated as errors.

### Task D5: Audit skipped / xfail tests

**Files:**
- Read-only: all `tests/` and `python/tests/` files.

**Step 1:** `Grep pattern="skip|xfail|SKIP|DISABLED_" path="tests" -n` and same for `python/tests`.

**Step 2:** Record findings in a short section appended to `docs/plans/2026-04-24-spec-drift-audit.md` under `## Skipped tests`. Don't fix — just list. Each entry: file, line, reason, whether the reason is still valid.

---

## Track E — Phase 4 Architecture Prep (read-only)

Phase 4 = MAPPO. That needs: (1) a batched / vectorized env (currently missing per README), (2) a 3v3 environment (currently `env.py` is 1v1 Ranger), (3) a centralized-critic / decentralized-actor rollout shape, (4) self-play scaffolding. This track produces a design doc — **no runtime code changes** — so it's safe to run while Phase 3 trains.

### Task E1: Single-agent assumption audit

**Files:**
- Read-only: `python/xushi2/env.py`, `python/envs/phase3_ranger.py`, `python/xushi2/reward.py`, `python/xushi2/runner.py`, `python/train/rollout_buffer.py`, `python/train/ppo_recurrent/{trainer,losses,orchestration,evaluate}.py`, `python/train/models.py`
- Create: `docs/plans/2026-04-24-phase4-prep.md`

**Step 1:** For each file, record a row: *file / shape assumption / becomes-what under MAPPO*. Example row: `rollout_buffer.py | obs tensor (T, obs_dim) per rollout | (T, n_agents, obs_dim) under MAPPO`.

**Step 2:** Tag each row *reusable-as-is / extend / rewrite / new module needed*.

**Step 3:** Verify: the doc has a row for every file listed in Step 1.

### Task E2: Centralized-critic readiness check

**Files:**
- Read-only: `src/sim/src/critic_obs.cpp`, `src/sim/include/xushi2/sim/obs.h`
- Append to: `docs/plans/2026-04-24-phase4-prep.md`

**Step 1:** Record whether the existing critic obs is already team-level (good — reusable) or actor-level (needs widening). MAPPO-CTDE wants a critic obs that sees the full team state; if `critic_obs.cpp` already does that, this is a big Phase-4 head start and should be called out.

### Task E3: VectorEnv interface spec

**Files:**
- Append to: `docs/plans/2026-04-24-phase4-prep.md`

**Step 1:** Draft a VectorEnv signature (Gymnasium-compatible, async-capable given `7ff25e7 Making async to try on my mac`). At minimum specify: `reset(seeds) -> batched_obs`, `step(batched_actions) -> batched_obs, rewards, terminateds, truncateds, infos`, shape conventions (N envs × n_agents × feature).

**Step 2:** Note which of gym's `AsyncVectorEnv` / `SyncVectorEnv` / custom is most appropriate given the C++ sim is already a cheap Python-held object.

### Task E4: MAPPO data-flow sketch

**Files:**
- Append to: `docs/plans/2026-04-24-phase4-prep.md`

**Step 1:** Diagram (ASCII or prose) the rollout → advantage → loss pipeline under CTDE: actor obs per agent → policy heads per agent (shared or per-role?); critic obs per team → value head per team; GAE over per-agent rewards; PPO loss decomposition across agents.

**Step 2:** Call out open questions: parameter sharing across Ranger slots? Per-role heads when second heroes arrive in Phase 10+? Self-play opponent pool or frozen-snapshot?

### Task E5: Phase-4 readiness conclusion

**Files:**
- Append to: `docs/plans/2026-04-24-phase4-prep.md`

**Step 1:** Write a short "Conclusion" section: *what's already Phase-4 ready, what needs extension, what's net-new, recommended implementation order for Phase 4.* Cap at ~300 words.

**Step 2:** This section is the actual value of Track E — it is what makes Phase 4 kickoff efficient. Bias toward concrete file-level recommendations over abstract principle.

---

## Final Review

After all tracks complete:

### Step 1: Surface follow-ups to the user

Print a summary of:
- Any audit rows marked "edit code" in the drift memo (B7).
- Any CI jobs skipped because they'd land red (D2, D4).
- Any build directories the user needs to decide on (C1).
- The Phase-4 prep doc's Conclusion (E5) as the recommended Phase 4 kickoff plan.

### Step 2: Verify the tree is clean

```bash
git status
```

Expected: only tracked-file modifications and new files under `docs/plans/`, `README.md`, `.github/workflows/ci.yml`, `.gitignore`. No stray build artifacts.

### Step 3: Run the full test suite once

```bash
ctest --test-dir build --output-on-failure   # or whichever build dir survived C1
pushd python && pytest tests && popd
```

Expected: both green. If Phase 3 is still running and monopolizing `build/`, skip the C++ run and note it in the final summary.

### Step 4: Single commit

User commits the whole delta themselves per the per-task commit policy above. Executing agent should **not** run `git commit`.

---

## Execution notes for the agent

- Tasks within a track are mostly independent; parallelize where safe.
- **Tracks C and E are maximally safe** to run first and in parallel — C is disk hygiene, E is read-only audit. Neither can break Phase 3.
- **Track B** needs sequence: B4/B5/B6 (audits) → B7 (apply edits) → B1/B2/B3/B8 (README + plan-doc tweaks) can run anywhere.
- **Track D** — run D5 first (read-only audit), then D1 (ruff, very low risk), then D3 (matrix). D2 and D4 are conditional on local dry-run result.
- If any task's preconditions aren't met (e.g., Phase 3 still writing to `runs/phase3_ranger/` during C4), defer that single task and continue the rest.
