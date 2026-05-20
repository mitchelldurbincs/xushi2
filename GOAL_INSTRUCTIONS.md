# Goal: Phase 4 Objective Timing Curriculum and Long-Run Progress

## Purpose

Investigate and fix the Phase 4 failure mode where agents can create kill or
majority-on-point advantage but cannot convert it into uncontested capture and
score. This goal is specifically about the "episode too short / capture too
slow / contest too brittle" axis.

This is not another smoke-only probe. Produce visible Phase 4 progress with
longer W&B-backed experiments, checkpoints, replay artifacts where useful, and
clear verification.

## First Reads

Before making code or config changes, read these in order:

1. `docs/journal/reinforcement_learning_journal.md`
2. `docs/game_design.md`
3. `docs/rl_design.md`
4. `docs/plans/README.md`
5. Relevant active Phase 4 plans in `docs/plans/active/`

Carry forward the journal evidence. Do not rediscover or repeat already
falsified hyperparameter-only variants.

## Context From Latest Phase 4 Evidence

The majority-on-point curriculum was implemented and verified. It was useful as
a diagnostic but did not clear Phase 4.

Observed local results:

- The no-anneal diagnostic produced Team A majority-on-point windows but still
  scored `0.00/0.00`.
- The annealed smoke produced live majority/cap-progress diagnostics, but final
  real-reward eval at alpha `0.0` ended `0W/20L/0D`, score `0.00/2.90`.
- Team A can have substantial majority-on-point time while still getting almost
  no uncontested capture/scoring time.
- This points at objective conversion: one living enemy contesting the point
  freezes progress, capture takes 8 seconds, unlock takes 15 seconds, and
  current Phase 4 episodes are often 30-60 seconds.

Working hypothesis:

```text
The learner can reach/contest/fight on point, but the canonical objective timing
is too steep for the current 3v3 Ranger policy to discover the full chain:

fight -> clear contest -> complete capture -> keep owner uncontested -> score
```

## Non-Negotiables

- Use W&B for experiment runs. Do not set `WANDB_MODE=disabled` except for tiny
  local import/config checks that are not claimed as experiment evidence.
- If W&B auth or network fails, block with `HUMAN_INSPECTION_REQUIRED` and the
  exact failure. Do not silently replace W&B with local-only tracking.
- Do not call a smoke run Phase 4 progress. Smoke can verify plumbing only.
- Do not silently change canonical game constants. Any objective timing change
  must be explicitly curriculum-only or config-gated with canonical defaults.
- Do not claim Phase 4 gate clearance from noncanonical objective timing.
- Self-play win rate is not evidence by itself.
- Keep MAPPO spelled MAPPO.

## Primary Work

Create a curriculum or controlled experiment path that directly tests objective
timing:

1. Verify whether objective unlock and capture duration are configurable today.
2. If not configurable, add narrow config-gated support for objective timing:
   - canonical defaults must remain unchanged,
   - deterministic integer-tick behavior must be preserved,
   - replay/obs contracts must remain coherent,
   - docs/tests must make the curriculum-vs-canonical distinction explicit.
3. Add trainer/environment scheduling if needed so a run can start with easier
   objective timing and anneal toward canonical timing.
4. Run longer W&B experiments that can produce meaningful behavior, not just
   import or one-update checks.
5. Record results in the RL journal with W&B URLs, config paths, seeds,
   checkpoints, replay paths, metrics, and a decision.

## Suggested Implementation Shape

Prefer config-gated objective timing fields with canonical defaults. Names can
change if the codebase has a better local pattern, but the behavior should be
equivalent to:

```yaml
env:
  objective_timing_curriculum:
    enabled: true
    initial_unlock_seconds: 5
    initial_capture_seconds: 2
    final_unlock_seconds: 15
    final_capture_seconds: 8
    anneal_updates: 400
    eval_canonical_every: 25
```

If this is implemented in C++ `MatchConfig`, preserve the canonical default:

```text
objective_unlock_seconds = 15
objective_capture_seconds = 8
```

If curriculum timing is trainer-side, make sure:

- rollout envs get the scheduled timing before rollout collection,
- normal eval can report both current-curriculum eval and canonical eval,
- final eval includes canonical timing,
- W&B logs current unlock/capture seconds every update.

## Diagnostics To Preserve And Use

Use the existing Phase 4 diagnostics from the majority-on-point work:

- `mean_majority_on_point_seconds_a/b`
- `mean_uncontested_on_point_seconds_a/b`
- `mean_alive_edge_no_score_seconds_a/b`
- `mean_cap_progress_gain_ticks`
- `mean_cap_progress_loss_ticks`
- Team A/B score, kills, hit/fire, visible fire, aim error
- rollout `self_on_point_fraction`
- majority shaping alpha if used
- objective timing values if newly added

Add missing diagnostics only if they are necessary to answer the timing
question.

## Experiments To Create

Create configs under `experiments/configs/phase4/probe/`. Include
`metadata.hypothesis`, `metadata.falsification_criteria`, and
`metadata.max_updates_if_no_signal` on every config.

### 1. Fixed Easy Timing Long Diagnostic

Path suggestion:

```text
experiments/configs/phase4/probe/phase4_mappo_objective_timing_easy_long.yaml
```

Purpose: prove that the current policy/training stack can score when the
objective conversion chain is made learnable.

Recommended shape:

- opponent: `weak_basic_v2`
- warm-start: `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`
- BC: existing `walk_and_shoot`
- objective unlock: 5 seconds
- capture: 2 seconds
- round length: 60 seconds
- updates: at least 250
- eval every: 25
- eval episodes: at least 50
- W&B enabled

This is diagnostic only, not Phase 4 gate evidence.

### 2. Timing Curriculum Long Run

Path suggestion:

```text
experiments/configs/phase4/probe/phase4_mappo_objective_timing_curriculum_long.yaml
```

Purpose: test whether the easier objective can be annealed back toward the
canonical rule without losing score conversion.

Recommended shape:

- opponent: `weak_basic_v2`
- warm-start: `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`
- BC: existing `walk_and_shoot`
- objective unlock: 5s -> 15s
- capture: 2s -> 8s
- anneal: 300-500 updates
- total updates: at least 500
- eval every: 25
- eval episodes: at least 50
- include canonical eval every eval interval if feasible
- W&B enabled

This is the main run for visible progress.

### 3. Canonical Longer-Episode Control

Path suggestion:

```text
experiments/configs/phase4/probe/phase4_mappo_canonical_120s_long.yaml
```

Purpose: separate "episode too short" from "capture timing too steep."

Recommended shape:

- canonical unlock: 15 seconds
- canonical capture: 8 seconds
- round length: 120 seconds
- opponent: `weak_basic_v2`
- warm-start and BC same as above
- total updates: at least 250
- eval every: 25
- eval episodes: at least 50
- W&B enabled

If this scores while 60s canonical does not, episode length is a real blocker.
If this still does not score while easy timing does, capture/contest difficulty
is the main blocker.

## Run Strategy

Run in this order:

1. Do a tiny local config/import check only if needed. Do not log it as
   experiment evidence.
2. Run the fixed easy timing long diagnostic.
3. If it cannot produce score or cap conversion by its stop point, stop and
   write the falsification.
4. If it does produce score/cap conversion, run the timing curriculum long run.
5. Run the canonical 120s control unless the first two runs already clearly
   prove a different blocker.
6. If any long run shows positive score conversion, rerun the best config on
   at least three seeds before making a strong claim.

Use these seeds unless there is a reason to choose different ones:

```text
3519994490
3519994491
3519994492
```

## Stop And Continue Criteria

Stop early and record falsification if by `metadata.max_updates_if_no_signal`:

- Team A score is still zero,
- Team A cap-progress gain is not meaningfully above the prior majority smoke,
- Team A uncontested-on-point seconds remain near zero,
- rollout `self_on_point_fraction < 0.25`,
- Team A hit/fire collapses below recent weak_basic_v2 baselines.

Continue a run if at least one is true:

- Team A score is nonzero,
- Team A wins at least one eval episode,
- Team A uncontested-on-point seconds rise materially,
- Team A cap-progress gain rises materially,
- replay shows coherent fight-then-point behavior.

## Verification Before Long Runs

Run the relevant build/tests before claiming the code path is ready:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=C:\Python313\python.exe
cmake --build build --config Release
cd python
py -3.13 -m pytest tests/test_reward.py tests/test_phase4_mappo_env.py tests/test_mappo_team_spirit_ramp.py tests/test_mappo_bc_freeze.py tests/test_mappo_public_api.py tests/test_bindings_obs.py tests/test_obs_manifest.py -q
```

If C++ objective rules/configs are touched, also run:

```powershell
ctest --test-dir build -C Release -R "Objective|Determinism|GoldenReplay|ActorLeak|ActorObs|CriticObs|ObsDims|ObsUtils" --output-on-failure
```

Add focused tests for any new objective timing config:

- canonical defaults match existing 15s unlock and 8s capture,
- shortened capture completes in the expected integer number of ticks,
- shortened unlock starts objective updates at the expected tick,
- same seed plus same actions remains deterministic,
- canonical eval ignores curriculum timing when requested.

## W&B And Artifact Requirements

For every long experiment, record:

- git commit,
- config path,
- seed,
- W&B run URL,
- checkpoint path,
- replay path if produced,
- objective timing schedule used,
- majority/capture diagnostics,
- score/win/loss/draw,
- Team A/B kills,
- Team A/B hit/fire and aim error,
- whether eval was curriculum timing or canonical timing.

The trainer may not print the W&B URL directly. Parse it from the run metadata
under the relevant `wandb/latest-run/files/wandb-metadata.json` path if needed.
Do not fabricate URLs.

Dump a replay for:

- any run with Team A score > 0,
- any run with Team A win > 0,
- any run where cap-progress conversion materially improves,
- the best checkpoint of the main timing curriculum run.

If replay judgment is required, block with:

```text
HUMAN_INSPECTION_REQUIRED
```

Include W&B URL, replay path, viewer command, exact questions for the human,
and the comment format needed to unblock.

## Journal Update

After each run, append to `docs/journal/reinforcement_learning_journal.md`.
Each entry must include:

- hypothesis,
- config path,
- seed,
- git commit,
- W&B URL,
- checkpoints,
- replay artifacts,
- test commands and results,
- objective timing values,
- key metrics,
- decision: cleared, not cleared, falsified, blocked, or evidence insufficient.

## Expected Outcome

The goal is not merely to create configs. The goal is to determine whether
objective timing is a real blocker and to create visible Phase 4 progress.

Useful outcomes include:

- fixed easy timing scores but canonical longer episode does not: capture/contest
  timing is the blocker,
- canonical 120s scores but 60s does not: episode length is the blocker,
- timing curriculum scores early but loses score as it anneals: need a slower or
  staged curriculum,
- none of the timing variants score: objective timing is probably not the only
  blocker, and attention should return to actor information/coordination.

Do not stop at "implemented." Run the experiments, verify them, record the
evidence, and make a concrete next recommendation.
