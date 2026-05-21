# Goal: Solve Phase 4 Via Cap-Duel Self-Play, Then Transfer To 3v3

## Purpose

Build the missing rung between the two skills Phase 4 has already proven it
can learn in isolation — cap holding (`phase4_mappo_basic_v6_5`) and 1v1
combat (`phase4_mappo_combat_1v1_v2` in the May-18 journal) — by adding a
new Phase 4-compatible mini-game in which both halves of the composition
exist together: one active learner, one active enemy, both spawn near the
objective, score ticks require the learner to kill or displace the enemy and
then remain on point. Then train that env with current-vs-current self-play,
and finally probe transfer into the full 3v3 environment.

This is Strategy 2 ("Objective-Coupled Combat Micro-Curriculum") from
`docs/reports/2026-05-18-phase4-strategic-proposal.md`. It is the next
escalation after the May-21 stop note: composition rehearsal cannot launch
because the combat teacher checkpoint is missing on this machine, and anchor
mix v2 hit the canonical-score falsification trip again.

## Non-Negotiables

- Do not change sim rules, reward formulas, observation layout, action
  semantics, replay format, deterministic sim, W&B metric schemas, or MAPPO
  core. The new env may define its own *mini-game reward* (score/kill/hit
  signals), but it must read sim state through existing Python bindings and
  must not modify C++.
- Do not widen actor observation. Cap-duel must preserve the same 3-agent
  Phase 4 actor obs and 6-dim action shape as `combat_1v1`; inactive
  learner/enemy slots stay in tensor shape but contribute zero actions.
- Do not weaken phase gates. Each stage's `phase_gate:` block is the bar.
  If a run does not meet it, change the experiment, not the gate.
- Do not skip the journal entry. Every completed (or skipped) run gets a
  section in `docs/journal/reinforcement_learning_journal.md` with W&B URL,
  config path, manifest summary, gate decision artifact, and a one-line
  decision.
- Do not launch a new training run while one is in flight. One run at a time.
- Do not commit changes unless explicitly asked.
- Keep actor/critic separation. No actor-observation path may call
  hidden-enemy/full-state helpers.
- Do not modify the C++ simulator, replay file format, or `MatchConfig`
  schema. The mini-game lives in Python.

## First Reads

Before writing code, read in order:

1. `docs/reports/2026-05-18-phase4-strategic-proposal.md` — Strategy 2
   ("cap-duel") section. Treat it as the design intent, not as a contract.
2. `python/envs/phase4_combat_1v1_mappo.py` (if present; otherwise search
   `python/envs/` for the `combat_1v1` registry entry) — the closest
   template for a Phase 4-compatible single-active-slot mini-game.
3. `python/envs/__init__.py` — the env registry where the new mini-game is
   wired in.
4. `python/train/phases.py` — how `env.mini_game` is routed to env bundles,
   and what extra fields a mini-game config can carry.
5. `python/train/mappo.py` and `python/train/mappo_eval_checkpoint.py` —
   train entrypoint and where mini-game configs flow through PPO and eval.
6. `python/scripts/eval_mappo_matrix.py` and
   `python/train/mappo_matrix_eval.py` — anchor-transfer evaluation;
   relevant for Stage 3 (transfer probe).
7. `python/train/phase_gate/README.md` — the gate's evidence shape and CLI.
8. The May-18 journal entries for `combat_1v1_v1`, `combat_1v1_v2`,
   `combat_1v1_transfer_v1`, `cap_duel_v1 and transfer`, and
   `composition_rehearsal_v2_2000` (lines ~737-947 of
   `docs/journal/reinforcement_learning_journal.md`). They are the closest
   prior art for this strategy.
9. The May-21 entries (the most recent two journal sections) — what was
   tried last and why this strategy was chosen.

If a doc and the current code disagree, stop and surface the disagreement
rather than picking silently.

## The Three Stages

Stages are sequential. Do not skip Stage 1's tests, do not start Stage 2
until Stage 1 passes the tests below, and do not start Stage 3 until Stage 2
has a checkpoint that solves the duel gate.

### Stage 1 — Build the cap_duel mini-game (code)

Create `python/envs/phase4_cap_duel_mappo.py`. Mirror
`Phase4Combat1v1MappoEnv` for tensor shapes, action routing, and bot-slot
zeroing. Differences from `combat_1v1`:

- Both the active learner slot and the active enemy slot spawn within
  `point_radius` of the objective center.
- Score ticks only when the learner is on point AND the enemy is dead OR
  the enemy has been displaced off-point for ≥ `enemy_recontest_delay`
  decisions.
- Enemy can be either a scripted recontesting bot (Stage 2 self-play uses
  a current-policy enemy via the existing self-play schedule; the
  scripted-bot variant is the anchor-mix piece).
- Episode length: `episode_decisions` (config-driven; default 96).

Config surface (extends the existing `env.mini_game` route):

```yaml
env:
  mini_game: cap_duel
  mini_game_config:
    episode_decisions: 96
    enemy_hp: 3
    point_radius: 0.18
    score_ticks_to_clear: 12
    enemy_recontest_delay: 12
    hit_tolerance: 0.12
    hit_reward: 1.0
    kill_bonus: 4.0
    score_per_tick: 0.1
    off_point_penalty: 0.0
    time_penalty_per_decision: 0.0
```

Wiring:

- Register in `python/envs/__init__.py` next to `combat_1v1`.
- Add to `python/train/phases.py` mini-game routing if mini-game configs
  are dispatched there.
- Add `python/tests/test_phase4_cap_duel_mappo.py` with at least:
  - tensor-shape parity with `combat_1v1`,
  - active-slot mask correctness (only one learner slot acts),
  - score ticks DO NOT advance while the enemy is alive and on point,
  - score ticks DO advance after kill + the configured on-point delay,
  - deterministic reset under a fixed seed.

Acceptance for Stage 1:

```powershell
cd python
py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py `
  tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py `
  tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py -q
py -3.13 -m scripts.check_import_boundaries
```

All tests pass; import-boundary check passes. Existing `combat_1v1` env
tests remain green (no regressions in the shared route).

### Stage 2 — Train cap_duel with self-play (config + run)

Write `experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v1.yaml`:

- Warm-start from `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` (the
  cap teacher) so the policy starts knowing cap approach.
- `env.mini_game: cap_duel`, with the `mini_game_config:` block above.
- `env.self_play: { enabled: true }`, schedule weights `current: 0.7,
  anchor: 0.3, snapshot: 0.0`, `anchor_bot: noop` (scripted recontester is
  the active enemy slot only when the self-play schedule picks `anchor`).
  Anchor mixing keeps the policy honest against a non-policy enemy.
- PPO knobs: `num_envs: 64`, `rollout_len: 128`, `learning_rate: 1.0e-6`,
  `entropy_coef: 0.02`, `value_normalization: true`, `gamma: 0.997`,
  `gae_lambda: 0.95`, `lr_schedule: cosine`, `lr_final_ratio: 0.1`.
- BC pretrain: optional, `bc_pretrain_variant: walk_and_shoot,
  bc_pretrain_steps: 500` to re-anchor cap movement and seed the fire
  action.
- `run.total_updates: 250`, `eval_every: 25`, `checkpoint_every: 25`,
  `output_dir: runs/phase4_mappo_cap_duel_selfplay_v1`.

Add a `phase_gate:` block that scores Stage 2 on the duel itself, not the
3v3 game:

```yaml
phase_gate:
  phase: phase4_cap_duel_selfplay_v1
  identity_requirements:
    min_unique_seeds: 1
  objective_checks:
    - id: cap_duel_score
      metric: eval/mean_score_a
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 6.0
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: cap_duel_kills
      metric: eval/team_a_kills
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 5.0
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: cap_duel_wins
      metric: eval/wins
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 25
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
  subjective_checks:
    required: true
    trigger_if_objective_passed: true
    approval_rule: all_yes
    questions:
      - id: replay_kill_then_hold
        prompt: "In a greedy cap_duel replay, does the learner kill or
                 displace the enemy and then visibly remain on point long
                 enough to convert ticks to score, rather than just
                 trading fire?"
```

Launch the run. Capture stderr/stdout to `runs/<output_dir>/launch.log`.
While the run is in flight, do nothing except check at long intervals for
completion.

Post-training:

1. Dump one stochastic and one greedy replay of the best checkpoint via
   `python -m scripts.dump_replay` (the cap_duel env path; if the existing
   dumper does not support cap_duel, treat that as a Stage 1 follow-up
   instead of broadening scope).
2. Build evidence with `python/scripts/build_run_evidence.py`, pointing
   `--config` at the Stage 2 yaml.
3. Invoke the gate:
   `py -3.13 -m train.phase_gate.cli --phase-config <yaml>
    --run-evidence <evidence.json>
    --output runs/<output_dir>/gate_decision.json`.
4. Journal the run per the template at the bottom of this file.

Branching:

- `CLEARED` (objectives passed) or `HUMAN_INSPECTION_REQUIRED` (objectives
  passed, subjective replay check needed): move to Stage 3 using the best
  checkpoint as the transfer warm-start.
- `NOT_CLEARED`: apply ONE config-only change per Decision Rules below and
  loop Stage 2 once. After two `NOT_CLEARED` Stage-2 iterations, stop and
  report.
- `BLOCKED`: fix the crash/NaN/import error and rerun.
- `EVIDENCE_INSUFFICIENT`: rebuild the missing artifact and re-invoke the
  gate without re-training.

### Stage 3 — Transfer probe to 3v3 (config + run)

Write `experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml`:

- Warm-start from the Stage 2 best checkpoint
  (`runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt`).
- Full Phase 4 3v3 env: `opponent_bot: weak_basic_v2`, NO mini-game flag,
  `env.self_play.enabled: false`.
- PPO knobs: `num_envs: 64`, `rollout_len: 128`, `learning_rate: 1.0e-6`,
  `entropy_coef: 0.02`. Conservative LR so the duel skill is not
  immediately unlearned under the wider 3v3 reward.
- BC pretrain: off (warm-start already carries combat behavior; full-env
  `walk_and_shoot` BC has previously erased it).
- `run.total_updates: 200`, `eval_every: 25`, `checkpoint_every: 25`,
  `output_dir: runs/phase4_mappo_cap_duel_transfer_v1`.

`phase_gate:` block reuses the canonical Phase 4 anchor-transfer bar — the
same thresholds applied to `anchor_mix_v2_long`:

```yaml
phase_gate:
  phase: phase4_cap_duel_transfer_v1
  identity_requirements:
    min_unique_seeds: 1
  objective_checks:
    - id: weak_basic_v2_score
      metric: eval/mean_score_a
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 3.0
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: weak_basic_v2_wins
      metric: eval/wins
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 5
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: hit_fire_floor
      metric: eval/team_a_hit_fire
      source: wandb
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 0.04
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: anchor_vs_basic_score
      metric: matrix/anchor/basic/mean_score_a
      source: local
      aggregation: { type: max, window: all }
      comparator: ">="
      threshold: 1.0
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
    - id: anchor_vs_noop_no_loss
      metric: matrix/anchor/noop/loss_rate
      source: local
      aggregation: { type: max, window: all }
      comparator: "<="
      threshold: 0.5
      min_samples: 1
      on_missing: EVIDENCE_INSUFFICIENT
  subjective_checks:
    required: true
    trigger_if_objective_passed: true
    approval_rule: all_yes
    questions:
      - id: replay_transfer_intact
        prompt: "In a greedy 3v3 replay vs weak_basic_v2, does the kill-and-hold
                 behavior from cap_duel still happen, or has full-env PPO
                 erased it?"
```

Post-training: matrix eval vs `noop`, `weak_basic_v2`, `basic`; stochastic
+ greedy replays; evidence; gate; journal. Same shape as Stage 2.

## Decision Rules For Config Changes

Apply at most ONE change per iteration so signal is interpretable.

**Stage 1 (env build) failures:**

- A focused test fails → fix the env code, don't relax the test. If the
  test reveals a design ambiguity, surface it.
- An unrelated `combat_1v1` test regresses → revert the change that
  introduced the regression; the new env must not share mutable state
  with `combat_1v1`.

**Stage 2 (cap_duel self-play) failures:**

- BC produces zero score and PPO never moves off it → raise
  `bc_pretrain_steps` (500 → 1000) once, then if still flat, switch
  warm-start from `phase4_mappo_basic_v6_5` to
  `phase4_mappo_combat_1v1_v2` if/when that checkpoint exists; otherwise
  stop and report.
- Eval score moves but never crosses gate → reduce
  `time_penalty_per_decision` to 0 if non-zero, or raise `kill_bonus`
  (4.0 → 6.0). One tweak, then loop.
- Self-play collapses into mutual stalemate (both score 0 across all
  evals) → raise `current` weight in the self-play schedule (0.7 → 0.85)
  so anchor episodes are rarer. One tweak.
- LR feels too aggressive (oscillating eval score) → halve LR (1.0e-6 →
  5.0e-7). One tweak.

**Stage 3 (transfer probe) failures:**

- Transfer eval still scores 0 by update 50 → escalate. The cap_duel skill
  is not surviving the 3v3 reward gradient. Recommend one of: Strategy 3
  (focus-fire target conditioning, code change), or composition rehearsal
  with the new cap_duel teacher (config change once that path is unblocked).
- Transfer scores nonzero against `weak_basic_v2` but loses 50/50 vs
  `basic` → that is partial progress; CLEARED is reserved for the
  weak_basic_v2 anchor bar. Document and surface.

**Out of scope without explicit user approval:**

- Modifying C++ sim or `MatchConfig` schema.
- Modifying replay format.
- Adding new aux losses or new actor heads.
- Modifying the canonical Phase 4 reward.
- Modifying `phase_gate:` thresholds, metric names, or comparators on
  either stage gate.

## Stop Conditions

The loop stops when any of these are true:

- Stage 3 returns `CLEARED`.
- Stage 3 returns `HUMAN_INSPECTION_REQUIRED`.
- Two consecutive Stage 2 iterations return `NOT_CLEARED`.
- Two consecutive Stage 3 iterations return `NOT_CLEARED`.
- Total wall-clock training time exceeds 24 hours.
- Disk space under `runs/` drops below 10 GB free.
- A run returns `BLOCKED` and the cause is not config-fixable in under 30
  minutes (then surface to user).

## Verification Commands

Stage 1 (env build), before declaring Stage 1 complete:

```powershell
cd python
py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py `
  tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py `
  tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py -q
py -3.13 -m scripts.check_import_boundaries
```

Stage 2 / Stage 3, before launching each training run:

```powershell
cd python
py -3.13 -m pytest tests/test_phase4_mappo_env.py `
  tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py `
  tests/test_mappo_team_spirit_ramp.py -q
```

After each completed run, before invoking the gate:

```powershell
py -3.13 -m pytest tests/test_phase4_checkpoint_replay_dump.py -q
```

## Completion Criteria

The loop is complete when **either**:

1. A `gate_decision.json` with `status: CLEARED` exists for the Stage 3
   transfer config, the result is journaled, a phase-result doc is written
   under `docs/plans/archive/<date>-phase4-result.md`, and the user has
   been notified. **OR**

2. A stop condition has been hit, every stage and iteration is journaled
   with its gate decision artifact (or with an explicit explanation of why
   the gate was not reached), and a short hand-off note explains what
   was tried, what failed, and which escalation (Strategy 3 focus-fire,
   composition rehearsal once the combat teacher is rebuilt, or a different
   code-level change) is recommended next.

## Completion Metadata

When reporting completion, include:

```json
{
  "stages": [
    {
      "stage": "cap_duel_env_build",
      "files_added": [],
      "tests_run": [],
      "test_status": null,
      "regressions": []
    },
    {
      "stage": "cap_duel_selfplay",
      "iterations": [
        {
          "config_path": null,
          "git_commit": null,
          "seed": null,
          "wandb_run_url": null,
          "output_dir": null,
          "evidence_path": null,
          "gate_decision_path": null,
          "status": null,
          "failing_checks": [],
          "replay_artifacts": [],
          "viewer_command": null
        }
      ]
    },
    {
      "stage": "cap_duel_transfer",
      "iterations": []
    }
  ],
  "final_status": null,
  "stop_reason": null,
  "next_step_recommendation": null,
  "residual_risk": []
}
```

## Journal Entry Template

```
## YYYY-MM-DD — Phase 4 <stage> <config purpose> (iteration N)

**Config:** ../experiments/configs/.../<file>.yaml
**Git commit:** <hash>  **Seed:** <seed_base>
**W&B:** <url>  **Output:** runs/<output_dir>/
**Gate status:** CLEARED / NOT_CLEARED / ...
**Gate reason:** <decision.final_reason>
**Failing checks (if any):** id=value vs threshold
**Manifest summary:** best_eval_update_idx, score_a/score_b, kills, wins/losses
**Anchor transfer (Stage 3 only):** vs noop / weak_basic_v2 / basic — one line each
**Decision:** continuing with <next config or change> / stopping / awaiting review
```

## Good `/goal` Prompt

```text
Use GOAL_INSTRUCTIONS.md as the active goal. Solve Phase 4 by building the
cap_duel mini-game env, training it with current-vs-current self-play
warm-started from phase4_mappo_basic_v6_5, and then warm-starting a 3v3
transfer probe against weak_basic_v2. For each stage, follow the gate
loop: build evidence, invoke phase_gate.cli, journal the result, branch
per Decision Rules. Stage 1 acceptance is focused-pytest pass +
check_import_boundaries pass without regressing combat_1v1. Stage 2
clears when the cap_duel-specific gate (eval/mean_score_a >= 6.0,
eval/team_a_kills >= 5.0, eval/wins >= 25) passes. Stage 3 clears when
the canonical Phase 4 anchor gate passes (eval/mean_score_a >= 3.0 vs
weak_basic_v2, eval/wins >= 5, matrix transfer to basic, no losses to
noop). Do not weaken gate thresholds, do not change game/reward/obs/
action/replay/MAPPO-core or C++ sim semantics, do not commit, and do
not exceed two NOT_CLEARED iterations per stage or 24h of total training
wall time before stopping and reporting. On CLEARED, write a phase-result
doc under docs/plans/archive/ and return the completion metadata block.
```
