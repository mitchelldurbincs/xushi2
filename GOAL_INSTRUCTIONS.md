> **STATUS 2026-08-02:** This protocol targets the ORIGINAL Phase-4 gate and
> predates the 2026-07-30 gate redefinition (stochastic conversion at 600t —
> CLEARED; see docs/journal/reinforcement_learning_journal.md). Treat the
> journal as authoritative for gate accounting; this document is retained
> for the supervision-protocol mechanics only.

# Goal: PPO-Time Cap-Duel v2 Distillation Anchor For Phase 4

## Status Going Into This Goal

Phase 4 is still the active focus. We finally got a real combat/objective
teacher, but the first transfer attempt failed in a useful way.

Known facts:

- `phase4_mappo_cap_duel_selfplay_v2` is an honest cap-duel teacher.
  It removed the v1 engineered quirks (spawn-on-point, shot knockback,
  respawn-on-point), trained from `phase4_mappo_basic_v6_5`, and still
  produced strong kill-then-hold behavior.
- The result is documented in
  `docs/plans/archive/2026-05-21-cap-duel-v2-result.md` and the
  2026-05-21 journal entry.
- The completed composition-rehearsal attempt is documented in
  `docs/plans/archive/2026-05-21-phase4-composition-cap-duel-v2-result.md`
  and the latest entries of `docs/journal/reinforcement_learning_journal.md`.
- Composition rehearsal with cap_duel v2 failed before PPO:
  - 2000-step run:
    `full_hit_fire=0.0047 < 0.0400`
  - 4000-step fallback:
    `full_hit_fire=0.0116 < 0.0400`
  - Both also failed objective/contact and combat-kill retention.
- Matrix eval after both skipped-PPO runs still produced `0.00` Team A
  score against `weak_basic_v2` and `basic`.

Conclusion: do **not** spend another config-only run on more composition
rehearsal length. The failure mode is not "needs a few more BC steps"; the
cap_duel skill is not staying bound to the full 3v3 distribution.

## What This Goal Does

Add a PPO-time distillation anchor so full Phase 4 PPO can keep learning
against `weak_basic_v2` while a small auxiliary loss continuously preserves
the cap_duel v2 teacher's aim/fire behavior on combat samples.

This is intentionally code-scope. The previous goal made clear that a
one-shot pretrain is too fragile. The next most direct attempt is to keep the
teacher attached during PPO instead of hoping BC survives the full-env
gradient.

## Recommended Order

1. Add diagnostics that explain the post-BC failure more clearly.
2. Add the PPO-time cap_duel distillation anchor.
3. Run a short smoke/probe to prove the anchor is active and metrics are
   logged.
4. Run the Stage 1 transfer config against `weak_basic_v2`.
5. If this still fails, stop and recommend Strategy 3 focus-fire target
   conditioning rather than another rehearsal-length tweak.

## Non-Negotiables

- Do not change C++.
- Do not change sim rules, reward formulas, observation layout, action
  semantics, replay format, determinism behavior, W&B existing metric names,
  or the phase-gate thresholds.
- Do not weaken any `phase_gate:` block.
- Do not launch a new training run while one is in flight.
- Do not commit changes unless explicitly asked.
- Do not retrain `combat_1v1_v2`.
- Do not add a new actor head unless the user explicitly approves Strategy 3.
- Do not hide failed diagnostics. A bad metric result is done, not blocked.

Allowed scope for this goal:

- Python trainer/eval code.
- Python tests.
- New probe config(s) under `experiments/configs/phase4/probe/`.
- New diagnostics/scripts if they are narrow and support this experiment.
- New W&B metrics under a new prefix such as `distill/*` or
  `cap_duel_anchor/*`.

## First Reads

1. `docs/journal/reinforcement_learning_journal.md` — read the latest entries.
2. `docs/plans/archive/2026-05-21-cap-duel-v2-result.md`.
3. `docs/plans/archive/2026-05-21-phase4-composition-cap-duel-v2-result.md`.
4. `experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v2.yaml`.
5. `experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml`.
6. `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml`.
7. `python/train/composition_rehearsal.py`.
8. `python/train/mappo_pretrain_hooks.py`.
9. `python/train/mappo_rollout_trainer.py` and the PPO loss/update path it uses.
10. `python/train/phase_gate/README.md`.

If the docs and code disagree, ask before changing load-bearing behavior.

## Stage A — Diagnostic First

Before adding the anchor, add or extend a focused diagnostic that can evaluate
a checkpoint/policy on:

- objective-only/full weak_basic_v2 behavior,
- cap_duel v2 behavior,
- teacher/student agreement on the action heads that matter:
  - movement rows if relevant,
  - aim row,
  - primary_fire binary head.

Minimum useful output:

- objective on-point,
- objective losses,
- cap_duel kills,
- full-env Team A hit/fire,
- full-env aim error,
- aim MSE vs cap_duel teacher,
- fire BCE or fire agreement vs cap_duel teacher,
- mean teacher fire probability and mean student fire probability.

This can be a script, a helper plus tests, or both. Keep it narrow.

## Stage B — PPO-Time Cap-Duel Distillation Anchor

Implement an optional PPO-time auxiliary loss controlled entirely by config.

Suggested config shape:

```yaml
run:
  cap_duel_distill:
    enabled: true
    teacher_checkpoint: runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt
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
        knockback_magnitude: 0.0
        spawn_distance: 0.4
        respawn_at_spawn_position: true
    batch_size: 256
    every_updates: 1
    coef: 0.05
    aim_coef: 1.0
    fire_coef: 1.0
```

Implementation requirements:

- The feature must be off by default.
- Teacher checkpoint must be frozen.
- The cap_duel v2 mini-game config must be mirrored field-for-field from
  `phase4_mappo_cap_duel_selfplay_v2.yaml`.
- The anchor should train only the relevant actor behavior:
  - aim delta,
  - primary fire,
  - optionally movement if diagnostics show it helps.
- Do not alter PPO advantage calculation, rewards, env stepping, or action
  semantics.
- Log separate metrics, for example:
  - `distill/loss`
  - `distill/aim_loss`
  - `distill/fire_loss`
  - `distill/student_fire_prob`
  - `distill/teacher_fire_prob`
  - `distill/updates`
- Add NaN/finite checks for the distillation loss.
- Make the code path deterministic under the run seed.

## Stage C — Probe Config

Create a new config, suggested name:

```text
experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml
```

Base it on the conservative full-env transfer settings:

- `env.opponent_bot: weak_basic_v2`
- `env.self_play.enabled: false`
- student warm-start:
  `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`
- teacher:
  `runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt`
- `num_envs: 64`
- `rollout_len: 128`
- `learning_rate: 1.0e-6`
- `entropy_coef: 0.02`
- `gamma: 0.997`
- `gae_lambda: 0.95`
- `lr_schedule: cosine`
- `lr_final_ratio: 0.1`
- `value_normalization: true`
- `total_updates: 200`
- `eval_every: 25`
- `checkpoint_every: 25`
- no `bc_pretrain_*`
- no `composition_pretrain`

Include matrix eval vs:

- `noop`
- `weak_basic_v2`
- `basic`

Use the same Phase 4 anchor-transfer gate:

```yaml
phase_gate:
  phase: phase4_cap_duel_distill_anchor_v1
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
      - id: replay_kill_then_hold_3v3
        prompt: >
          In a greedy 3v3 replay vs weak_basic_v2, does Team A create a
          combat advantage, kill or displace enemies, and hold the point
          long enough to convert score without obvious reward hacking?
```

## Stage D — Run And Gate

Before launch:

- Check no training run is in flight.
- Check disk space under `runs/` is above 10 GB.
- Run focused tests plus import boundary checks.

Launch from `python/`:

```powershell
py -3.13 -m train.train --config ../experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml
```

After training:

1. Run matrix eval if the config did not already produce it.
2. Dump one greedy and one stochastic full-3v3 replay.
3. Build evidence.
4. Invoke the phase gate.
5. Journal the run.
6. Write a result doc under
   `docs/plans/archive/<date>-phase4-cap-duel-distill-anchor-result.md`.

## Decision Rules

- Distillation loss is NaN or non-finite -> stop, fix the implementation,
  rerun tests. This is blocked until code is fixed.
- Distillation metrics show the anchor is inactive
  (`distill/updates=0`, missing fire/aim losses, or teacher not loaded) ->
  fix before launching a long run.
- Eval `team_a_hit_fire` never reaches `0.04` by update 50 and score remains
  `0.00` -> stop early and report. The anchor did not solve the same failure
  mode.
- Hit/fire clears `0.04` but score remains `0.00` through update 100 ->
  finish the run once, then report whether the bottleneck is objective
  conversion rather than combat preservation.
- Score clears `3.0` but wins remain below `5` -> document partial progress;
  do not relax the gate.
- If this distillation-anchor run fails cleanly, recommend Strategy 3
  focus-fire target conditioning as the next code-scope goal.

## Stop Conditions

- Gate returns `CLEARED`.
- Gate returns `HUMAN_INSPECTION_REQUIRED`.
- The run reaches a documented early-stop decision rule.
- One full distillation-anchor run completes and returns `NOT_CLEARED`.
- A run returns `BLOCKED` and the cause is not fixable in under 30 minutes.
- Total training wall-clock time exceeds 24 hours.
- Disk space under `runs/` drops below 10 GB free.

## Verification Commands

Before launch, run at least:

```powershell
cd python
py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py `
  tests/test_phase4_mappo_env.py tests/test_mappo_matrix_eval.py `
  tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py `
  tests/test_mappo_team_spirit_ramp.py -q
py -3.13 -m scripts.check_import_boundaries
```

Add focused tests for the new anchor. Minimum expectations:

- Config defaults keep the feature disabled.
- Enabled config loads the frozen teacher.
- One anchor update produces finite `loss`, `aim_loss`, and `fire_loss`.
- Teacher parameters receive no gradients.
- PPO loss path includes the auxiliary only when enabled.
- The cap_duel v2 mini-game block is honored.

After training, run the current replay-dump smoke suite. In this checkout the
old path `tests/test_phase4_checkpoint_replay_dump.py` is stale; use:

```powershell
py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q
```

If the file layout changes, use the current test that covers Phase 4 MAPPO
checkpoint replay dumping and cite the exact command.

## Completion Criteria

The goal is complete when one of these is true:

1. `gate_decision.json` exists with `status: CLEARED` or
   `HUMAN_INSPECTION_REQUIRED`, the run is journaled, a result doc exists,
   and replay artifacts are listed.
2. The configured early-stop rule fires, the run is journaled, a result doc
   explains the failed metric and next recommended code-scope move, and no
   required artifact is silently missing.
3. One full distillation-anchor run completes with `NOT_CLEARED`, the result
   is journaled, a result doc is written, and the next recommendation is
   explicit.

Use completion metadata in this shape:

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

```text
Use GOAL_INSTRUCTIONS.md as the active goal. We already proved cap_duel v2 is
an honest kill-then-hold teacher, but composition rehearsal with 2000 and 4000
steps failed before PPO (`full_hit_fire` stayed below 0.04). Do not try more
config-only rehearsal length. Implement a PPO-time cap_duel v2 distillation
anchor, off by default and controlled by config, that freezes
runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt and adds a small
aim/fire auxiliary loss during full Phase 4 PPO against weak_basic_v2. Do not
change C++, sim rules, rewards, obs/action semantics, replay format,
determinism, existing W&B metric names, or gate thresholds. Add focused tests,
create phase4_mappo_cap_duel_distill_anchor_v1.yaml, run the focused suite and
import-boundary check, then run the probe. Stop on CLEARED,
HUMAN_INSPECTION_REQUIRED, early stop if hit_fire remains below 0.04 with zero
score by update 50, or one completed NOT_CLEARED run. Journal every run and
write a result doc under docs/plans/archive/.
```
