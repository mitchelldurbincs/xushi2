# Goal: Composition Rehearsal With The v2 Cap-Duel Checkpoint

## Status Going Into This Goal

- The 2026-05-21 cap_duel v1 inspection found 100% kill_then_hold in v1
  but discovered that v1's score path leaned on three engineered env
  quirks not present in the canonical Phase 4 3v3 C++ sim (spawn-on-
  point, shot knockback ~0.693 units per hit, respawn-on-point).
- Cap_duel v2 (`phase4_mappo_cap_duel_selfplay_v2`) retrained from the
  same v6.5 warm-start under stricter rules (knockback 0, corner
  spawns at 0.4, respawn-at-spawn). v2 wins 8/10 greedy + 10/10
  stochastic with 94.3% combined kill_then_hold and zero v1-style
  spawn-on-point exploits. Full results in
  `docs/plans/archive/2026-05-21-cap-duel-v2-result.md` and the
  2026-05-21 v2 journal entry.

The combat teacher that the prior plan needed but couldn't have
(because `combat_1v1_v2` is missing on this machine) is now real.

## What This Goal Does

Composition rehearsal with the v2 cap_duel checkpoint as the combat
teacher, then a 3v3 transfer probe against `weak_basic_v2`. Same gate
as the original Strategy 1 path; no thresholds relaxed.

## Non-Negotiables (carried over)

- Do not change sim rules, reward formulas, observation layout, action
  semantics, replay format, deterministic sim, W&B metric schemas, or
  MAPPO core. Do not modify C++.
- Do not weaken phase gates. Each stage's `phase_gate:` block is the bar.
- Do not skip the journal entry per run.
- Do not launch a new training run while one is in flight.
- Do not commit changes unless explicitly asked.
- Out of scope without explicit user approval: new aux losses, new actor
  heads, Strategy 3 (focus-fire) code changes, retraining `combat_1v1_v2`
  on this machine.

## First Reads

1. `docs/plans/archive/2026-05-21-cap-duel-v2-result.md` — what changed
   in v2 and what the inspection confirmed.
2. The 2026-05-21 v2 journal entry (the most recent in
   `docs/journal/reinforcement_learning_journal.md`).
3. `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v2_2000.yaml` — the rehearsal config template.
4. `experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v2.yaml` — copy the `mini_game_config` block verbatim into the rehearsal combat env so the rehearsal samples match the teacher's training distribution.
5. `python/train/composition_rehearsal.py` and
   `python/train/mappo_pretrain_hooks.py` — the pretrain path.
6. `python/train/phase_gate/README.md` — gate CLI shape.

## Stage 1 — Composition Rehearsal Cap-Duel v2

Write `experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml`
by templating off `phase4_mappo_composition_rehearsal_v2_2000.yaml`.
Required differences:

- Student warm-start stays `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`.
- Objective teacher stays the same v6.5 checkpoint with
  `composition_objective_env.opponent_bot: weak_basic_v2`.
- **Combat teacher swaps to**
  `composition_combat_teacher_checkpoint: runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt`.
- **`composition_combat_env` swaps to cap_duel v2** by mirroring the
  v2 yaml's `mini_game_config` block verbatim (point_radius 0.18,
  knockback_magnitude 0.0, spawn_distance 0.4,
  respawn_at_spawn_position true, episode_decisions 96, enemy_hp 3,
  score_ticks_to_clear 12, enemy_recontest_delay 12, hit_tolerance
  0.12, hit_reward 1.0, kill_bonus 4.0, score_per_tick 0.1,
  off_point_penalty 0.0, time_penalty_per_decision 0.0).
- Keep `composition_pretrain_steps: 2000`,
  `composition_objective_batch_size: 256`,
  `composition_combat_batch_size: 256`. Do not shrink — the May-18
  failure modes include "BC composition didn't run long enough to
  bind both skills."
- After composition rehearsal: full Phase 4 3v3 PPO with the same
  conservative knobs the cap_duel transfer used: `num_envs: 64`,
  `rollout_len: 128`, `learning_rate: 1.0e-6`, `entropy_coef: 0.02`,
  `gamma: 0.997`, `gae_lambda: 0.95`, `lr_schedule: cosine`,
  `lr_final_ratio: 0.1`, `value_normalization: true`. No `bc_pretrain_*`
  (composition rehearsal *is* the pretrain).
- `env.opponent_bot: weak_basic_v2`, `env.self_play.enabled: false`.
  No `mini_game` flag on the PPO env (only on the rehearsal combat env).
- `run.total_updates: 200`, `eval_every: 25`, `checkpoint_every: 25`,
  `output_dir: runs/phase4_mappo_composition_rehearsal_cap_duel_v2`.

Stage 1 `phase_gate:` block is the canonical Phase 4 anchor-transfer
bar. Do not weaken any threshold:

```yaml
phase_gate:
  phase: phase4_composition_rehearsal_cap_duel_v2
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
        prompt: "In a greedy 3v3 replay vs weak_basic_v2, does the
                 kill-or-displace-then-hold behavior visible in cap_duel
                 v2 still happen in the full 3v3, or has full-env PPO
                 erased it?"
```

**Pre-PPO kill-switch.** Before letting the PPO loop run, the
composition rehearsal path produces a post-BC eval. Watch:

- Team A `hit_fire` on `weak_basic_v2`: should be visibly above `0.04`.
  If at `~0.0145` like the failed cap_duel transfer v1's update-25
  eval, composition rehearsal didn't preserve combat and PPO won't
  rescue it — stop and apply the Stage 1 fallback below.

Post-training: matrix eval vs `noop`/`weak_basic_v2`/`basic`, dump one
greedy and one stochastic 3v3 replay, build evidence, invoke gate,
journal.

## Decision Rules

- Post-BC `hit_fire` collapsed below `0.04` before PPO starts →
  composition rehearsal failed to preserve combat. Raise
  `composition_pretrain_steps` (2000 → 4000) once and verify the
  `composition_combat_env.mini_game_config` block matches the cap_duel
  v2 yaml field-for-field. One tweak, then loop.
- Post-BC OK but PPO drifts and final `hit_fire` ends below `0.04`
  with score 0 → reduce LR (`1.0e-6 → 5.0e-7`) once. If that also
  fails, surface for user decision (a distillation-anchor pass during
  PPO would attack this directly but is code-change scope).
- Post-BC OK, PPO holds combat, learner still loses objective race vs
  `weak_basic_v2` → raise `composition_objective_batch_size`
  (256 → 512) once.
- `weak_basic_v2_wins` > 0 but < 5 while score clears 3.0 → partial
  progress; document and surface, do not relax the gate.

## Stop Conditions

- Stage 1 returns `CLEARED` or `HUMAN_INSPECTION_REQUIRED`.
- Two consecutive Stage 1 `NOT_CLEARED` iterations.
- Total wall-clock training time exceeds 24 hours.
- A run returns `BLOCKED` and the cause is not config-fixable in under
  30 minutes (surface to user).
- Disk space under `runs/` drops below 10 GB free.

## Verification Commands

Before launching:

```powershell
cd python
py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py `
  tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py `
  tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py `
  tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py `
  tests/test_mappo_team_spirit_ramp.py -q
py -3.13 -m scripts.check_import_boundaries
```

If `test_mappo_composition_rehearsal.py` does not yet exercise the
`mini_game: cap_duel` combat env path, add one focused parameterization
(tensor shape + finite BC loss + verifies v2 knobs are honored) before
launching. A single focused test counts as bug-fix scope.

After the completed run, before the gate:

```powershell
py -3.13 -m pytest tests/test_phase4_checkpoint_replay_dump.py -q
```

## Completion Criteria

- `gate_decision.json` with `status: CLEARED` or `HUMAN_INSPECTION_REQUIRED`
  exists for the Stage 1 composition-rehearsal-v2 config, the result
  is journaled, a phase-result doc is written under
  `docs/plans/archive/<date>-phase4-composition-cap-duel-v2-result.md`,
  and the user has been notified, **OR**
- A stop condition has been hit, every iteration is journaled with its
  gate decision artifact (or an explicit explanation why it wasn't
  reached), and a short hand-off note explains what was tried and what
  to try next.

## Good `/goal` Prompt

```text
Use GOAL_INSTRUCTIONS.md as the active goal. Stage 1: composition
rehearsal with cap_duel v2 as the combat teacher. Student warm-start
phase4_mappo_basic_v6_5; objective teacher stays
phase4_mappo_basic_v6_5; combat teacher
runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt;
composition_combat_env.mini_game=cap_duel with the v2 mini_game_config
block mirrored verbatim (knockback_magnitude 0, spawn_distance 0.4,
respawn_at_spawn_position true). composition_pretrain_steps=2000.
Then 200 PPO updates in full 3v3 vs weak_basic_v2 at lr=1e-6,
entropy 0.02, no bc_pretrain. Kill-switch: if post-BC team_a_hit_fire
on weak_basic_v2 < 0.04, stop and apply the Stage 1 fallback. Gate
is the canonical Phase 4 anchor-transfer bar (mean_score_a >= 3.0,
wins >= 5, hit_fire >= 0.04, matrix vs basic >= 1.0, no losses to
noop). Do not weaken thresholds, do not change game/reward/obs/action/
replay/MAPPO-core or C++ sim, do not add aux losses or actor heads,
do not commit. Stop on CLEARED or HUMAN_INSPECTION_REQUIRED, after two
NOT_CLEARED iterations, or 24h total training wall time. On CLEARED,
write a phase-result doc under docs/plans/archive/ and return the
completion metadata block.
```
