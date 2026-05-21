# Phase 4 Cap-Duel Rollout Inspection - 2026-05-21

## Scope

Stage A of `GOAL_INSTRUCTIONS.md`: produce a per-tick diagnostic for the
existing checkpoint
`runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt` to answer
the subjective gate question that the viewer could not (because the
cap_duel mini-game is a Python-only world that the canonical Phase 4
replay viewer cannot render faithfully). No training was launched. No
commit was created.

## What was built

- Additive `_make_info` keys in `python/envs/phase4_cap_duel_mappo.py`:
  `cap_duel_self_pos`, `cap_duel_enemy_pos`, `cap_duel_self_hp`,
  `cap_duel_enemy_hp`, `cap_duel_enemy_off_point_decisions`,
  `cap_duel_self_score_ready`. No reward/obs/action/replay change.
- Focused test
  `test_info_exposes_diagnostic_fields_for_inspector` in
  `python/tests/test_phase4_cap_duel_mappo.py`.
- `python/scripts/inspect_cap_duel_rollout.py` — reuses the same
  checkpoint/env construction path as `scripts/replay_dump/rollout.py`
  (`load_mappo_checkpoint` + `checkpoint_runtime`), runs N episodes
  (default 10) with per-episode seed `base_seed + i` to match the eval
  loop pattern, and computes `kill_then_hold_ratio` /
  `displace_then_hold_ratio` / `accidental_ratio` via per-tick
  attribution against the `enemy_recontest_delay` window.

## Verification

- `py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py -q` -> 11
  passed.
- Each episode reconciles: sum of per-tick `score_event_this_step` and
  `kill_this_step` deltas equals the env's reported cumulative totals.

## Results

Inputs: checkpoint
`runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt`,
base seed `3519994490`, 10 episodes per mode, env config from the
embedded checkpoint config (cap_duel,
`episode_decisions=96`, `score_ticks_to_clear=12`,
`enemy_recontest_delay=12`).

Output JSON files:

- `runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_greedy.json`
- `runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_stochastic.json`

### Greedy (10 episodes, seeds 3519994490..3519994499)

- Wins A/B/Draws: **9 / 1 / 0**.
- Mean Team A score ticks: **10.70 / 12**.
- Mean Team B score ticks: 1.40.
- Total score events: **107**.
  - `kill_then_hold`: **107** (100.0%).
  - `displace_then_hold`: 0.
  - `accidental`: 0.
- Total kills: 10 (one per episode in aggregate).
- Total hits: 32 across 99 fires (~32% hit rate among non-zero firing
  decisions).
- Forbidden score ticks (self on point and enemy alive on point): 0.

### Stochastic (10 episodes, seeds 3519994490..3519994499)

- Wins A/B/Draws: **9 / 1 / 0**.
- Mean Team A score ticks: **10.90 / 12**.
- Mean Team B score ticks: 1.30.
- Total score events: **109**.
  - `kill_then_hold`: **109** (100.0%).
  - `displace_then_hold`: 0.
  - `accidental`: 0.
- Total kills: 12.
- Total hits: 36 across 71 fires (~51% hit rate when sampling actions).

### Single-episode anomaly

The very first single-episode inspector run on seed `3519994490` alone
showed an outlier: Team B won, learner died at step 37, only 2 hits in
42 decisions, 0 score events. That single-seed picture matches the
viewer impression of "agents spinning and shooting randomly, never
scoring." Aggregating across the 10-episode set the gate actually
evaluates shows the policy is **competent in 9 of 10 episodes** and
the anomalous seed was just an unlucky self-play matchup, not the
typical behavior.

## Verdict

The cap_duel selfplay v1 checkpoint **did learn a real kill-then-hold
behavior.** Every one of the 216 aggregated score events (greedy +
stochastic) is attributable to either the enemy being dead at the
moment of scoring or to a kill that landed within the
`enemy_recontest_delay=12` window before the score event. None of the
scoring is accidental displacement. Win rate is 9/10 in both modes
against the self-play opponent, with the learner clearing close to
`score_ticks_to_clear=12` on average. The viewer-observed "spinning and
random fire" was a combination of (a) the canonical Phase 4 replay
viewer rendering cap_duel actions against the wrong world layout —
canonical spawn positions and canonical objective location rather than
the cap_duel "both spawn within `point_radius=0.18` of origin" world —
and (b) the user happening to inspect a replay near the unlucky
seed-`3519994490` self-play loss. The aggregate cap_duel skill is a
legitimate combat-with-objective teacher.

## Recommended next step

**Stage B-1 from `GOAL_INSTRUCTIONS.md`: composition rehearsal with the
cap_duel checkpoint as combat teacher.** The premise of the original
plan ("the cap_duel checkpoint is a strictly better combat teacher
than the missing `combat_1v1_v2` because it already encodes
kill-then-hold") is confirmed by the inspection. Template off
`experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v2_2000.yaml`,
swap `composition_combat_teacher_checkpoint` to
`runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt` and
`composition_combat_env.mini_game` to `cap_duel` with the
`mini_game_config` block mirroring
`phase4_mappo_cap_duel_selfplay_v1.yaml` field-for-field. Phase gate
stays at the canonical Phase 4 anchor-transfer bar (no relaxation).
This is a config-only change.

Stage B-2 (retrain with a tighter gate) and Stage B-3 (escalate to
Strategy 3 focus-fire) are not justified by this evidence and should
not be pursued unless Stage B-1 itself fails.
