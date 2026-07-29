# Phase 4 `conversion_v2_respawn` — post-hoc eval (2026-07-29)

Sources: `runs/phase4_mappo_conversion_v2_respawn/` (run artifacts, 2026-07-11,
previously unanalyzed), `docs/reports/2026-07-09-phase4-3v3-review-recommendations.md`,
`experiments/configs/phase4/probe/phase4_mappo_conversion_v2_respawn.yaml`,
matrix-eval sweeps and greedy behavior probes run today. All sweeps: 20
checkpoints (`ckpt_0025`–`ckpt_0500`) plus the June bridge warm start
(`data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt`), vs
{`weak_basic_v2`, `noop`, `basic`}, 50 episodes/cell, greedy.

## TL;DR

The July 11 combined run **did produce the first checkpoint in project history
that beats `weak_basic_v2` with nonzero objective score at canonical settings**
(15s unlock / 8s capture / 240t respawn): `ckpt_0100` = `ckpt_best_eval`,
50/0/0 with score 1.30 vs the bridge's 0.00. The July 9 three-part diagnosis
(respawn treadmill, cold critic, missing anchor) is validated. But the margin
is small, only 3 of 20 checkpoints score at canonical at all (100: 1.30,
275: 0.97, 350: 0.10), and the second half of the respawn anneal *degraded*
conversion instead of consolidating it. Four additional findings below change
how the next run should be evaluated before it is launched.

## Finding 1 — every post-training eval of this run was on easy mode

Checkpoints serialize the env's **resolved initial sim config**: curriculum
initial values `objective_unlock_seconds: 5.0`, `objective_capture_seconds:
2.0`, `mechanics.respawn_ticks: 2400` are baked into `ckpt["config"]["env"]["sim"]`.
Post-training matrix eval, `scripts/eval_mappo_matrix.py`, and
`scripts/dump_replay.py` all rebuild envs from checkpoint config, so every one
of them evaluated at eased settings without saying so. The run's
`transfer_summary.md` headline (50/0/0, score 3.70 vs weak) was measured with
respawns effectively off and 2s captures. The `gate_status:
evidence_insufficient` verdict was doubly meaningless: eased env + blind
funnel columns (Finding 2).

This is the same *class* of bug as the July 9 setter-drop: curriculum state
and eval state sharing one config with no explicit "canonical" contract.

## Finding 2 — the transfer gate read all-zero funnel stats by construction

`mappo_matrix_row()` emitted only ~10 hand-picked fields; the transfer summary
reads `mean_majority_on_point_seconds_*`, `mean_uncontested_on_point_seconds_*`,
`mean_cap_progress_gain_ticks` via `.get(..., 0.0)`. Every funnel number in
every transfer summary was a default zero — visibly impossible in the July 11
artifact itself (team B: 45.07 score with "0.0 s" majority vs `basic`).
**Fixed today**: the row now spreads the full `eval_stats_dict`
(`python/train/mappo_matrix_eval.py`); `tests/test_mappo_matrix_eval.py` 6/6.

## Finding 3 — canonical curve: real but fragile conversion, and the anneal tail un-learns it

Canonical (unlock 15s / capture 8s / respawn 240t), greedy, score vs
`weak_basic_v2` (draws are 0-0 stalemates; W/L/D over 50 identical episodes —
see Finding 5):

| ckpt | W/L/D | scoreA | majA s | uncA s | capG | capL | kills |
|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge | 0/0/50 | 0.00 | 17.6 | 7.5 | 284 | 246 | 10-0 |
| 0025 | 0/0/50 | 0.00 | 17.6 | 7.5 | 284 | 246 | 10-0 |
| 0050 | 0/0/50 | 0.00 | 29.1 | 4.1 | 217 | 178 | 13-0 |
| 0075 | 0/0/50 | 0.00 | 12.0 | 3.6 | 300 | 91 | 5-0 |
| **0100** | **50/0/0** | **1.30** | 22.6 | 10.2 | 315 | 270 | 8-0 |
| 0125 | 0/0/50 | 0.00 | 37.0 | 5.2 | 247 | 220 | 15-0 |
| 0150 | 0/0/50 | 0.00 | 41.3 | 8.6 | 340 | 309 | 17-0 |
| 0175 | 0/0/50 | 0.00 | 30.0 | 1.6 | 116 | 110 | 13-0 |
| 0200 | 0/0/50 | 0.00 | 33.8 | 3.9 | 233 | 130 | 16-0 |
| 0225 | 0/0/50 | 0.00 | 32.4 | 1.1 | 148 | 106 | 16-0 |
| 0250 | 0/0/50 | 0.00 | 17.2 | 7.0 | 226 | 210 | 7-0 |
| **0275** | **50/0/0** | **0.97** | 29.6 | 9.5 | 261 | 261 | 9-0 |
| 0300 | 0/0/50 | 0.00 | 28.7 | 6.6 | 278 | 242 | 17-0 |
| 0325 | 0/0/50 | 0.00 | 20.8 | 0.5 | 93 | 0 | 7-0 |
| **0350** | **50/0/0** | **0.10** | 21.3 | 8.6 | 378 | 290 | 12-0 |
| 0375 | 0/0/50 | 0.00 | 44.5 | 6.2 | 261 | 153 | 21-0 |
| 0400 | 0/0/50 | 0.00 | 26.4 | 6.0 | 273 | 219 | 16-0 |
| 0425 | 0/0/50 | 0.00 | 11.8 | 3.6 | 185 | 92 | 8-0 |
| 0450 | 0/0/50 | 0.00 | 36.8 | 9.0 | 325 | 324 | 19-0 |
| 0475 | 0/0/50 | 0.00 | 40.2 | 8.4 | 335 | 311 | 21-0 |
| 0500 | 0/0/50 | 0.00 | 38.8 | 4.6 | 180 | 135 | 20-0 |

Reading:
- `bridge` == `ckpt_0025` bit-for-bit in behavior (and the first 25 updates
  were value-only critic warmup — the actor freeze worked as designed).
- `ckpt_0100` (= `ckpt_best_eval` by state-dict hash) is the canonical peak:
  ~10s uncontested → one full capture → 1.3 score ticks/round. First canonical
  score ever.
- After ~upd 300 the kill counts climb (16–21 per round) while conversion
  dies — as the annealed respawn pressure returned, the policy drifted back to
  the kill treadmill despite `kill_bonus: 0.1`. The anneal was too fast to
  consolidate defend-and-hold (200 updates), matching the config's own
  falsification warning.
- vs `basic`: 0/50 at every checkpoint (basic scores 15–35). Untouched, as
  expected — ladder work is Step 3.
- At eased settings (for reference): 50/0 vs weak at 17 of 20 checkpoints,
  score 1.1–8.8, three total-collapse checkpoints (50, 175, 375 — distinct
  weights, identical failure trajectory where *weak* out-scores).

The in-training best-eval score (8.07 at upd 100) is not comparable to any of
the above: at update 100 the training env's respawn was mid-anneal (~1320t),
so the eval that selected `ckpt_best_eval` ran on a different difficulty than
either sweep. It still picked the canonical-best checkpoint.

## Finding 4 — the noop "anomaly" is the policy's actual objective model

Greedy behavior probe at canonical settings (flat-obs capture per decision),
`ckpt_best_eval` vs `noop`: the squad crosses the map, walks **past** the
objective, parks at the enemy spawn (~team-frame (+0.35, +1.00)), and stands
there for the rest of the round — zero shots fired, zero captures, 0-0 draw.
Every checkpoint from 0025 to 0500 scores 0.00 vs noop at both difficulty
settings. vs `weak_basic_v2` the same policy holds the point 75–100% of the
round and converts after team wipes.

Interpretation: there is no unconditional "stand on the unlocked point"
behavior anywhere in the population. Point play is entirely cued by enemy
contact and combat flow. This is a latent failure for the planned self-play
stage — an opponent that refuses contact (or a passive league snapshot)
zeroes the policy's offense. Watchable replays:
`runs/phase4_mappo_conversion_v2_respawn/replays/best_eval_canonical_vs_{noop,weak_basic_v2,basic}.replay`.

## Finding 5 — greedy eval on the fixed map is a single sample

Reset seeds 11 and 12 produce bit-identical trajectories (fixed spawns,
`randomize_map: false`, deterministic scripted bots, greedy policy, no
stochastic sim events). Every 50-episode greedy eval cell in project history
is therefore **one effective sample repeated 50×** — which is why W/L columns
are always 50/0, 0/50, or 0/0/50, and why the July 11 transfer score (3.70)
reproduced today to the digit under a different seed base. All-or-nothing
gate flips between adjacent checkpoints (e.g. 0250 → 0275 → 0300) are one
trajectory bifurcating, not a distribution shifting.

## Recommendations (ordered)

1. **Canonical eval contract** — either the trainer writes curriculum *final*
   values into the checkpoint sim config, or matrix eval applies explicit
   canonical overrides (today's sweep wrapper does the latter; adopt one).
   Without this, every future curriculum run mis-reports transfer.
2. **Eval episode diversity** — stochastic eval actions (or spawn/seed
   randomization in the sim) before trusting any gate; report mean ± spread
   over genuinely distinct episodes.
3. **Next training run** (the July 9 plan survives contact with this data):
   slower/step-held respawn anneal (200 updates was too fast — canonical
   conversion existed at upd 100 and was gone by 125), consider re-anchoring
   on `ckpt_0100` for the anneal tail, and add a small no-contact curriculum
   mix (episodes vs `noop`/passive bots) so "walk onto the unlocked point and
   stand" exists as a behavior at all.
4. Funnel-stat fix is in; the transfer gate can now actually gate on
   uncontested seconds and captures.
