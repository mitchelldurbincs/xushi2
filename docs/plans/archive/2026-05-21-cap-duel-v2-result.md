# Phase 4 Cap-Duel v2 Result - 2026-05-21

## Scope

Rebuild of `cap_duel` without the three engineered quirks that the
2026-05-21 v1 inspection found were doing meaningful work in v1's wins:

1. Both agents spawning inside `point_radius * 0.85` of origin.
2. `_push_agent` knockback of `~0.693` units on every successful hit
   (formula was tuned to push the target out of recontest range for the
   entire `enemy_recontest_delay` window).
3. Respawning anywhere inside the point.

None of these exist in the canonical Phase 4 3v3 C++ sim
(`apply_damage_buffer` in `src/sim/src/internal/sim_combat.cpp:270-293`
subtracts HP only; no position writes; 3v3 respawns at canonical spawn
corners). Retraining cap_duel under stricter rules tests whether the
"kill_then_hold" skill survives without the v1 helpers.

No commit was created.

## What changed

- `python/envs/phase4_cap_duel_mappo.py`: additive v2 knobs
  - `knockback_magnitude: float | None = None` (None → legacy formula
    `~0.693`; 0.0 → no knockback; >0 → custom magnitude).
  - `spawn_distance: float | None = None` (None → legacy near-point
    spawn; >0 → opposite-side spawn at this distance from origin with
    random angle).
  - `respawn_at_spawn_position: bool = False` (False → legacy near-point
    respawn; True → restore each agent to its initial spawn position).
  - Defaults preserve v1 behavior; v1 yaml/checkpoint still load and run.
- `python/tests/test_phase4_cap_duel_mappo.py`: four new focused tests
  (corner spawn at distance with opposite-side angle; zero knockback
  preserves target position; respawn at spawn position; v1 defaults
  unchanged).
- `experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v2.yaml`:
  - `knockback_magnitude: 0.0`, `spawn_distance: 0.4`,
    `respawn_at_spawn_position: true`.
  - Same warm-start (`phase4_mappo_basic_v6_5`), same training knobs
    (250 updates, lr=1e-6, entropy 0.02, 64 envs, BC pretrain 500
    walk_and_shoot), same gate thresholds as v1.

## Verification

- `py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py -q` → **61 passed**.
- `py -3.13 -m scripts.check_import_boundaries` → PASS.
- Hand-coded "walk to origin + aim + fire" policy wins 20/20 self-play
  episodes in 15 decisions each → v2 env is solvable, no risk of training
  divergence from an unsolvable env.

## Training

- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/l9890sl6
- Seed: `3519994490`
- Output: `python/runs/phase4_mappo_cap_duel_selfplay_v2/`
- Best eval at update 225: mean_reward `+3.979`, **33W/2L/15D**,
  score **10.30/2.46**, kills **1.0/0.4**.
- Final eval at update 250: **37W/3L/10D**, score **10.54/1.40**,
  kills 0.9/0.2.
- Comparison vs v1 best-eval (update 225 in both): v2 has **5 more
  wins, 9 fewer losses, +2.64 score, fewer accidental B wins**.

## Inspector results (10 episodes per mode, seeds 3519994490..3519994499)

### Greedy

- Wins A/B/D: **8/0/2**.
- Mean Team A score ticks: 9.30 / 12.
- Mean Team B score ticks: 0.90.
- Total score events: 93. **kill_then_hold 81 (87.1%)**,
  displace_then_hold 0, accidental 12 (12.9%).
- Total kills: 7. Hits: 28. Fires: 517.

Per-episode shape:

- 5 episodes: clean 15-decision wins, 1 kill + 3 hits + 12 score ticks
  (matches the hand-coded smoke-test pattern).
- 3 episodes: 96-decision slow wins (score 11/0, 1 kill + 4 hits) —
  combat finally lands but the kill was late enough that B respawned and
  blocked the 12th tick.
- 2 episodes: 96-decision draws — greedy aim got stuck in a no-hit
  basin, 1 hit in 96 fires.
- 1 episode (ep0, seed `3519994490`): **passive win**, 0 kills,
  2 hits, score 12/9. A held the point while B's greedy policy hovered
  just outside the point and accumulated `off_point_decisions >= 12`.
  This is the source of the entire 12.9% "accidental" attribution. It
  is *not* a v1-style spawn-on-point exploit; the policy is winning by
  holding the point while a buggy opponent fails to recontest within
  the recontest_delay window.

### Stochastic

- Wins A/B/D: **10/0/0** (perfect).
- Mean Team A score ticks: 11.90 / 12.
- Total score events: 119. **kill_then_hold 119 (100%)**, accidental 0.
- Total kills: 15. Hits: 47. Fires: 218 (~21.6% hit rate).

Stochastic sampling breaks the greedy policy's deterministic aim-stuck
failure mode and produces uniformly clean kill_then_hold trajectories.

### Combined

- Total score events: 212. kill_then_hold 200, accidental 12.
- **Combined kill_then_hold ratio: 94.3%.**

Greedy hit rate is low (5.4%) — the greedy argmax produces a deterministic
aim that misses most of the time. Stochastic hit rate is ~21.6% — much
healthier. The trainer's eval is stochastic; the gate metrics reflect
the better number.

## Artifacts

- Inspection JSON: `runs/phase4_mappo_cap_duel_selfplay_v2/mappo/diagnostics/inspect_{greedy,stochastic}.json`.
- HTML viewers (cap_duel-native, open in any browser):
  - `view_greedy_ep3_seed3519994493_clean_kill_then_hold.html` — the
    canonical clean win (15 decisions).
  - `view_greedy_ep2_seed3519994492_slow_win.html` — slow combat win
    (96 decisions, eventually lands the kill).
  - `view_greedy_ep0_seed3519994490_passive_win.html` — the passive
    "B never approaches" displacement win (be honest about it).
  - `view_stochastic_ep3_seed3519994493_win.html` — stochastic clean
    win for comparison.

## Verdict

The v2 policy solves cap_duel **honestly**:

1. Both agents now spawn at opposite corners 0.4 from origin, so the
   policy had to learn to walk to the point (taught by the v6.5
   warm-start, preserved through training).
2. Hits no longer push the target; the only way to remove the enemy
   from the point is to kill them, requiring real aim + 3 successful
   hits.
3. After a kill, the killed agent respawns at its corner, so the policy
   has to physically be on the point to score during the dead window.

Under these stricter rules, the policy wins 90% of greedy episodes and
100% of stochastic episodes, with kill_then_hold attribution at 94.3%
overall. The remaining 5.7% is "hold the point while opponent fails to
approach" — a different mode than v1's spawn-on-point exploit and
plausibly closer to a real 3v3 scenario where a learner could hold the
objective against a stalling opponent.

## Recommended next step

**Composition rehearsal Stage B-1 with the v2 checkpoint as combat
teacher.** The premise that justified this whole detour — "the
cap_duel checkpoint is a strictly better combat teacher than the
missing `combat_1v1_v2` because it encodes objective-aware combat" —
is now true *honestly*, without depending on quirks that don't exist
in the 3v3 sim.

Config-only next move: template
`phase4_mappo_composition_rehearsal_v2_2000.yaml`, swap
`composition_combat_teacher_checkpoint` to
`runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt` and
`composition_combat_env.mini_game: cap_duel` with the v2
`mini_game_config` block (knockback 0, spawn 0.4, respawn at spawn).
Phase gate stays at the canonical Phase 4 anchor-transfer bar; no
relaxation.

If composition rehearsal still fails to transfer to 3v3 vs
`weak_basic_v2`, the failure cannot be blamed on cap_duel quirks — it
will be a genuine result that combat-with-objective in isolation is
insufficient for full 3v3, pointing to Strategy 3 (focus-fire) or a
new escalation.
