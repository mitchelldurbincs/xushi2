# Phase 4 (3v3) Plateau Review — 2026-07-09

Sources: `docs/journal/reinforcement_learning_journal.md` (all entries through
2026-06-10), `docs/reports/2026-06-09-phase4-breakthrough-analysis.md`,
`src/sim/src/internal/sim_objective.cpp`, `tests/sim/test_respawn.cpp`,
`python/train/mappo_runtime_context.py`, `python/train/mappo_evaluate.py`,
`experiments/configs/phase4/probe/phase4_mappo_conversion_v1.yaml`.

---

## Where things actually stand

The 2026-06-09 breakthrough analysis was right about the diagnosis, and the
2026-06-10 runs sharpened it further. The current frontier checkpoint
(closed-loop bridge, multi-enemy entity-grid obs) against `weak_basic_v2`:

- wins the fight decisively (10–20 kills vs 0, hit/fire 0.04–0.07)
- holds majority presence (31–38 s of a 60 s round), on-point 0.5
- accumulates a full capture's worth of progress (`cap_gain = 238` of 240)
- and then loses every tick of it (`cap_loss = 238`), never holding an
  uncontested window longer than **0.5 s** against the 8 s capture requirement.

Every PPO attempt from this checkpoint (1e-5, 3e-6) erases the behavior within
25–50 updates instead of adding the hold. The 3e-6 run even passed through a
visibly better state at update 30 (kills 20/0, majority 38.6 s, cap gain 97)
and then regressed by update 50.

Two things are jointly responsible, and neither has been attacked directly.

## Root cause 1 — the respawn treadmill (never touched in ~40 experiments)

`mechanics.respawn_ticks: 240` (8 s) appears in every config ever run and has
never been varied or curriculum-ed. Combined with the objective state machine:

- Capture requires 240 ticks of *zero living enemies inside the circle*.
- Enemies respawn every 240 ticks and walk straight back to the circle.
- Killing three enemies one at a time (which is what a 10–20 kill/round policy
  does) produces staggered arrivals every ~3–5 s. Each arrival contests and
  freezes progress; each moment our team steps off, progress decays.

So with respawn = capture = 8 s, the only winning shapes are "wipe all three
nearly simultaneously while standing on point" or "zone the approach lanes
from on-point" — sophisticated coordinated behaviors, with zero reward
gradient pointing at them until the first score tick ~80 decisions later.

Every curriculum lever tried so far (damage, fire cooldown, round length,
capture time, unlock time, opponent aim noise) is a *proxy* for opening the
uncontested window. `respawn_ticks` is the direct lever, and it converts the
policy's already-solid kill skill straight into uncontested time:

- At `respawn_ticks: 2400` (no respawn within a 60 s round), the existing
  checkpoint's 10+ kills mean the enemy team is permanently wiped early. The
  point becomes uncontested by construction; capture and score follow from
  behaviors the policy already has (walk to point, stand, shoot). PPO gets a
  live terminal win/score gradient from update 1.
- Annealing respawn back down (2400 → 240) then *gradually* re-introduces the
  defend/zone problem on top of an established scoring behavior — exactly the
  Phase-3 recipe ("survival pressure teaches combat because there's an
  existing behavior worth defending"), applied to conversion.

This is the same class of knob as the already-implemented objective-timing
curriculum (`objective_unlock_seconds` / `objective_capture_seconds` anneal in
`mappo_runtime_context.py`); it should be classified with "curriculum," not
with the frozen sim rules. Config validation rejects 0 and missing but has no
upper bound issue for large values (verify once with a smoke).

**Cheapest possible test (no training, ~30 min):** matrix-eval the existing
bridge checkpoint at `respawn_ticks: 2400` and `720` vs `weak_basic_v2`. If
`mean_score_a > 0` at 2400, the whole hypothesis is confirmed before writing
any training code.

## Root cause 2 — PPO has no anchor, and the critic starts cold

The LR dichotomy the journal keeps rediscovering (≤1e-6 freezes, ≥2e-6
collapses, nothing in between works) is the textbook signature of running PPO
from a strong behavioral prior with (a) no trust region beyond the clip and
(b) a value function that doesn't match the new reward scheme. Both June-10
conversion runs changed the reward terms (`cap_progress_potential_coef`,
`capture_completed_bonus`) *and* warm-started — so the critic's values were
wrong on day one, early advantages were noise, and the policy got wrecked
before the critic caught up. Neither standard fix exists in the trainer:

1. **Critic warmup**: `ppo.critic_warmup_updates: ~25` — freeze the actor,
   train only the value head after warm start. Costs nothing, removes the
   most destructive phase.
2. **KL-to-anchor loss** (AlphaStar-style): keep a frozen copy of the
   warm-start policy and add `anchor_kl_coef * KL(π_θ ‖ π_anchor)` on rollout
   states, annealed to 0 over ~200–300 updates. This is what makes LR 1e-5+
   safe and dissolves the freeze/collapse dichotomy. Note the existing
   `cap_duel_distill_anchor` failed because it anchored on a *mini-game*
   teacher over *mini-game* states; anchoring on the bridge policy over the
   *full-env rollout* states is the correct version and can likely reuse that
   plumbing.
3. **Best-eval checkpointing + funnel gates**: the 2026-05-14 entry already
   observed `ckpt_final` was last-update, not best-eval. With critic warmup,
   the first ~50 updates are *expected* not to improve eval — stop killing
   runs at update 50 on score.

## Root cause 3 — the hold still pays less than the chase, locally

In `conversion_v1`: `kill_bonus: 0.5` vs PBRS at `coef 1.0` ⇒ one tick of
capture progress ≈ 1/240 ≈ 0.004. Leaving the point for ~2 s to secure a kill
trades ~0.08 of foregone progress for +0.5 — chasing still wins ~6×. The
policy already knows how to kill and kills are already *instrumentally*
rewarded under conversion shaping (dead enemies stop contesting). During the
conversion stage, drop `kill_bonus` to 0–0.1. The KL anchor protects the fire
behavior from being unlearned, which was the original reason for keeping the
kill bonus high.

## Recommended plan

**Step 0 (no training):** respawn-ablation matrix eval of
`ckpt_multi_enemy_closed_loop_supervised_bridge.pt` at respawn 2400 / 720 /
240 vs `weak_basic_v2`. Judge on `mean_score_a`, captures, uncontested
seconds. This is the falsification gate for everything below.

**Step 1 (small, additive, config-gated code):**
- Add `respawn_ticks` start/end to the objective-timing curriculum annealer
  in `mappo_runtime_context.py` (same shape as capture/unlock anneal).
- Add `ppo.critic_warmup_updates` (actor frozen, value-only updates).
- Add `ppo.anchor_kl_coef` + `ppo.anchor_kl_anneal_updates` with a frozen
  copy of the warm-start checkpoint.

**Step 2 (the run):** one combined run, not another single-lever probe:
- warm start: closed-loop bridge checkpoint; `actor_obs: multi_enemy_entity_grid`
- opponent: `weak_basic_v2`
- reward: conversion_v1 terms, but `kill_bonus: 0.1`
- curriculum: respawn 2400 → 240 over ~200 updates; capture 2 s → 8 s as in
  conversion_v1; unlock 5 s → 15 s
- ppo: LR 1e-5, critic_warmup 25, anchor KL annealed over 250; 500 updates
- gates on the conversion funnel, in order (each stage gated on the previous,
  not on score): kill edge → team-wipe events → uncontested seconds →
  captures → score. Expect nothing from eval before update ~50.

**Step 3 (after first stable score vs weak_basic_v2):** anneal
`capture_completed_bonus`/`uncontested_on_point_coef` (keep the PBRS term —
it's policy-invariant), then re-introduce anchor-mixed self-play (plumbing
exists) and climb the opponent ladder toward `basic`.

## Process notes

- **The pre-PPO score gate is blocking the only mechanism that can produce
  score.** Requiring `mean_score_a ≥ 1.0` from the *clone* before PPO may
  never be satisfiable — compounding imitation error is exactly what PPO is
  for. Gate PPO launch on conversion precursors (hit/fire, on-point,
  majority, cap gain — all of which the bridge passes), and gate the *PPO
  run* on the funnel.
- **Single-lever probes judged at update 50 have hit diminishing returns.**
  ~40 falsifications established which levers are dead; the remaining
  hypothesis space is combinations. Run the combined recipe and ablate
  backwards only if it works.
- **Scope falsifications to their warm-start** (already noted on 06-09; worth
  re-stating because the easy-timing falsification nearly buried the
  conversion path for three weeks).
- **Classify `respawn_ticks` annealing as curriculum** in the
  GOAL_INSTRUCTIONS non-negotiables, alongside objective timing.

## Implementation status (2026-07-09, same day)

Steps 0 and 1 are implemented and verified on this branch; see the journal
entry of the same date for details. Headlines:

- **Step 0 CONFIRMED the respawn hypothesis** at canonical 15s/8s timing
  (50 episodes/cell, `docs/reports/2026-07-09-respawn-ablation.json`):
  respawn 240t → 0W/50D score 0.00/0.00; 720t → 0W/50L score 2.20/7.03
  (first canonical-timing score from this checkpoint); 2400t →
  **50/50 wins, score 2.70/0.00**.
- Found and fixed a bug that changes the interpretation of the June 10 runs:
  the multi-enemy wrapper silently dropped ALL runtime setters (timing
  anneal, team_spirit, eval alpha/timing overrides) — `conversion_v1` never
  actually annealed timing.
- New trainer features: `env.respawn_curriculum` anneal,
  `ppo.critic_warmup_updates`, `ppo.anchor_kl_coef` (+ anneal), all additive
  and off by default. The combined run is
  `experiments/configs/phase4/probe/phase4_mappo_conversion_v2_respawn.yaml`.
