# Phase 4 Review: Why You're Still Stuck, and the Way Out

Date: 2026-07-07
Scope: full read of `docs/journal/reinforcement_learning_journal.md` (2026-05-08 →
2026-06-10), `docs/reports/2026-06-09-phase4-breakthrough-analysis.md`,
`src/sim/src/internal/sim_objective.cpp`, `python/xushi2/reward.py`,
`python/train/` (trainer, update, distill, rehearsal), and the
`experiments/configs/phase4/` conversion configs.

---

## TL;DR

**The June 9 conversion thesis was validated, not falsified.** The smoke run won
4/4 vs `weak_basic_v2` at update 2 — the first Team A wins against an
objective-contesting bot in the project's history. What failed on June 10 was not
the reward design; it was **PPO's optimization dynamics destroying a working
policy within 25–50 updates**, which is the single most-repeated event in the
journal (v4 BC→loss collapse, v7 warm-start collapses, combat_1v1 transfer
collapse, cap_duel transfer collapse, conversion_v1 at both 1e-5 and 3e-6).

That destruction has a specific, fixable mechanical cause that has never been
addressed, because the trainer literally has no lever for it:

> **Every warm-started PPO run computes GAE advantages against a critic that is
> stale or untrained for the current reward function.** The bridge checkpoint's
> critic was never trained at all on the conversion reward (the supervised bridge
> touches only the actor; its critic weights date to `v6_5`, a different reward
> scheme, different opponent, and `shaping_clip 3.0` vs the new `30.0`).
> Garbage value baseline → garbage advantages → the first few PPO updates rip
> apart the best policy you have ever had, at exactly the moment it is at its
> best. Lowering LR only slows the destruction (3e-6 bought you 30 updates of
> improvement before the same collapse).

`python/train/mappo_rollout_trainer.py:99` builds one Adam over
`model.parameters()` with a single LR; there is no critic-only warmup, no actor
freeze during PPO, no value-normalizer reset on reward change, and
`warmup_updates: 0` in the conversion configs. Fixing this is ~a day of work and
is the highest-leverage change available. Everything else below is ordered after
it.

---

## 1. Where you actually are (so the hole has a shape)

Frontier as of the last journal entry (2026-06-10):

| Capability | Status | Evidence |
|---|---|---|
| Locomotion / reach point | solved | onpt 0.5+ across many runs |
| Fire at visible enemies | solved | bin ≈ 1.0, visible-fire ≈ 1.0 |
| Win the fight vs weak_basic_v2 | solved | bridge ckpt: 0/50 losses, 13.4/0 kills |
| Hit conversion | passable | hit/fire 0.043–0.07 vs 0.04 gate |
| Majority presence | solved | 24–38 s majority windows |
| Accumulate capture progress | solved | cap gain 238 ticks (~1 full capture) |
| **Retain progress / finish capture** | **missing** | cap loss 197–238, uncontested 0.5–4.9 s |
| Score / win | blocked on the above | 0 |

And the win condition math (from `sim_objective.cpp`): capture needs 240
*consecutive-ish* uncontested ticks (progress freezes when contested, decays when
empty), respawn is also 240 ticks, and the bot walks straight back. Staggered
kills (13 kills / 60 s ≈ one every 4.6 s) mean someone is nearly always walking
back in, so conversion at canonical timing requires "wipe all three fast, from on
point, and stand still" — a genuinely narrow behavior, but one a 4-line scripted
teacher (`cpp_basic`, and `multi_enemy_visible`) provably executes from the
actor's own observations. The task is solvable; the missing piece is narrow.

You have been down ~a month since June 10. Nothing in that gap changed the
picture; the June 10 stopping point ("focus directly on conversion retention,
not more PPO scalar tuning") is still the correct frontier — but the next move
is not another reward/bridge variant, it's making PPO stop destroying what the
warm start hands it.

## 2. Recommendation 1 — stabilize PPO-from-checkpoint (do this first)

Add three small, config-gated trainer mechanics. None change the game, rewards,
obs/actions, or gates.

**(a) Critic-only warmup.** New `ppo.critic_warmup_updates: N` (start N=30). For
the first N updates: collect rollouts normally, but zero/skip the policy and
entropy losses (or put actor params in an LR-0 param group) and train only the
value head + value normalizer on the new reward. The policy stays exactly the
bridge policy while the critic learns what that policy is worth *under the
conversion reward*. Only then let the actor move.

**(b) Reset value-normalization statistics at warm start** whenever the reward
config differs from the checkpoint's (or unconditionally under a flag). Stale
running mean/var from a `shaping_clip 3.0` era is another silent advantage
distorter under `shaping_clip 30.0`.

**(c) Reference-policy anchor during early PPO.** You already built exactly the
right machinery in `python/train/cap_duel_distill.py` — it just anchored the
wrong teacher (a mini-game policy) on the wrong state distribution (cap_duel
batches). Reuse it with: teacher = the frozen warm-start checkpoint itself,
states = the current full-env rollout batch, loss = aim/fire/move imitation (or
a KL to the reference distributions), coefficient annealed to 0 over ~100
updates. This bounds how far early PPO can wander from the only known-good
policy while the dense conversion gradient takes over. (OpenAI Five and most
BC→RL pipelines use exactly this shape: KL-to-anchor, annealed.)

Also set `ppo.warmup_updates` (LR warmup already exists in the config surface —
it is set to 0 today) to ~20 for the actor phase.

**Instrument value explained-variance** (`1 - Var(returns - values)/Var(returns)`)
in W&B if not already logged. It makes this whole failure mode visible at a
glance: EV near/below 0 at warm start + policy collapsing = stale-critic
destruction, and you'll never have to guess again.

**Then rerun `phase4_mappo_conversion_v1.yaml` otherwise unchanged.** Gate at
update 50+N on: eval not worse than the BC/bridge baseline, `captures_a > 0`
under eased timing, uncontested seconds ≥ baseline 4.9. The June 10 rule "stop
if it collapses by 50" stays — but with warmup, collapse-by-50 would now be
genuine negative evidence instead of a foregone conclusion.

## 3. Recommendation 2 — fix the curriculum's two design flaws

**(a) The anneal is a treadmill.** `anneal_updates: 150` advances on the clock
regardless of whether the policy keeps up, and both prior timing-curriculum runs
regressed exactly when the anneal completed (2026-05-19 run; the June 9 smoke's
update-4 regression). Make it performance-gated: hold the current
unlock/capture tier until eval at that tier clears a threshold (e.g. win rate
≥ 40% or captures_a ≥ 1/episode over an eval), then advance one step; drop back
one tier on two consecutive regressed evals. The runtime scheduler plumbing
already exists (`objective_timing_curriculum`); this is a scheduling-policy
change only.

**(b) Symmetric easing helps the bot too — maybe more than it helps you.**
With capture eased to 2 s, any 2 s window where all of Team A is dead or
off-point hands `weak_basic_v2` ownership; the June 10 gate's strange
"losses 50/50, score B 0.03" is exactly that: B captures once, banks a 1-tick
lead, and A (who cannot finish a recapture) loses every episode by one tick.
Likewise easing *unlock* from 15 s → 5 s favors the scripted beeline bot, not
the learner. Two changes:

- **Ease capture only; keep unlock at canonical 15 s** (config-only, do it
  immediately).
- **Consider per-team capture ticks** (learner 2 s → 8 s annealed, bot fixed at
  8 s). This is a sim-rule change and so needs your explicit approval under the
  non-negotiables, but it is the clean version of the curriculum: it eases the
  task without simultaneously arming the opponent. Config-gated, canonical
  defaults unchanged, one field threaded through `MatchConfig`.

## 4. Recommendation 3 — if conversion still doesn't stick: scenario resets

The states that matter most — "fight just won, point clear, now hold" — are a
tiny fraction of rollout frames, so both PPO and DAgger starve on them (the
conversion bridge's hold-state oversampling weighted *samples*, but it can only
weight states the policy actually visits). Add config-gated **initial-state
injection** to the Phase 4 env: a fraction of episodes begin in engineered
states —

- enemies dead with staggered respawn timers, learner squad on/near the point,
  `cap_progress` partial (drill: finish the capture, then defend);
- point owned by B, learner alive nearby (drill: recapture);
- normal spawns (the rest, so nothing is forgotten).

This is reverse-curriculum / "backplay": start at the goal-adjacent state and
grow backward. It converts the missing behavior from a rare discovery into the
majority of experience. Medium-sized change (initial-state injection through the
bindings), no reward or rule changes, and it composes with Recommendations 1–2.
A cheap Python-side approximation if you want it this week: a scripted
"pre-roll" wrapper that plays the known-good teacher until the wipe happens,
then hands control to the learner.

## 5. Recommendation 4 — when a run finally has slope, let it run

The 50-update falsification discipline was the right tool for pruning a dead
search space; it is the wrong tool for consolidating a live gradient. Your own
history says so: v6_5's only breakthrough appeared at update 1325/1500;
conversion_v1_lr3e6's best eval (20/0 kills, 38.6 s majority — the best combat
eval of the entire project) was at update 30, and the run was stopped at 50 on a
single regressed eval. PPO from a clone characteristically dips before it
recovers; evals oscillate on a ~100–200-update period in this codebase.

Concretely: once the stabilized run shows `captures_a > 0` trending, commit to
**one long overnight run (1500–3000 updates)**, best-eval checkpointing on (it
already exists), judged on the conversion funnel (uncontested seconds → captures
→ score → wins), with abort only on *sustained* collapse (3–4 consecutive evals
below the warm-start baseline). Twenty more 50-update probes cannot find what
only consolidation time produces. If CPU allows, also raise `num_envs` (64×128
is a small advantage batch for a 6-agent stochastic game; eval noise is partly
just this).

## 6. Smaller items worth 30 minutes each

- **PBRS gamma:** `reward.py:_objective_conversion_term` pays
  `coef · (Φ(s′) − Φ(s))`; strict PBRS is `coef · (γΦ(s′) − Φ(s))`. At
  γ=0.997 the residual mildly *rewards holding high potential*, which here is
  benign-to-helpful, but it is not exactly policy-invariant — worth a comment,
  and verify Φ is settled to a terminal convention at episode end/truncation so
  nothing leaks across resets.
- **Journal the gap.** Nothing has been journaled since 2026-06-10; the 05-20
  uncontested-anchor configs remain unjournaled (flagged on 06-09 too). The
  Circling Detector is blind to anything unjournaled.
- **Widen the non-negotiables deliberately.** Both escapes so far (multi-enemy
  obs on 05-22, conversion rewards on 06-09) came from widening a frozen search
  space after weeks of disciplined falsification inside it. Explicitly whitelist
  as always-allowed: optimizer/trainer mechanics (warmups, freezes, anchors),
  curriculum *scheduling*, and initial-state distributions — none of these
  change the game, the reward optimum, or the gates.

## 7. The sequence, as a checklist

1. **Trainer stabilization** (critic warmup + value-norm reset + reference
   anchor + LR warmup + EV metric) → rerun `conversion_v1`. ~1 day. This is the
   unblocking move.
2. **Curriculum fixes** (unlock stays 15 s; performance-gated anneal; ask
   yourself whether to approve per-team capture ticks). Config + small trainer
   change.
3. **If holds still fragment:** scenario resets (post-wipe hold, recapture).
4. **First nonzero canonical score:** one long consolidation run, then the
   existing ladder — `weak_basic` (±0.5 rad) → `basic`, then anchor-mixed
   self-play for robustness. All machinery already built.

The June 9 report estimated ~70% odds of nonzero score for the conversion combo;
what June 10 actually tested was that combo *plus an unstabilized fine-tune*,
and the instability — not the thesis — is what failed. The smoke run remains the
strongest positive result this project has produced. You are one trainer fix and
one honest long run away from finding out if the thesis holds.
