# Phase 4 Breakthrough Analysis: Why You're Stuck and the Way Through

Date: 2026-06-09
Sources: `docs/journal/reinforcement_learning_journal.md` (all Phase 4 entries),
`docs/reports/2026-05-18-phase4-strategic-proposal.md`, the 2026-05-22 audit chain
(`docs/plans/active/2026-05-22-phase4-*.md`), `src/sim/src/internal/sim_objective.cpp`,
`python/xushi2/reward.py`, `python/train/full_env_rehearsal.py`, and the current
`experiments/configs/phase4/` tree.

---

## TL;DR

**You are not stuck on combat, aim, composition, or architecture anymore. You are
stuck on an 8-second capture timer that no reward term has ever pointed at.**

The 2026-05-22 closed-loop supervised bridge checkpoint
(`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`)
already does everything except the last step. Against `weak_basic_v2` over 50 episodes:

| Metric | Bridge checkpoint | What's needed |
|---|---|---|
| Losses | **0/50** | ≤ some |
| Opponent score | **0.00** | — |
| Kills | **13.4 vs 0.0** | any edge |
| Hit/fire | 0.047 | ≥ 0.04 ✓ |
| Majority-on-point seconds | 24.3 | — |
| Cap-progress gain | 238 ticks (~1 full capture worth) | 240 *consecutive-ish* |
| Cap-progress **loss** | 197 ticks | ~0 |
| Uncontested-on-point seconds | **4.9** | **≥ 8.0** |
| Score | **0.00** | > 0 |

It totally neutralizes the opponent, accumulates nearly a full capture's worth of
progress, and then walks off the point and lets it decay. One behavior is missing:
**stay on the point after winning the fight.** Everything below is about why ~40
experiments couldn't produce that behavior and how to produce it now.

The two highest-leverage moves, both cheap:

1. **PPO from the bridge checkpoint with capture-progress potential shaping +
   the already-implemented `uncontested_on_point_coef` + the already-implemented
   objective-timing curriculum (capture 2s → 8s).** All prior PPO runs faced a
   ~80-decision zero-reward gap between "won the fight" and "first score tick."
   This closes the gap. The timing curriculum was falsified once — but with the old
   blind flat-obs spray policy that had *zero* uncontested time. The bridge
   checkpoint has 4.9s; under a 2s capture it scores immediately, and PPO finally
   gets a live score gradient to climb.
2. **Approve and run the objective-conversion bridge** the 2026-05-22 audit
   proposed (it's been waiting for your approval since then) — same closed-loop
   DAgger machinery, with labels/diagnostics for the hold-and-convert phase.

Do #1 first. It needs one small additive change to `reward.py` plus one config.

---

## 1. The scoring mechanics, and the math nobody's reward ever covered

From `src/sim/src/internal/sim_objective.cpp` (the 5-case state machine):

1. Objective is locked for the first **15s** (450 ticks).
2. To *capture*, a team must be on point **alone** (≥1 alive member inside, zero
   enemies inside) for **240 cumulative ticks (8s)**. Progress **freezes** while
   contested, **decays 1/tick** while the point is empty, and resets if the other
   team caps instead.
3. Only after ownership flips does standing on the point uncontested produce score
   ticks. Win = higher score at the 60s timeout.

Combine that with `respawn_ticks: 240` (8s) and `action_repeat: 3` at 30Hz
(10 decisions/s) and you get the real shape of the task:

- The first possible score tick requires **~80 consecutive decisions** of "stand
  here and don't chase anyone" *after* clearing the point — with **zero reward**
  along the way under every config you've ever run.
- A staggered kill pattern (one enemy every few seconds — which is what 13.4
  kills/60s looks like) means someone is *always* walking back onto the point.
  Each arrival contests, freezing capture. Killing one-at-a-time never opens an
  8s clean window unless the team wipes all three close together **while already
  standing on the point**, or zones the approach lanes from on-point.
- Worse: the *current reward gradient actively teaches the wrong thing.* Kills and
  damage pay immediately (`kill_bonus: 1.0`, `damage_dealt_coef: 0.01`); standing
  on an uncontested point pays nothing until 80 decisions later. A policy with a
  kill edge gets *more* reward by leaving the point to chase than by holding. The
  bridge checkpoint's `cap gain 238 / cap loss 197` is exactly that policy.

Now look at `python/xushi2/reward.py`: the available terms are terminal win/loss,
`score_per_second`, kill/death, damage, distance, `on_point`, `time_penalty`,
`majority_on_point`, `uncontested_on_point`. **There is no term tied to
`cap_progress_ticks` — the literal progress variable in the sim that gates all
scoring.** And `uncontested_on_point_coef` exists in code but appears in only two
configs (`phase4_mappo_current_selfplay_uncontested_anchor_*.yaml`, 2026-05-20)
that have no journal entry — it has effectively never been exercised in a real
Phase 4 vs-bot run.

This is the single structural hole that explains the whole plateau.

## 2. Why 40+ experiments failed: one diagnosis, four symptoms

Reading the journal end to end, every failure is a projection of the same cause —
**no learning signal exists inside the kill→hold→capture→score chain** — refracted
through whatever lever was being tested:

**Symptom 1: The draw basin (v5–v7 era, weak_basic, aux-aim, per-action entropy,
fire-mask, etc.).** All outcomes were draws → terminal reward identical → PPO had
literally no gradient toward winning, only noise. Damage/round-length/fire-rate
changes moved the *kill exchange* but kills don't connect to reward beyond
`kill_bonus`, and kills alone can't score. Every "the draw basin is invariant to X"
entry is the same observation: X was never on the path between the policy and the
first score tick.

**Symptom 2: The LR dichotomy (1e-6 freezes, ≥2e-6 collapses).** With zero usable
objective gradient, the only consistent gradients are "getting shot near the point
is bad" and "damage/kills are good." A small LR preserves the BC prior; a larger LR
follows the only live gradients — flee the point, or spray. This isn't a PPO tuning
problem; it's what PPO correctly does when the reward surface between "draw" and
"win" is flat with a cliff on one side. Once a dense conversion signal exists, the
usable LR window should widen substantially (you already saw this: the cap_duel and
aim-only mini-games trained fine at 5e-5–1e-4 because their rewards were dense).

**Symptom 3: Mini-game skills that never transfer (aim_only, combat_1v1, cap_duel
v1/v2, composition rehearsal, distill anchor).** Each mini-game gave dense reward
for its skill, so the skill trained. Full 3v3 then offered no reward for *deploying*
the skill in the conversion chain, so PPO (or full-env BC) promptly traded it for
whatever the full-env gradients did pay. Transfer didn't fail because composition
is impossible — it failed because the destination environment paid nothing for the
composed behavior.

**Symptom 4: The bridge clone that can't finish (the current frontier).** The
teacher in `python/train/full_env_rehearsal.py:multi_enemy_visible_targets` is
trivially simple — *walk to objective; once on point, set move=0 and never leave;
aim/fire at nearest visible enemy* — and it beats weak_basic_v2 10W/0L, score 9.2,
on-point 0.875. The clone matched the labels (movement MSE 0.015, aim err 0.20 rad,
fire acc 1.0) but in closed loop drifts off-distribution, wanders off point
(on-point 0.29 vs teacher's 0.875), and fragments the capture. Classic compounding
imitation error — and the proposed PPO stage that would normally fix this had no
conversion-aligned reward to fix it *with*, so the gate (correctly) blocked PPO.

**The meta-cause:** the autonomous-experiment guardrails (`GOAL_INSTRUCTIONS.md`
non-negotiables) froze "reward formulas" alongside sim rules from mid-May onward.
That was a sensible anti-reward-hacking rule, but it removed the one lever the
evidence kept pointing at. The Escape Protocol then systematically — and correctly —
falsified everything in the *allowed* space: opponents, damage, LR, entropy, BC
variants, five architecture probes, three teachers, self-play. The search was
disciplined; the search space just didn't contain the answer.

A second process issue compounded it: probes were judged on **score by update
50–100**. Score is ~80 decisions downstream of the behavior being learned, so every
probe that was building the right precursors (majority time, kill edge, hit/fire)
still read as "falsified." Gate on the leading indicators of conversion instead:
uncontested seconds, capture-completion events, cap-progress retention.

## 3. What was never tried (the gap map)

| Lever | Status | Note |
|---|---|---|
| Potential shaping on `cap_progress_ticks` | **Never tried; not implemented** | The textbook fix. Policy-invariant (PBRS), so it's not "reward hacking" — it cannot change the optimal policy, only the gradient density. |
| Capture-completion event bonus (ownership flip) | **Never tried** | One-line event in the same delta extractor that already tracks kills. |
| `uncontested_on_point_coef` in a vs-bot run | **Implemented, never run** | Two un-journaled 05-20 self-play configs only. |
| Objective-timing curriculum **from a checkpoint that has uncontested time** | **Never tried** | The 05-19 easy-timing run is recorded as falsifying this, but it warm-started the old flat-obs v6_5 spray policy (uncontested = 0.00s). With 2s capture, the bridge checkpoint (4.9s uncontested) scores on day one. The falsification doesn't apply to the new checkpoint. |
| PPO from the closed-loop bridge checkpoint | **Never tried** | The pre-PPO score gate blocked it — correct under the old reward, moot under a conversion-aligned one. |
| The audit's objective-conversion bridge | **Designed, awaiting your approval since 05-22** | |
| Wipe/zone diagnostics (team-wipe events, on-point-at-wipe, capture attempts) | Partially (cap-progress counters exist) | Cheap and would make the next runs legible. |

## 4. Recommended plan

### Step 0 — small additive code change (½ day)

Add to `RewardCalculator` (off by default, config-gated, mirrors how
`majority_on_point_coef` was added):

- `cap_progress_potential_coef`: per-step reward
  `coef * (Φ(s') − Φ(s))` where `Φ = team_cap_progress_ticks / capture_ticks`
  (signed: A's potential minus B's, zero-sum like the other terms). Holding an
  uncontested point pays every tick; letting progress decay *costs* every tick;
  contested freeze pays nothing. This is exactly the missing gradient, and because
  it's potential-based you can leave it on forever without distorting the optimum.
- `capture_completed_bonus`: one-time team bonus (e.g. +2.0) on ownership flip,
  detected from the owner field the same way kills are delta-detected. (Not
  potential-based, but small and aligned; anneal later if paranoid.)

Also surface in eval/W&B: capture completions, team-wipe count, on-point-at-wipe
fraction. The counters mostly exist (`cap-progress gain/loss`, majority/uncontested
seconds) — this is mostly plumbing into the eval summary.

### Step 1 — the breakthrough run (config-only after Step 0)

`phase4_mappo_conversion_v1.yaml`, roughly:

```yaml
env:
  actor_obs: multi_enemy_entity_grid
  opponent_bot: weak_basic_v2
  reward:
    score_per_second: 1.0
    kill_bonus: 0.5            # demote chasing relative to converting
    death_penalty: 0.5
    damage_dealt_coef: 0.005
    uncontested_on_point_coef: 0.15   # already implemented
    cap_progress_potential_coef: 1.0  # new (PBRS)
    capture_completed_bonus: 2.0      # new
    on_point_shaping_coef: 0.02
  # objective timing curriculum (already implemented, 05-19):
  # unlock 5s→15s, capture 2s→8s annealed over ~150 updates
ppo:
  learning_rate: 1.0e-5        # the dense signal should make this safe;
                               # fall back to 3e-6 if eval degrades by update 25
  entropy_coef_move: 0.01      # per-action entropy (already implemented):
  entropy_coef_aim: 0.03       # keep move stable, let aim/fire explore
  entropy_coef_binary: 0.02
run:
  init_from_checkpoint: runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt
  total_updates: 300
```

**Gates (leading indicators, not score-by-50):**
- update 50: capture completions > 0 under the eased timing, uncontested seconds
  trending up from 4.9, no loss collapse, hit/fire ≥ 0.03.
- update 150 (timing fully annealed to 15s/8s): mean_score_a > 0.
- update 300: phase-gate checks (`weak_basic_v2` score ≥ 3, wins ≥ 5).
- Abort triggers: on-point < 0.1 (flee collapse) or hit/fire < 0.015 for two
  consecutive evals.

Why this is different from all 40 predecessors, in one sentence each:
- *Starting point*: the first checkpoint in the project's history that wins the
  fight and only lacks the hold.
- *Reward*: the first run where every step of kill→hold→capture→score pays.
- *Curriculum*: the first time eased capture timing is paired with a policy that
  can actually exploit it, so `score_per_second` is live from update 1.
- *Gates*: judged on conversion precursors, so a working gradient isn't killed at
  update 50 for not having compounded yet.

Estimated probability of nonzero score vs weak_basic_v2: **~70%**; of clearing the
weak_basic_v2 phase-gate within ~2 runs of iteration: **~45–55%**. (Compare: every
prior single-lever probe was realistically <15% ex ante, and the journal bears
that out.)

### Step 2 — if Step 1 produces score but it's fragile

- Anneal `capture_completed_bonus` and `uncontested_on_point_coef` toward 0;
  keep the PBRS term (it's invariant — it can stay forever).
- Matrix-eval vs `noop` / `weak_basic_v2` / `basic` as usual; expect `basic` to
  still win — that's the *next* curriculum rung (it has perfect aim; consider
  `weak_basic` ±0.5 rad noise as the intermediate opponent), not a falsification.
- Then reintroduce anchor-mixed self-play for robustness, which you already have
  plumbing for, now that there's a transferable scoring behavior to protect.

### Step 3 — if Step 1 fails the update-50 gate (no captures even at 2s)

That would mean the clone's drift is too large for PPO to fix even with dense
reward. Then approve the audit's **objective-conversion bridge** (the 05-22
proposal): more closed-loop DAgger rounds with hold-phase states oversampled
(states where ≥1 enemy is dead or `cap_progress > 0` — i.e., exactly where the
clone currently leaves the point), gate on uncontested seconds ≥ 8 and ≥1 capture
per episode, *then* run Step 1's PPO from that checkpoint. The teacher provably
contains the behavior; this just spends more imitation budget on the phase where
the clone diverges.

A cheap parallel diagnostic worth 30 minutes: eval the *teacher itself* with the
conversion counters (wipe events, on-point-at-wipe, capture completions per
episode) to quantify what "good conversion" looks like, so the clone/PPO targets
are numbers, not vibes.

## 5. Process changes (so the next plateau costs days, not weeks)

1. **Amend the non-negotiables**: allow *additive, config-gated, potential-based*
   reward terms. PBRS is mathematically guaranteed not to change the optimal
   policy — it belongs with "curriculum" not with "reward hacking." Keep the ban on
   modifying existing terms, sim rules, and terminal structure.
2. **Gate probes on the bottleneck's leading indicators.** "Score by update 50"
   killed several runs that were building the right precursors. The conversion
   funnel is now measurable: kill edge → wipe/zone events → uncontested seconds →
   capture completions → score. Gate each stage on the stage before it.
3. **Journal the 2026-05-20 uncontested-anchor configs and zerosum probe** (logs
   in `benchmark_results/phase4_zerosum_probe_*.log`) — they're invisible to the
   Circling Detector right now, and one of them contains the only prior use of
   `uncontested_on_point_coef`.
4. **When a falsification depends on a checkpoint, record that scope.** The
   easy-timing curriculum was marked falsified, but only *for the v6_5 spray
   policy*. Scoping falsifications to their warm-start would have surfaced the
   Step 1 combination weeks ago.

## 6. One-paragraph summary

Phase 4's plateau was never about whether the agents could fight — they can
(13.4:0 kills), and a four-line scripted teacher proves the full task is solvable
from the actor's own observations. It's that the game requires 8 uncontested
seconds on the point before any score exists, and across ~40 experiments no reward
term, gate, or curriculum ever placed gradient inside that window — while kills
paid instantly for leaving it. You now hold a checkpoint that is one behavior away,
a reward term (`uncontested_on_point_coef`) and a timing curriculum that were built
but never combined with it, and a one-day PBRS addition that makes capture progress
itself rewarding. Run that combination.
