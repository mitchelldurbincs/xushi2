# Reinforcement Learning Journal

Short, dated lessons learned while training xushi2 policies. Reference for future runs and design decisions.

---

## 2026-05-08 — Phase 4 (3v3 MAPPO)

**Per-agent rewards are mathematically cleaner but produce ~3× weaker gradient than the old broadcast path.** The previous `np.full(3, team_scalar)` triple-counted shaping events at the gradient level. After moving to true per-agent attribution, hyperparameters and shaping coefs that were "tuned" on the broadcast scale need either compensation or fresh tuning.

**team_spirit interpolation is sound but doesn't fix bad reward magnitudes.** Verified ramp 0.3→1.0 (OAI Five) flows through correctly; sum-invariant under the team mean. But it can't rescue a policy whose reward signal is too weak — that's a separate axis.

**BC pretrain bootstraps a cap-pointed policy in seconds, but PPO with `entropy_coef=0.01` unlearns it within ~25 updates against any firing opponent.** Lower `entropy_coef` (~0.005) preserves BC structure through PPO. The BC-eval test passes only because it does 1 update; long runs erase it.

**Damage-dealt shaping cannot bootstrap aim from random.** Rewarding HP applied requires hits, which require aim, which we don't have. The signal stays at zero. Damage shaping is useful for *reinforcing* hits once they exist, not for *teaching* aim from a randomly-aiming starting point.

**The decisive lesson: against a firing opponent from random init, "learn cap-holding AND combat simultaneously" is a chicken-and-egg.** Phase 3's working recipe didn't teach shooting on its own — it used a two-stage curriculum: opponent_bot=`walk_to_objective` with `kill_bonus: 0` first (cap-holding only, ~1500 updates), then warm-start into a `basic` opponent run (~3000 updates). Survival pressure under combat then teaches firing/aim, because there's an existing cap-holding behavior worth defending.

**Tooling note: `dump_replay` uses `greedy_action` by default, which can hide what the policy "intends."** Greedy makes a 28%-fire-probability policy look like 0%-fire. Added `--stochastic` flag (`scripts/dump_replay.py`) to sample from the policy distribution instead — closer to training-time behavior, useful for viewer-based diagnosis.

**Phase 3's curriculum recipe is the foundation for Phase 4 too.** Mirroring `phase3_ranger_objective_curriculum_warmstart_v3.yaml` (opponent=`walk_to_objective`, `kill_bonus: 0.0`, `score_per_second: 0.05`, `time_penalty_per_second: 0.05`, `entropy_coef: 0.005`, 1500 updates) produced our first non-loss Phase 4 result: 50/50 draws, agents reach the cap (`onpt ≈ 0.30`, `dist ≈ 0.20`) and contest 3v3, occasionally kill enemy bots, never die. `time_penalty_per_second` is the lever that breaks the deny-stalemate equilibrium — without it, "stay home" and "contest cap" both yield ~0 reward and PPO drifts.

**Deny-stalemate equilibrium plateau is normal at 1v1→3v3.** Phase 3 hit this and didn't break through inside the curriculum stage; they moved on to the adversarial-push stage (warm-start + firing opponent). 3v3 is harder because majority requires *killing* one of three enemies before scoring becomes possible. The push stage is where combat actually gets learned — survival pressure under fire teaches aim/fire timing because there's now a hard-won cap-holding behavior worth defending.

**Phase 4's curriculum CAN find wins, but PPO will oscillate around them.** v6.5 reached 50/50 wins at update 1325 of 1500 (`mean_reward = +9.388`, `score 1.00/0.00`), then drifted back to draws for the remaining 175 updates. The trainer's best-eval-checkpoint logic preserves the winning policy as `ckpt_final.pt`, so warm-starts pick up the breakthrough state, not the late-training drift. If you want a stable winner from a single run, you'd need either an early-stop on best-eval or a tighter LR floor.

**Phase 3's `walk_to_objective` → `basic` jump doesn't transfer cleanly to Phase 4 (3v3).** Direct warm-start of v6.5's winning cap-holder into a `basic` opponent run produced `0/50 wins, 0/30 kills` inside 50 updates — the policy collapsed under enemy fire because it had no combat skill. Insert an intermediate stage: `hold_and_shoot` (stationary shooter, doesn't contest cap). Lets the policy keep scoring against an uncontested cap while learning to dodge / return fire from a fixed-position threat. Then warm-start *that* into the `basic` run.

## 2026-05-14 — Phase 4 v6_5 reproduction run (full 1500 updates)

**Reproduced v6.5 on clean `origin/main` (`fe15b3e`) with automated kanban benchmarker.** Run completed 1500 updates in ~106 minutes. Config: `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v6_5.yaml`. Seed `0xD1CEDA7A`. W&B: `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/9s4era9p`.

**Result: 50/50 draws, 0 wins, 0 losses, mean_reward = -0.387.** This is a timeout-draw equilibrium — agents consistently reach and contest the cap (`onpt = 0.52`, `dist = 0.13`) but never score and never die. The run did NOT reproduce the earlier journal entry's 50/50 wins at update 1325; the prior breakthrough may have been seed- or config-dependent.

**Eval oscillation pattern: intermittent "loss=50/50" blips (~-10.7 reward) interleaved with stable "draw=50/50" epochs (~-0.5 reward).** Every ~100-200 updates the eval would flip to all-losses for a single eval round, then revert to all-draws. This suggests map-seed or opponent-randomization variance in the eval environment is large enough to produce adversarial spawns occasionally.

**The policy converged to "contest cap, survive, do not score."** No kills in eval (4 kills total across all episodes, likely from enemy self-damage or edge cases). Score stayed 0/0 throughout. `onpt` climbed steadily to 0.52, `dist` fell to 0.13 — locomotion to the cap is solid. What's missing is the kill-and-score transition.

**No replays were generated.** The benchmarker run used `xushi2-train` directly, which does not auto-dump replays. Without replay artifacts, viewer-based behavior inspection is impossible. For future gate runs, invoke `scripts/dump_replay.py` post-training or add replay-dump to the training loop.

**Warm-start checkpoint is valid and ready.** `ckpt_final.pt` (186KB) saved at `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`. The next curriculum stage, `v7_holdshoot`, is already queued and warm-starting from this checkpoint. If v7 also fails to produce wins, the bottleneck is likely the transition from "reach cap" to "kill enemy then score" — which is exactly what the `hold_and_shoot` intermediate stage was designed to bridge.

**Key open question: does the earlier v6.5 win-breakthrough require a different seed, a different eval schedule, or was it a transient best-eval that `ckpt_final.pt` missed?** The prior journal entry says best-eval logic preserves the winner, but our run's `ckpt_final.pt` corresponds to the final update (-0.387), not a best-eval checkpoint. If the trainer's `best_eval` logic was working, there might be a `ckpt_best.pt` or similar — none was found in the run dir.

## 2026-05-14 — Phase 4 v7_holdshoot run (stopped early at ~update 825)

**Warm-started v6_5 cap-holder into `hold_and_shoot` opponent. Agents abandoned the cap and learned to run away.** User stopped the run after observing eval trends.

**Metrics trajectory (evals every 50 updates):**
- `onpt` dropped from 0.041 → 0.000 by update 500 — agents stopped touching the cap
- `dist` (distance to objective) rose from 0.30 → 0.66 by update 500, then fell back to 0.35 by update 800 — agents first retreated, then found a "safe corner"
- Opponent kills fell from 16/50 → 0/50 — agents got excellent at NOT dying (by avoiding combat entirely)
- Our kills stayed 0.0 throughout — never learned to fire back
- All 50/50 draws (timeouts), 0 wins, 0 losses, score 0/0
- `mean_reward` rose to +1.000 — the "run away" strategy is being positively reinforced

**Root cause: reward structure makes "avoid cap + avoid death + timeout" the optimal policy.**
- `hold_and_shoot` opponent never moves and never contests the cap. It only fires at nearest enemy from spawn.
- `on_point_shaping_coef: 0.0` — no direct reward for being on the cap
- `distance_shaping_coef: 0.01` rewards Team A when closer to cap than Team B. Since the opponent stays at spawn (far from cap), the learner can retreat to a mid-distance and still satisfy this differential, collecting positive shaping while avoiding deaths.
- `score_per_second: 0.10` only triggers when actually scoring, which requires capping. With no on-point pressure, agents never cap.
- `kill_bonus: 0.25` requires firing back, which requires being in line-of-fire, which gets you killed. Death penalty dominates.
- `time_penalty_per_second: 0.05` is too weak to break the timeout equilibrium (and may be undercharged by `action_repeat: 3`).

**Codex audit conclusion: the v7 config design was flawed.** The assumption was "hold_and_shoot lets agents score unopposed while learning to survive fire." In practice, PPO discovered that NOT scoring, NOT fighting, and NOT dying yields the highest stable reward. The cap-holding behavior from v6_5 was actively unlearned because it correlated with deaths.

**Recommended fixes (ranked):**
1. **Set `on_point_shaping_coef: 0.02`** (config-only, highest impact). This was the coef used in working Phase 4 probe configs. It directly rewards cap contact and prevents the "hover safely at mid-distance" basin.
2. **Increase `distance_shaping_coef` to 0.05** or replace differential shaping with absolute "distance to cap" penalty. Differential shaping is exploitable when the opponent never approaches the cap.
3. **Verify/fix `time_penalty_per_second` scaling under `action_repeat: 3`**. The penalty may be 3× weaker than intended because reward is computed per env step, not per sim tick.
4. **Use a harder opponent** — e.g. `basic` with reduced damage, or a `walk_and_shoot` that slowly approaches cap while firing. Pure stationary shooter creates no score-pressure for the learner to ever cap.
5. **Do NOT raise `kill_bonus` alone** — the learner has 0 kills. It cannot bootstrap combat from zero. Need on-point pressure first, then combat emerges as a side effect of defending the cap.
6. **Consider re-warming from a stronger cap-holding checkpoint** or adding a brief BC pretrain toward cap-hold before PPO resumes under fire.

**Next experiment:** create `phase4_mappo_basic_v7_holdshoot_v2.yaml` with `on_point_shaping_coef: 0.02`, `distance_shaping_coef: 0.05`, and keep `hold_and_shoot` opponent. Judge success by `onpt > 0.1` and `score > 0` within first 100 updates, not by mean_reward.

## 2026-05-14 — Phase 4 v7_basic_reduced (stopped at update ~225)

**Warm-started v6_5 into `basic` opponent with 2500 centi-HP damage. Agents were slaughtered 50/50 for 200+ updates with zero improvement.**

**Metrics (evals every 50 updates):**
- Update 50: 0/50 wins, 50/50 losses, score 0.00/7.00, kills 0.0/27.0, onpt 0.021, mean_reward -11.000
- Update 100: same — 0/50 wins, 50/50 losses, score 0.00/7.00, kills 0.0/27.0, onpt 0.002, mean_reward -11.000
- Update 150: same — flatlined at total annihilation
- Update 200: same — 0/50 wins, 50/50 losses, score 0.00/7.00, kills 0.0/27.0, onpt 0.049, mean_reward -11.000

**Slight behavioral drift but no functional improvement:** `onpt` slowly crept up from 0.002 → 0.049 and `dist` dropped from 0.540 → 0.394 across 200 updates. Agents were slowly re-learning to approach the cap, but dying before achieving anything. The basic bot consistently scored ~7 points and got ~27 kills per 50-episode eval. Our agents: 0 kills, 0 score.

**Warm-start from v6_5 is actively harmful against a firing opponent that contests the cap.** v6_5 agents know "walk to cap + stand still." Against `basic`, that means marching directly into a bot that walks to cap AND shoots. Cap = instant death. PPO unlearns the warm-start faster than combat can emerge.

**Reduced damage (2500 centi-HP, ~4 hits to die) was not enough.** The problem isn't damage magnitude — it's that the policy has zero combat skill and zero reason to fire back. Even surviving 4 hits on cap is irrelevant if the policy doesn't know how to shoot.

**Stopped at update ~225. No sign of breakthrough.** Mean_reward flat at -11.000 for 4 consecutive evals. Continuing would waste ~90 minutes with no evidence of improvement.

**Next approach: BC pretrain on top of warm-start.** The existing `bc_pretrain_walk_to_objective` re-trains cap-walking behavior in seconds. Running it AFTER warm-start but BEFORE PPO should give a stronger behavioral prior than warm-start weights alone. entropy_coef=0.005 should preserve the BC structure through early PPO updates. If BC + PPO still fails within 200 updates, the jump from "no-fire" to "cap-fighting" may simply be too large for this architecture/curriculum — need a different strategy entirely (e.g., even lower damage, different opponent, or architectural changes).

## 2026-05-14 — Phase 4 v7_basic_reduced_bc (stopped at update 200)

**BC pretrain converged (loss 0.23 → 0.0004) but agents still slaughtered 50/50 by basic bot at 2500 damage.**

**BC phase:** 500 steps, loss dropped from 0.2325 to 0.0004. BC successfully re-anchored cap-walking behavior.

**BC eval (before PPO):** 0/50 wins, 50/50 losses, score 0.00/7.00, mean_reward -11.000. Even a pure walk-to-cap policy gets massacred.

**PPO evals (updates 50-200):**
- Update 50: 0/50 wins, 50/50 losses, score 0/7.00, kills 0.0/27.0, onpt 0.091, dist 0.357, mean_reward -11.000, bin=0.000
- Update 100: same — 0/50 wins, 50/50 losses, score 0/7.00, kills 0.0/27.0, onpt 0.099, dist 0.451, mean_reward -11.000, bin=0.000
- Update 150: same — flatlined, onpt 0.048, dist 0.409, mean_reward -11.000, bin=0.000
- Update 200: same — 0/50 wins, 50/50 losses, score 0/7.00, kills 0.0/27.0, onpt 0.072, dist 0.385, mean_reward -11.000, bin=0.001

**Agents walk to cap but never fire.** `move=0.82-0.94`, `dist~0.38`, `onpt~0.10` confirms BC re-anchored locomotion. But `bin=0.000-0.001` — the fire action is completely unused. Agents reach cap, basic bot is already there, they die in ~4 hits. No survival window to discover firing.

**2500 damage is still too lethal for a policy with zero combat skill.** The problem isn't warm-start or reward shaping — it's that the gap between "walk to cap" and "cap + shoot back" is too large. Agents need to survive 8-10+ hits to learn combat through trial and error.

**Stopped at update 200 per gate criteria.** No sign of breakthrough across 4 consecutive evals. Mean_reward flat at -11.000.

**Next approach: much lower damage (500 centi-HP = 5 HP per shot, ~20 hits to die).** This should give agents enough survival time on cap to discover the fire action exists, learn that shooting back reduces deaths, and eventually hold cap + shoot = score. If 500 damage also fails, the problem is not damage magnitude — need architectural changes or a completely different curriculum strategy.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v2 (500 damage, stopped at update 200)

**500 damage converted slaughter into stalemate — neither team can kill the other.**

**BC phase:** 500 steps, loss 0.1697 → 0.0005. BC re-anchored cap-walking.

**BC eval:** 0/50 wins, 0/50 losses, 50/50 **draws**, score 0.00/0.00, mean_reward -1.000. Even with walk-to-cap only, the basic bot at 500 damage couldn't kill our agents reliably.

**PPO evals (updates 50-200):**
- Update 50: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, kills 0.0/6.0, onpt 0.580, dist 0.157, mean_reward -1.000, **bin=0.000**
- Update 100: same — 50/50 draws, score 0/0, kills 0.0/6.0, onpt 0.674, dist 0.140, mean_reward -1.000, **bin=0.000**
- Update 150: same — 50/50 draws, score 0/0, kills 0.0/0.0, onpt 0.604, dist 0.163, mean_reward -1.000, **bin=0.000**
- Update 200: same — 50/50 draws, score 0/0, kills 0.0/3.0, onpt 0.622, dist 0.160, mean_reward -1.000, **bin=0.001**

**Agents hold cap beautifully but still never fire.** `onpt=0.58-0.67`, `dist=0.14-0.18` — locomotion is excellent. But `bin=0.000` throughout. The policy never discovers the fire action exists.

**500 damage removed ALL combat pressure.** The basic bot walks to cap and shoots, but at 5 HP per shot it can't reliably kill in a 30-second round. Our agents sit on cap, tank damage, and timeout. Bot kills dropped from 27/50 (at 2500 damage) to 0-6/50. Neither team scores. No wins, no losses, just draws.

**The reward signal is just time_penalty + minor shaping.** With 50/50 draws, terminal reward = 0. Score = 0. Kills ≈ 0. PPO optimizes inside a low-information draw basin with no gradient pushing toward combat.

**Key insight: the problem is not damage magnitude — it's that agents NEVER discover firing.** At 2500 damage, they die before discovering it. At 500 damage, there's no reason to discover it. We need to explicitly teach firing, not just hope PPO stumbles upon it.

**Next approach: BC pretrain that explicitly includes firing.** Modify the BC target generator to set `primary_fire=1` when enemies are visible, and aim toward them. This initializes the binary action head away from zero before PPO even starts. Combined with 500 damage (survivable), PPO can then reinforce actual kills once they happen.

**Also added:** `docs/reports/v7_500dmg_stalemate_analysis.md` — full Codex analysis of why neither extreme (2500 slaughter vs 500 stalemate) works, and ranked recommendations including the BC-with-firing approach.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v3 (BC with firing + 500 damage, stopped at update 425)

**BC-with-firing pretrain WORKED: binary head initialized away from zero (bin~0.33). But 500 damage was too low — pillow-fight stalemate.**

**BC phase:** 500 steps, loss 0.4239 → 0.0005. `walk_and_shoot` variant successfully taught firing.

**BC eval:** 0/50 wins, 0/50 losses, **50/50 draws**, score 0.00/0.00, mean_reward -0.626. Agents held cap and fired, but nobody could kill anyone.

**PPO evals (updates 50-400):**
- Update 50: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, kills 0.0/3.0, onpt 0.572, dist 0.165, mean_reward -1.000, **bin=0.333**
- Update 100: same — 50/50 draws, score 0/0, kills 0.0/6.0, onpt 0.443, mean_reward -1.000, **bin=0.333**
- Update 150: same — 50/50 draws, score 0/0, kills 3.0/2.0, onpt 0.797, mean_reward -1.000, **bin=0.330**
- Update 200: same — 50/50 draws, score 0/0, kills 3.0/4.0, onpt 0.628, mean_reward -0.871, **bin=0.329**
- Update 250: same — 50/50 draws, score 0/0, kills 2.0/3.0, onpt 0.556, mean_reward -0.709, **bin=0.333**
- Update 300: same — 50/50 draws, score 0/0, kills 0.0/6.0, onpt 0.731, mean_reward -1.000, **bin=0.325**
- Update 350: same — 50/50 draws, score 0/0, kills 0.0/6.0, onpt 0.670, mean_reward -1.000, **bin=0.329**
- Update 400: same — 50/50 draws, score 0/0, kills 0.0/3.0, onpt 0.680, mean_reward -1.000, **bin=0.323**

**Binary head is ALIVE but combat produces no wins.** `bin=0.323-0.333` throughout all 425 updates — the BC firing initialization was preserved by PPO. But 500 damage means everyone fires ineffectually for 30 seconds and times out. Kills stayed at 0-3 per 50 episodes with no upward trend.

**500 damage removed all lethality.** Both teams walk to cap, shoot at each other, tank 20+ hits, and timeout. No score, no wins, no losses. PPO receives almost no terminal gradient (all draws, score=0, kills≈0).

**Key insight: the problem was never "agents don't know how to fire" — it was "damage is at an extreme that prevents combat from producing decisive outcomes."** 2500 = too lethal (instant death, no learning). 500 = too soft (pillow fight, no pressure). Need Goldilocks damage where kills happen consistently but agents survive long enough to learn.

**Next approach: 1000 centi-HP damage (~10 hits to die, 10 HP per shot).** Middle ground between slaughter and stalemate. BC-with-firing is already working — just need lethality that makes combat matter.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v4 (BC with firing + 1000 damage, stopped at update 200)

**SMOKING GUN: BC policy alone produces draws at 1000 damage. But PPO breaks it into losses within 50 updates.**

**BC phase:** 500 steps, loss 0.4095 → 0.0008. `walk_and_shoot` variant successfully taught firing (binary_loss 0.71 → 0.0008).

**BC eval (before PPO):** 0/50 wins, 0/50 losses, **50/50 draws**, score 0.00/0.00, mean_reward -0.686. The BC policy ALONE is viable — agents walk to cap, fire at enemies, and survive. Neither team scores. This is a **working starting point**.

**PPO evals (updates 50-200):**
- Update 50: 0/50 wins, **50/50 losses**, score 0.00/6.03, kills 0.0/15.0, mean_reward **-11.000**, bin=0.333
- Update 100: same — 50/50 losses, score 0.00/2.30, kills 0.0/12.0, mean_reward -11.000, bin=0.334
- Update 150: same — 50/50 losses, score 0.00/3.30, kills 0.0/12.0, mean_reward -11.000, bin=0.333
- Update 200: same — 50/50 losses, score 0.00/5.93, kills 0.0/15.0, mean_reward -11.000, bin=0.333

**PPO actively unlearned a draw-producing BC policy into a losing one.** `bin=0.333` maintained throughout — firing wasn't lost. But eval shows `term=50` (all episodes end by enemy scoring). Training metrics show `onpt` climbing (0.123 → 0.361 by update 150) — agents go to cap during training. But the eval opponent consistently scores 2-6 points while we score 0.

**Root cause: PPO LR (5e-5) is too aggressive for this adversarial jump.** The BC policy sits near a local optimum (draws). Large policy steps break the delicate aim+fire balance and fall into a losing basin. Low entropy (0.005) prevents recovery once collapsed.

**1000 damage IS the Goldilocks zone for the BC policy, but PPO cannot improve from it.** At 500 damage, PPO had no gradient (all draws, no score pressure). At 1000 damage, PPO has negative gradient (we're losing) but takes steps so large it makes things worse instead of finding the winning strategy.

**Key insight: the problem is NOT damage, NOT firing initialization, NOT warm-start. It's PPO's step size in an adversarial environment where the BC policy is already near-optimal.** Need conservative PPO that barely moves the BC policy, preserving draws while allowing tiny refinements toward wins.

**Next approach: dramatically lower LR (1e-6) + higher entropy (0.02).** Make PPO extremely conservative. Preserve the BC draw-producing behavior. If even a single win appears in 50 evals, the minuscule positive gradient at LR 1e-6 will slowly amplify it. Entropy 0.02 prevents collapse into a deterministic losing strategy.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v5 (conservative PPO: LR 1e-6 + entropy 0.02 + 1000 damage, ran 450 updates)

**Conservative PPO SUCCESSFULLY PRESERVED the BC draw policy — ZERO losses across 450 updates.** But also ZERO wins, ZERO score. The draw equilibrium is real and stable.

**BC phase:** 500 steps, loss 0.4095 → 0.0008. `walk_and_shoot` variant. BC eval: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, mean_reward -0.686.

**PPO evals (updates 50-450):**
- Update 50: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, kills 4.0/7.0, mean_reward -0.898, **bin=0.333**
- Update 100: same — 50/50 draws, score 0/0, kills 2.0/8.0, mean_reward -1.000, bin=0.334
- Update 150: same — 50/50 draws, score 0/0, kills 4.0/7.0, mean_reward -0.906, bin=0.331
- Update 200: same — 50/50 draws, score 0/0, kills 2.0/8.0, mean_reward -1.000, bin=0.333
- Update 250: same — 50/50 draws, score 0/0, kills 4.0/6.0, mean_reward -0.614, bin=0.333
- Update 300: same — 50/50 draws, score 0/0, kills 4.0/6.0, mean_reward -0.648, bin=0.333
- Update 350: same — 50/50 draws, score 0/0, kills 4.0/7.0, mean_reward -0.828, bin=0.333
- Update 400: same — 50/50 draws, score 0/0, kills 4.0/6.0, mean_reward -0.636, bin=0.333
- Update 450: same — 50/50 draws, score 0/0, kills 4.0/6.0, mean_reward -0.629, bin=0.333

**Perfect preservation of the draw basin.** `term=0`, `trunc=50` across all evals — all episodes timeout. Neither team scores. Kills trade at a steady 2-8 per 50 episodes but never convert to score within 30-second rounds. `bin=0.324-0.334`, `onpt=0.23-0.46`, `dist=0.15-0.26` — combat + cap behavior fully intact.

**The conservative PPO strategy worked exactly as designed.** It avoided the catastrophic collapse to losses that destroyed v4. But at LR 1e-6, the policy barely moves from the BC starting point. There is no gradient pushing it out of the draw basin because all outcomes are identical (draw, score 0, no terminal reward difference).

**30-second rounds are too short for 1000-damage combat to produce decisive outcomes.** Both teams walk to cap, trade shots, tank ~10 hits each, and timeout. No time for a kill advantage to translate into sustained cap-holding + score.

**Key insight: we have a VIABLE STARTING POINT (BC draws at 1000 damage). The problem is not PPO breaking it — it's that the game dynamics at 30s/1000dmg produce a genuine Nash equilibrium where neither side can win.** Need to change the game dynamics (longer rounds or slightly more lethal) to break the symmetry.

**Next approach: 60-second rounds + LR 2e-6.** Double the round time for sustained combat to produce decisive kill advantages. Slightly higher LR (still conservative, 2×) to escape the draw basin faster. Keep entropy 0.02 to prevent collapse.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v6 (60s rounds + LR 2e-6 + 1000 damage, stopped at update 250)

**60s rounds was CATASTROPHIC — the exact wrong direction. BC policy alone LOST at 60s rounds (score 0/7.53). PPO made it worse: bot score 7.50 → 25.33, our kills 4.0 → 0.0.**

**BC phase:** 500 steps, loss 0.3852 → 0.0006. `walk_and_shoot` variant. BC eval: 0/50 wins, 0/50 losses, **0/50 draws**, score 0.00/7.53, mean_reward **-11.000**. At 30s the same BC policy produced draws. At 60s the bot dominates.

**PPO evals (updates 50-250):**
- Update 50: 0/50 wins, **50/50 losses**, score 0.00/7.50, kills 4.0/12.0, mean_reward -11.000, bin=0.333
- Update 100: same — 50/50 losses, score 0.00/18.63, kills 2.0/15.0, mean_reward -11.000, bin=0.334
- Update 150: same — 50/50 losses, score 0.00/22.47, kills 2.0/21.0, mean_reward -11.000, bin=0.333
- Update 200: same — 50/50 losses, score 0.00/24.67, kills 0.0/29.0, mean_reward -11.000, bin=0.333
- Update 250: same — 50/50 losses, score 0.00/25.33, kills 0.0/30.0, mean_reward -11.000, bin=0.334

**Longer rounds favor the stronger combatant (the basic bot), not the learner.** The bot walks to cap AND fires more accurately. In a 30s skirmish, neither team can score decisively before time runs out. In 60s sustained combat, the bot's superior aim accumulates kills, takes the cap, and racks up 25+ points per 50-episode eval. Our agents' BC aim is too poor to compete in extended gunfights.

**PPO at LR 2e-6 actively worsened a losing BC policy.** Bot score climbed steadily while our kills dropped to zero. The policy was not preserving anything — it was sliding deeper into a losing basin.

**Key insight: the lever is NOT round length — it's making the bot WORSE at combat to create an exploitable asymmetry.** Our BC policy fires at bin=0.33. If the bot fires half as often (cooldown 30 instead of 15), we have a 2:1 DPS advantage. On equal aim quality, we should win the DPS race and produce kills that translate to score.

**The draw equilibrium at 30s/1000dmg/15cooldown exists because both teams trade ineffectually with equal fire rate.** Break the symmetry by nerfing the bot's fire rate, and our higher DPS should produce decisive outcomes.

**Next approach: 30s rounds + revolver_fire_cooldown_ticks: 30 (bot fires half as often) + LR 2e-6 + entropy 0.02.** Same 1000 damage. BC walk_and_shoot warm-start. Creates a natural DPS asymmetry: our BC policy fires twice as often as the bot, which should translate to more kills, cap-holding, and score.

## 2026-05-14 — Phase 4 v7_basic_reduced_bc_v7 (bot fire cooldown 30 + 30s + LR 2e-6, stopped at update ~325)

**The draw equilibrium is ROBUST against fire-rate changes. Even with 10:1 theoretical DPS advantage, bot aim > our aim.**

**BC phase:** 500 steps, loss 0.4386 → 0.0004. `walk_and_shoot` variant. BC eval: 0/50 wins, 0/50 losses, **50/50 draws**, score 0.00/0.00, mean_reward **-0.375**. Same draw as v5 (equal fire rate).

**PPO evals (updates 50-300):**
- Update 50: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, kills **0.0/12.0**, mean_reward -1.000, bin=0.333, onpt=0.485
- Update 100: same — 50/50 draws, score 0/0, kills **0.0/9.0**, mean_reward -1.000, bin=0.332
- Update 150: same — 50/50 draws, score 0/0, kills **0.0/12.0**, mean_reward -1.000, bin=0.333
- Update 200: same — 50/50 draws, score 0/0, kills **0.0/9.0**, mean_reward -1.000, bin=0.333
- Update 250: same — 50/50 draws, score 0/0, kills **2.0/7.0**, mean_reward -0.782, bin=0.333
- Update 300: same — 50/50 draws, score 0/0, kills **2.0/7.0**, mean_reward -0.865, bin=0.332

**Bot consistently gets 7-12 kills per 50 episodes; our agents get 0-2.** Despite firing at ~10× the bot's rate (bin=0.33 vs bot cooldown=60 ticks = ~0.017 shots/tick), our BC heuristic aim is too crude to hit moving targets. The bot's scripted aim compensates for lower fire rate with significantly higher accuracy.

**The bottleneck is AIM QUALITY, not fire rate, damage, or PPO step size.** No amount of fire-rate asymmetry at 1000 damage / 30s rounds can overcome the aim gap.

**All 10+ variants tested have converged on the same conclusion:**
- 2500 damage → slaughter (instant death, no learning window)
- 500 damage → pillow fight (nobody can kill anyone, no combat pressure)
- 1000 damage equal fire → draw (equal miss rate, timeout)
- 1000 damage bot at half fire rate → draw (bot aim compensates for lower DPS)
- 60s rounds → bot wins (longer rounds favor better aim)
- Conservative PPO → preserves draw but cannot improve
- Aggressive PPO → breaks draw into losses

**The ONLY untested path is explicit AIM TRAINING.** Our BC pretrain fires (bin=0.33) but aims heuristically at enemy's current position. In a real game, enemies move, and leading shots / tracking matters. The neural network has the capacity to learn this, but 3v3 objective complexity swamps the combat signal.

**This stage isolates combat learning:**
- `hold_and_shoot` opponent (stationary at spawn — easiest target possible)
- `revolver_damage_centi_hp: 250` (2.5 HP per shot, ~40 hits to die — massive survival window)
- `revolver_fire_cooldown_ticks: 60` (bot fires once per 2 seconds — minimal return fire)
- `death_penalty: 0` (no punishment for dying — encourage aggression)
- `kill_bonus: 1.0` (strong positive reward for kills)
- `damage_dealt_per_second: 0.2` (reward for damage — reinforces near-misses)
- `score_per_second: 0`, `time_penalty_per_second: 0` (no cap/timeout pressure)
- `distance_shaping_coef: 0`, `on_point_shaping_coef: 0` (no cap reward — purely combat)
- 60s rounds for extended combat practice
- LR 5e-5 (more aggressive — no delicate equilibrium to preserve)

**Success criteria:**
- Team A kills > bot kills consistently by update 500
- Upward-trending kill ratio (ours vs bot)
- `bin` maintained > 0.25
- Optional: replay inspection showing tracking/leading behavior

**Next stage after success:** warm-start the combat-capable checkpoint into 3v3 objective control with `basic` opponent and normal objective rewards. The policy should now have both cap-holding (from v6_5 warm-start) AND aim quality (from this combat stage).

## 2026-05-14 — Phase 4 v8_combat_pretrain_v1 (hold_and_shoot, 250 damage, pure combat, done)

**CATASTROPHIC FAILURE: hold_and_shoot opponent stays at spawn, so ZERO COMBAT HAPPENED.**

**BC phase:** 500 steps, loss 0.4386 → 0.0004. `walk_and_shoot` variant.

**BC eval:** **50/50 WINS**, score **37-0**, kills **0/0**, mean_reward **+37.000**.

**PPO evals (updates 50-500):**
- Update 50: 50/50 wins, score **37-0**, kills **0.0/0.0**, mean_reward +37.000, bin=0.333
- Update 100-400: same — 50/50 wins, score 37-0, kills 0.0/0.0, mean_reward +37.000, bin=0.333
- Update 450-500: degraded to **50/50 draws**, score **0-0**, kills 0.0/0.0, mean_reward -1.000, bin=0.333

**Root cause: hold_and_shoot bot stays at its spawn point. Our agents walk to the cap (opposite side of map). They never enter each other's line of fire.** The opponent never contests the cap, so our agents score 37 points uncontested. Zero shots fired in anger. Zero kills. Zero combat.

**PPO had no combat reward signal** (kill_bonus and damage_dealt never triggered). The only signal was cap-scoring from the BC behavior, so PPO reinforced capping. Eventually it drifted and lost even that.

**This proves: for combat pretraining to work, the opponent MUST be where our agents actually are.** `hold_and_shoot` at spawn + our agents at cap = no combat range.

**Next approach: use `basic` bot (walks to cap) with survivable DPS. Bot walks to the same location as our agents, guaranteeing close-range combat. 250 damage + 60-tick cooldown means bot cannot kill anyone even with 100% accuracy in 60s rounds. Our ~10× fire-rate advantage should produce hits through sheer volume at close range.

## 2026-05-15 — Phase 4 v8_combat_pretrain_v2 (basic bot at cap, 250 damage, survivable DPS, done)

**CATASTROPHIC FAILURE: basic bot + LR 5e-5 collapses into fleeing within 50 updates.**

**BC phase:** 500 steps, loss 0.4707 → 0.0002. `walk_and_shoot` variant.

**BC eval:** 0/50 wins, **50/50 draws**, score 0/0, mean_reward +1.000.

**PPO eval at update 50:** 0/50 wins, **50/50 LOSSES**, score 0/37, kills 0/0, mean_reward **-11.000**.

**Training progression (updates 5-50):**
- `onpt` collapsed: 0.165 → 0.001 → 0.000 (agents abandoned the cap)
- `dist` climbed: 0.340 → 0.523 → 0.519 (agents fled to map edges)
- `move` dropped: 0.690 → 0.506 (stopped moving toward objective)
- `bin` stable at 0.333 (still firing, but at empty space)

**Root cause: same as every variant with LR ≥ 2e-6 against basic bot.** The `basic` bot has perfect aim. At close range on the cap, it hits our agents. PPO at 5e-5 rapidly learns "cap = getting shot = losing = RUN AWAY." By update 50, all agents are at map edges, bot caps uncontested (score 37-0), 50/50 losses.

**The pure combat rewards (kill_bonus, damage_dealt) never triggered because agents never engaged.** They fled before any shots connected.

**This proves AGAIN: the cap + basic bot + LR ≥ 2e-6 = guaranteed collapse into fleeing/losses within 50-325 updates. There is no window between "frozen draw" (1e-6) and "collapse" (2e-6).**

## 2026-05-15 — Phase 4 v7_basic_reduced_bc_v5_high_entropy (30s, 1000dmg, basic, LR 1e-6, entropy 0.08, stopped at update ~340)

**RESULT: Draw basin is completely invariant to entropy. 4× higher entropy produced identical results to v5.**

**BC phase:** 500 steps, loss 0.4707 → 0.0002. `walk_and_shoot` variant.

**BC eval:** 0/50 wins, **50/50 draws**, score 0/0, mean_reward +1.000.

**PPO evals (updates 50-340):**
- Update 50: 0/50 wins, 0/50 losses, **50/50 draws**, score 0/0, kills **4.0/6.0**, mean_reward +0.959, onpt=0.528, bin=0.333
- Update 150: same — 0/50 wins, 50/50 draws, score 0/0, kills **4.0/6.0**, mean_reward +0.959
- Update 250: same — 0/50 wins, 50/50 draws, score 0/0, kills **4.0/6.0**, mean_reward +0.959, onpt=0.458, bin=0.333
- Update 300: same — 0/50 wins, 50/50 draws, score 0/0, kills **4.0/6.0**, mean_reward +0.958, onpt=0.456, bin=0.333

**Training progression (updates 200-340):** Entropy ~1.90-1.91, bin=0.331-0.333, onpt oscillating 0.09-0.60, dist oscillating 0.10-0.36. No meaningful trend.

**Root cause: the draw basin at 30s/1000dmg/basic/LR 1e-6 is completely invariant to entropy changes.** v5 (entropy 0.02) and v5_high_entropy (entropy 0.08) produced identical metrics through 340 updates. The noisy aim angles from higher entropy are uncorrelated with actual enemy motion — misses stay misses. The policy deterministically repeats the same crude aim angles regardless of entropy scale.

**This proves: after 13+ variants across every tested lever (damage, round length, fire rate, BC pretraining, LR, entropy), the draw basin is genuinely inescapable with current hyperparameters against the `basic` bot.** The only remaining untested opponent-design hypothesis is a genuinely weaker combatant.

**Next approach: `weak_basic` bot — walks to cap like `basic` but adds deterministic ±0.5 rad (~±28.6°) aim noise. Same fire rate, same movement, same cap-contesting, but misses frequently. This is the missing curriculum rung: a bot that contests the objective, is weak enough to beat, and still forces combat.**

## 2026-05-15 — Phase 4 weak_basic_v1 diagnostic shortcut (warm-start + BC-only, no PPO)

**Escape Protocol diagnostic before the 1000-update run:** evaluated the existing `phase4_mappo_weak_basic_v1` hypothesis without PPO by loading `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, applying the config's 500-step `walk_and_shoot` BC pretrain, then running 50 eval episodes against `weak_basic`.

**Result:** 0/50 wins, 0/50 losses, 50/50 draws, score 0/0, mean_reward +1.000, kills 6.0/5.0, final_tick 900, trunc=50.

**Interpretation:** BC alone still cannot convert cap-and-fire behavior into score, so this does not clear the draw basin. It is nevertheless a positive diagnostic signal for the `weak_basic` curriculum hypothesis: Team A led kills 6.0/5.0 against the noisy-aim bot, unlike the recent `basic` runs where the bot typically led kills or collapsed us into losses. This satisfies the config's early signal criterion ("Team A kills > bot kills") and justifies spending PPO updates, with `metadata.max_updates_if_no_signal=500` as the stop point if score/wins do not appear.

## 2026-05-15 — Phase 4 weak_basic_v1 PPO run (stopped at update 500)

**Result: weak_basic did not break the draw basin.** The run was intentionally stopped after update 500 because the config's `metadata.max_updates_if_no_signal: 500` criterion was reached with no score or wins.

**Identity:** commit `725fa61589e230f3015a505f1d8ed3f831dd9d0c`, config `experiments/configs/phase4/legacy/archive/phase4_mappo_weak_basic_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/1s3mtfap`, checkpoint `runs/phase4_mappo_weak_basic_v1/mappo/ckpt_0500.pt`, stochastic replay `data/replays/phase4_weak_basic_v1_ckpt0500_stochastic.replay`.

**Eval progression:**
- BC eval: 0/50 wins, 50/50 draws, score 0/0, kills 6.0/5.0, mean_reward +1.000.
- Update 50: 0/50 wins, 50/50 draws, score 0/0, kills 6.0/5.0, mean_reward +1.000.
- Update 100: 0/50 wins, 50/50 draws, score 0/0, kills 3.0/4.0, mean_reward -1.000.
- Update 200: 0/50 wins, 50/50 draws, score 0/0, kills 6.0/5.0, mean_reward +1.000.
- Update 300: 0/50 wins, 50/50 draws, score 0/0, kills 5.0/5.0, mean_reward +0.284.
- Update 400: 0/50 wins, 50/50 draws, score 0/0, kills 6.0/5.0, mean_reward +1.000.
- Update 500: 0/50 wins, 50/50 draws, score 0/0, kills 4.0/4.0, mean_reward +0.650.

**Behavioral autopsy from stochastic replay:** agents do fire constantly (`primary_fire` rate ~0.998 for Team A over 5 dumped episodes) and move while firing (mean move magnitude ~0.677), so the fire action is not dead and the policy is not standing still. The replay/action dump does not show target identity, but all three agents fire almost every decision, which looks like indiscriminate spray rather than selective focus fire. Training `onpt` continued to oscillate from ~0.16 to ~0.66 rather than collapsing permanently to zero; this is the same cap-contact cycle as prior draw-basin runs. Per-agent kill attribution is not available in current eval metrics, only team totals. The opponent's noisy aim reduced combat dominance sometimes, but not enough to create score.

**Conclusion:** the remaining opponent-design hypothesis is falsified at the configured stop point. `weak_basic` changes the kill exchange slightly but does not produce objective conversion. Do not queue another weak-basic hyperparameter variant. The next Phase 4 move should be an Escape Protocol Section 5 architecture change, starting with one isolated lever such as the auxiliary aim prediction head, or human escalation if that engineering direction needs approval.

## 2026-05-15 — Phase 4 auxiliary aim head diagnostic and config

**Implemented Escape Protocol Section 5.1 auxiliary aim prediction head.** This is an architecture change, not a hyperparameter/opponent variant: `ppo.aim_aux_coef` enables an `actor_aim_aux_head` trained with wrapped-angle RMSE to the visible enemy in flat Phase 4 actor observations. It does not change game rules, rewards, observations, actions, or replay format.

**Cheap diagnostic before creating the run config:** in-memory `phase4_mappo_weak_basic_v1 + aim_aux_coef=1.0`, warm-started from `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, trained 500 `walk_and_shoot` BC steps. Fixed-batch auxiliary RMSE dropped **1.8154 → 0.0079 rad**. This satisfies Escape Protocol 5.1's auxiliary loss target (<0.1 rad RMSE) and justifies an isolated PPO probe.

**New config:** `experiments/configs/phase4/probe/phase4_mappo_aux_aim_v1.yaml`. The only intended lever versus `weak_basic_v1` is `ppo.aim_aux_coef: 1.0`. Metadata includes hypothesis, falsification criteria, `max_updates_if_no_signal: 500`, and the diagnostic result. Falsification: if BC aux RMSE does not stay <0.1 rad or update-500 eval is still 50/50 draws, score 0/0, kills no better than weak_basic_v1's 4.0/4.0.

## 2026-05-15 — Phase 4 aux_aim_v1 PPO run (stopped at update 500)

**Result: auxiliary aim prediction learned the supervised target, but did not break the draw basin.** The run reached its configured `max_updates_if_no_signal: 500` stop point with 50/50 draws, score 0/0, and kills worse than `weak_basic_v1`.

**Identity:** commit `a07d73dc214d80d66ebcc2f3a0d0ea633561a448`, config `experiments/configs/phase4/probe/phase4_mappo_aux_aim_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/r41572eu`, checkpoint `runs/phase4_mappo_aux_aim_v1/mappo/ckpt_0500.pt`, stochastic replay `data/replays/phase4_aux_aim_v1_ckpt0500_stochastic.replay`.

**Auxiliary-head diagnostic inside the run:** BC pretrain `aim_aux_rmse` dropped from `1.7855` at step 1 to `0.0076` at step 500, comfortably below the Escape Protocol 5.1 target of 0.1 rad. The architecture probe therefore succeeded at representation supervision but failed at policy improvement.

**Eval progression:**
- BC eval: 0/50 wins, 50/50 draws, score 0/0, mean_reward +0.986.
- Update 50: 0/50 wins, 50/50 draws, score 0/0, kills 6.0/6.0, mean_reward +0.986.
- Update 100: 0/50 wins, 50/50 draws, score 0/0, kills 5.0/5.0, mean_reward +0.492.
- Update 150: 0/50 wins, 50/50 draws, score 0/0, kills 7.0/3.0, mean_reward +1.000.
- Update 200: 0/50 wins, 50/50 draws, score 0/0, kills 1.0/6.0, mean_reward -0.716.
- Update 300: 0/50 wins, 50/50 draws, score 0/0, kills 1.0/6.0, mean_reward -0.677.
- Update 400: 0/50 wins, 50/50 draws, score 0/0, kills 1.0/6.0, mean_reward -0.678.
- Update 500: 0/50 wins, 50/50 draws, score 0/0, kills 1.0/6.0, mean_reward -0.678.

**Behavioral autopsy from stochastic replay:** Team A still fires almost continuously (`primary_fire` rate 0.9987 over 5 dumped episodes), so the failure is not an inactive fire head. Team A also moves while firing (`move_mean` 0.659, moving-while-firing rate 0.982), so this is not a stationary policy. The replay action dump lacks target identity, so focus fire cannot be measured directly; given all policy agents fire nearly every decision and score remains 0/0, the observable behavior is consistent with indiscriminate spray rather than useful target selection. Training `onpt` continued to oscillate (`0.27` to `0.69` in late updates) and never produced objective conversion. Per-agent kills and bot body/headshot attribution are unavailable in current metrics.

**Conclusion:** Escape Protocol 5.1 auxiliary aim prediction is falsified as an isolated Phase 4 fix. It solves the auxiliary representation objective but does not change the PPO behavior enough to score, and the final kill exchange regressed to 1.0/6.0. Do not continue with more aux-coefficient variants. The next Phase 4 action should be a different Escape Protocol Section 5 architecture intervention, or human escalation before adding more Phase 4 configs.

## 2026-05-15 — Phase 4 per-action entropy implementation and config

**Implemented Escape Protocol Section 5.2 per-action-type entropy support** as an isolated MAPPO PPO-loss change. Existing configs preserve `ppo.entropy_coef * total_entropy`; new optional fields `ppo.entropy_coef_move`, `ppo.entropy_coef_aim`, and `ppo.entropy_coef_binary` split the entropy bonus across movement, aim, and binary action components. PPO metrics now log `entropy_move`, `entropy_aim`, `entropy_binary`, `entropy_other`, and `entropy_bonus`.

**Implementation verification:** focused tests passed for entropy decomposition, config parsing, update metrics, loss-mask behavior, Phase 4 config smoke, warm-start compatibility, ruff, and `git diff --check`. Unit-level diagnostic verified `entropy_bonus = 0.02*entropy_move + 0.15*entropy_aim + 0.05*entropy_binary` when configured.

**New config:** `experiments/configs/phase4/probe/phase4_mappo_per_action_entropy_v1.yaml`. This is an architecture probe based on `weak_basic_v1`; the only intended lever is independent entropy weighting with `move=0.02`, `aim=0.15`, `binary=0.05`. Metadata includes hypothesis, falsification criteria, `max_updates_if_no_signal: 500`, and diagnostic evidence. Falsification: update-500 eval still 50/50 draws, score 0/0, and kills no better than `weak_basic_v1`'s 4.0/4.0, or entropy/action metrics show aim exploration did not increase while fire stayed near `bin ~= 0.33`.

## 2026-05-15 — Phase 4 per_action_entropy_v1 PPO run (stopped at update 500)

**Result: per-action entropy improved the kill exchange slightly but did not break the draw basin.** The run reached the configured `max_updates_if_no_signal: 500` stop point with 50/50 draws and score 0/0 at every eval.

**Identity:** commit `53f1964f49ca1d92019d9a026a35c7c8a5ed5064`, config `experiments/configs/phase4/probe/phase4_mappo_per_action_entropy_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/n3gh2mea`, checkpoint `runs/phase4_mappo_per_action_entropy_v1/mappo/ckpt_0500.pt`, stochastic replay `data/replays/phase4_per_action_entropy_v1_ckpt0500_stochastic.replay`.

**Entropy diagnostics at update 500:** `entropy_move=1.2543`, `entropy_aim=0.6406`, `entropy_binary=0.0189`, `entropy_other≈0`, `entropy_bonus=0.1221`. The configured separate entropy path was active and kept `action_binary_mean=0.3322`, so the fire head did not collapse.

**Eval progression:** BC eval and every PPO eval from update 50 through update 500 were identical: 0/50 wins, 0/50 losses, 50/50 draws, score 0/0, kills 6.0/5.0, mean_reward +1.000.

**Behavioral autopsy from stochastic replay:** Team A fires almost continuously (`primary_fire` rate 0.9969) and moves while firing (`move_mean` 0.639, moving-while-firing rate 0.980), so neither firing nor strafing is dead. Aim deltas remain broad (`abs_aim_mean` 0.668, p90 0.759), but the replay action dump still lacks target identity, so focus fire cannot be measured directly; behavior is consistent with high-volume spray that does not convert into score. Training `onpt` continued to oscillate through late updates (`0.13` to `0.65`) rather than collapsing permanently, but cap contact still never becomes score. Per-agent kill attribution and body/headshot attribution are unavailable.

**Conclusion:** Escape Protocol 5.2 per-action entropy is falsified as an isolated Phase 4 fix. It preserves fire and yields a stable 6.0/5.0 kill edge over weak_basic, better than weak_basic_v1's 4.0/4.0 and aux_aim_v1's 1.0/6.0, but it still produces no wins and no score. Do not queue coefficient variants. The next valid Phase 4 action is another distinct Section 5 architecture intervention, likely 5.3 action masking / invalid fire suppression or human escalation.

## 2026-05-15 — Phase 4 invalid_fire_mask_v1 config

**Circling Detector:** reviewed the last five completed/blocked Phase 4 tasks before creating this config. The completed run tasks remain in the 50/50 draw basin with score 0/0, including `aux_aim_v1` and `per_action_entropy_v1`. This blocks further hyperparameter/opponent variants and permits only a distinct Escape Protocol Section 5 intervention.

**New config:** `experiments/configs/phase4/probe/phase4_mappo_invalid_fire_mask_v1.yaml`. This is an architecture probe based on `weak_basic_v1`; the only intended lever is `ppo.mask_fire_when_no_visible_enemy: true`. It does not combine auxiliary aim or per-action entropy. Metadata includes hypothesis, falsification criteria, `max_updates_if_no_signal: 500`, and the implementation diagnostic evidence. Falsification: update-500 eval still 50/50 draws, score 0/0, and kills no better than `weak_basic_v1`'s 4.0/4.0, or replay/action metrics show no improvement in primary-fire validity and kill-per-fire conversion.

## 2026-05-15 — Phase 4 invalid_fire_mask_v1 PPO run (stopped at update 500)

**Result: invalid-fire masking did not break the draw basin.** The run reached its configured `max_updates_if_no_signal: 500` stop point with 50/50 draws, score 0/0, and symmetric kills.

**Identity:** commit `928bf70ea25ac7da00598612f9b4f823b66bce2b`, config `experiments/configs/phase4/probe/phase4_mappo_invalid_fire_mask_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/x4mketjt`, checkpoint `runs/phase4_mappo_invalid_fire_mask_v1/mappo/ckpt_0500.pt`, stochastic replay `data/replays/phase4_invalid_fire_mask_v1_ckpt0500_stochastic.replay`.

**Final eval:** update 500 mean_reward `+0.995`, wins `0/50`, losses `0/50`, draws `50/50`, score `0.00/0.00`, kills `5.0/5.0`. Eval history: update 50 `6/5`, 100 `3/4`, 150 `6/5`, 200 `6/5`, 250 `4/2`, 300 `6/5`, 350 `5/5`, 400 `5/5`, 450 `5/5`, 500 `5/5`.

**Behavioral autopsy from stochastic replay:** Team A still fires almost continuously (`primary_fire` rate `0.9987`) and moves while firing (`move_mean` `0.664`, moving-while-firing rate `0.982`), so firing and strafing remain active. Aim deltas remain broad (`abs_aim_mean` `0.672`, p90 `0.760`), consistent with continued spray rather than reliable target conversion. The mask did not materially change action availability: W&B reports `train/fire_valid_fraction = 0.9994`, meaning almost every sampled timestep had an alive visible enemy under the actor-observation predicate. Focus-fire attribution, per-agent kills, and body/headshot attribution remain unavailable from the replay format.

**Conclusion:** Escape Protocol 5.3 invalid-fire masking is falsified as an isolated fix. It shows that wasted fire on invisible/no-enemy timesteps is not the main bottleneck in the current weak_basic draw basin. Do not queue mask variants or combine this with 5.1/5.2 without a new diagnostic. Next valid Phase 4 action is a distinct Section 5 intervention such as 5.4 aim-only mini-game, or human escalation.

## 2026-05-15 — Phase 4 aim_only_v1 mini-game config

**Circling Detector:** reviewed the last five completed/blocked Phase 4 tasks before creating this config. Recent full 3v3 probes still show the same scoreless draw basin, and the invalid-fire mask showed `fire_valid_fraction = 0.9994`; further hyperparameter, opponent, entropy, aux-aim, or mask variants remain blocked.

**New config:** `experiments/configs/phase4/probe/phase4_mappo_aim_only_v1.yaml`. This is an Escape Protocol 5.4 synthetic mini-game diagnostic using `env.mini_game: aim_only`, not a full 3v3 objective run. It preserves the Phase 4 actor observation/action/checkpoint interface and warm-starts from `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`. Metadata includes hypothesis, falsification criteria, `max_updates_if_no_signal: 200`, and implementation diagnostic evidence. Success threshold: by update 200 greedy eval `mean_team_a_kills >= 48` per 32-decision episode, equivalent to at least 50% hit rate over 3 agents.

## 2026-05-15 — Phase 4 aim_only_v1 mini-game run (success at update 80, completed update 200)

**Result: the actor can learn visible-target aim in isolation.** The aim-only mini-game crossed its success threshold at update 80 and reached near-ceiling greedy hit rate by update 120. This is a positive diagnostic for Escape Protocol 5.4, not a full Phase 4 escape.

**Identity:** commit `1f93020a78d9b3f8d0a1112c89523127e3ff6ee3`, config `experiments/configs/phase4/probe/phase4_mappo_aim_only_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/d6qgug61`, checkpoint `runs/phase4_mappo_aim_only_v1/mappo/ckpt_0200.pt`, final checkpoint `runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt`.

**Eval trajectory:** update 20 `12.06`, 40 `18.80`, 60 `28.66`, 80 `64.66`, 100 `86.54`, 120 `93.12`, 140 `94.68`, 160 `95.04`, 180 `94.02`, 200 `94.96` mean Team A hits per 32-decision episode. The configured success threshold was `48` hits, so the mini-game succeeded by update 80 and finished at `94.96 / 96` possible hits.

**Interpretation:** The Phase 4 actor architecture and PPO loop can learn the direct mapping from visible `enemy_relative_position` to `aim_delta` plus `primary_fire` when reward is immediate. The full 3v3 draw basin is therefore not explained by a basic inability to represent or optimize the visible-target aim mapping. The next valid diagnostic is to warm-start a full weak_basic 3v3 probe from `runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt` without combining other Section 5 interventions, then test whether the isolated aim skill transfers under movement, cooldown, and objective pressure.

## 2026-05-15 — Phase 4 aim_transfer_v1 full 3v3 config

**Circling Detector:** reviewed the last five completed/blocked Phase 4 tasks before creating this config. The only recent positive result is the synthetic aim-only mini-game; full 3v3 probes remain scoreless. This permits a Section 5.4 transfer probe, not another hyperparameter/opponent variant.

**New config:** `experiments/configs/phase4/probe/phase4_mappo_aim_transfer_v1.yaml`. This returns to the normal weak_basic 3v3 objective environment and inherits `weak_basic_v1` except for `run.init_from_checkpoint: runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt`. It does not enable auxiliary aim, per-action entropy coefficients, or invalid-fire masking. Metadata includes hypothesis, falsification criteria, `max_updates_if_no_signal: 500`, and the successful aim-only diagnostic. Falsification: by update 500 still 50/50 draws, score 0/0, and kills no better than `weak_basic_v1`'s 4.0/4.0, or replay shows active fire/movement without transfer into kill/score conversion.

## 2026-05-15 — Phase 4 aim_transfer_v1 full 3v3 run (stopped at update 500)

**Result: aim-only skill did not transfer into weak_basic 3v3.** The run reached its configured stop point with 50/50 draws, score 0/0, and kills worse than `weak_basic_v1`.

**Identity:** commit `34a2985351f546526c31f9f8ee99acb626ebfa29`, config `experiments/configs/phase4/probe/phase4_mappo_aim_transfer_v1.yaml`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/9n07ntl5`, checkpoint `runs/phase4_mappo_aim_transfer_v1/mappo/ckpt_0500.pt`, stochastic replay `data/replays/phase4_aim_transfer_v1_ckpt0500_stochastic.replay`.

**Final eval:** update 500 mean_reward `-1.000`, wins `0/50`, losses `0/50`, draws `50/50`, score `0.00/0.00`, kills `0.0/3.0`. Eval history: update 50 `6/5`, 100 `5/5`, 150 `1/7`, 200 `0/6`, 250 `0/3`, 300 `0/3`, 350 `0/3`, 400 `0/3`, 450 `0/3`, 500 `0/3`.

**Behavioral autopsy from stochastic replay:** Team A still fires almost continuously (`primary_fire` rate `0.9996`) and moves while firing (`move_mean` `0.655`, moving-while-firing rate `0.980`), so the policy did not lose the basic fire/strafe behavior. Aim deltas remain broad (`abs_aim_mean` `0.677`, p90 `0.754`), but the full 3v3 result still produces no score and no learner kills in greedy eval after update 150. The replay format still cannot identify focus-fire targets or body/headshot attribution. This is consistent with the synthetic aim mapping being overwritten, miscalibrated, or insufficient once moving targets, cooldown timing, objective pressure, and BC/PPO full-env updates are reintroduced.

**Conclusion:** Escape Protocol 5.4 transfer is falsified in this form. The actor can learn aim in isolation, but warm-starting full weak_basic 3v3 from that checkpoint plus the standard walk-and-shoot BC does not preserve/transfer the skill into score or kills. Do not queue aim-transfer variants without a new diagnostic that directly measures whether BC or early PPO erases the mini-game aim mapping. The remaining valid options are human escalation or a new structural diagnostic that instruments hit/aim error during full-env BC/PPO.

## 2026-05-15 — Phase 4 aim-only retention diagnostic

**Question:** where does the aim-only mapping disappear: before full-env BC, during the standard `walk_and_shoot` BC pass, or later during PPO?

**Diagnostic:** loaded `runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt` into the full Phase 4 MAPPO model, evaluated it on the synthetic aim-only env, then applied the same 500-step `walk_and_shoot` BC pretrain used by `aim_transfer_v1`, and evaluated again.

**Result:** before BC, the checkpoint retained the synthetic skill: aim-only eval `94.84/96` hits. Direct full weak_basic eval before BC was unusable for objective play: `0/50` wins, `0/50` draws, score `0/7`, kills `0/0`. After the 500-step full-env BC pass, aim-only eval collapsed to `0.02/96` hits, while full weak_basic eval returned to the old draw-basin BC behavior: `50/50` draws, score `0/0`, kills `6/5`. Actor policy mean absolute drift across actor body/heads/log_std was `0.0321`.

**Conclusion:** the standard walk-and-shoot BC pass erases the synthetic aim-only mapping before PPO. That explains why `aim_transfer_v1` showed the old draw basin rather than transfer. Do not retry aim-transfer with the same post-load BC. The next useful work is either human escalation or a new structural design that composes movement/objective BC with a protected aim skill, such as freezing aim-related layers during movement BC or replacing the heuristic BC target with the mini-game-trained aim target.

## 2026-05-15 — Phase 4 aim_freeze_bc_v1 first-eval probe

**Circling Detector:** reviewed the last five completed/blocked Phase 4 tasks before creating this config. Recent full 3v3 probes remain scoreless or worse, so this work is allowed only as the human-selected structural Option 1 from `docs/reports/phase4_stalemate_escalation.md`, not as a hyperparameter variant.

**Implementation:** commit `84ee9ac` added `run.bc_freeze_actor_aim`, which freezes the shared actor trunk during BC and masks gradient updates for continuous action index 2, the aim row. Focused tests verified that the actor embed/body/GRU and aim row remain unchanged with the flag enabled while movement rows and the binary head can still train.

**Retention diagnostic:** loaded `runs/phase4_mappo_aim_only_v1/mappo/ckpt_final.pt`, ran 500-step `walk_and_shoot` BC with `freeze_actor_aim=true`, then evaluated both environments. Aim-only skill was preserved: `94.80/96` hits before BC and `94.58/96` after BC. Protected parameters had zero max drift for actor embed/body and the aim row. Full weak_basic eval after frozen BC was negative: `0/50` wins, `50/50` losses, score `0/7`, kills `0/3`.

**First PPO eval:** config `experiments/configs/phase4/probe/phase4_mappo_aim_freeze_bc_v1.yaml`, config commit `e894072`, seed `3519994490`, W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/88w0yivd`, checkpoint `runs/phase4_mappo_aim_freeze_bc_v1/mappo/ckpt_0050.pt`, stochastic replay `data/replays/phase4_aim_freeze_bc_v1_ckpt0050_stochastic.replay`. The run was intentionally capped at update 50 (`metadata.max_updates_if_no_signal: 50`).

**Eval result:** BC eval and update-50 eval were identical: mean_reward `-11.000`, wins `0/50`, losses `50/50`, draws `0/50`, score `0.00/7.00`, kills `0.0/3.0`. This is worse than the standard-BC retention baseline of `50/50` draws, score `0/0`, kills `6/5`.

**Behavioral autopsy:** stochastic eval did not reveal hidden capability: wins `0/50`, losses `43/50`, draws `7/50`, score `0.00/4.88`, kills `0.0/1.4`, mean_reward `-9.6`. The stochastic replay shows the policy can still fire and move while firing (`primary_fire` rate `0.725`, `move_mean` `0.723`, moving-while-firing rate `0.722`, `abs_aim_mean` `0.554`), but it does not convert any learner kills or score. Focus-fire attribution, per-agent kills, and body/headshot attribution remain unavailable from the replay format.

**Conclusion:** Escalation Option 1 is partially successful but falsified as a transfer fix. Freezing the aim pathway during BC preserves the synthetic aim-only mapping, but it leaves the full 3v3 policy unable to contest weak_basic; both BC eval and the first PPO eval collapse to 50/50 losses with zero learner score and zero learner kills. Do not run `aim_freeze_bc_v1` to 500 updates without a new diagnostic or design change. The next useful direction is a structural composition change that preserves movement/objective competence while adding the aim skill, such as starting from the movement checkpoint and distilling the aim-only head into it, or replacing heuristic BC aim targets with the mini-game-trained aim target without freezing the full shared trunk.

## 2026-05-15 — Phase 4 aim_target_bc_v1 first-eval probe

**Human direction:** after `aim_freeze_bc_v1` was falsified, the human selected
the structural target-replacement path: use the mini-game-trained aim mapping
as the aim label during full-env `walk_and_shoot` BC while leaving
movement/fire labels intact and not freezing shared actor layers.

**Implementation/config:** commit `24a348d` added `run.bc_aim_target_checkpoint`
for `walk_and_shoot` BC. When set, BC keeps the existing movement/fire targets
but replaces continuous aim action index 2 with deterministic inference from
the frozen aim-only checkpoint on visible-enemy timesteps. The probe config
`experiments/configs/phase4/probe/phase4_mappo_aim_target_bc_v1.yaml` also
uses `run.bc_aim_rehearsal_batch_size: 1024`, mixing full-env BC samples with
mini-game rehearsal samples so movement/objective BC and mini-game aim are
learned simultaneously without freezing.

**Diagnostics before PPO:** checkpoint-only aim labels on full-env samples
preserved only `42/96` synthetic hits after BC, so rehearsal was required.
With rehearsal enabled, 500-step BC preserved the synthetic aim skill at
`90/96` hits and `96/96` fires. Full weak_basic BC eval was still the old draw
basin: `0/50` wins, `50/50` draws, score `0/0`, kills `5/5`.
Diagnostic artifact:
`runs/phase4_mappo_aim_target_bc_v1/diagnostics/bc_aim_target_diagnostic.json`.

**First PPO eval:** commit `24a348d`, config
`experiments/configs/phase4/probe/phase4_mappo_aim_target_bc_v1.yaml`, seed
`3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/8pgcxmbt`, checkpoint
`runs/phase4_mappo_aim_target_bc_v1/mappo/ckpt_0050.pt`, stochastic replay
`data/replays/phase4_aim_target_bc_v1_ckpt0050_stochastic.replay`. A prior
W&B run `4tgm7i0q` failed immediately because it was launched from `python/`
and resolved checkpoint paths relative to the wrong working directory; the
root venv launch succeeded.

**Eval result:** BC eval and update-50 eval were both mean_reward `+1.000`,
wins `0/50`, losses `0/50`, draws `50/50`, score `0.00/0.00`, kills `5.0/5.0`.
This passes the synthetic aim-retention gate but fails the full weak_basic
transfer gate: no wins, no score, and no kill improvement over the standard
BC retention baseline (`6/5`) or the per-action-entropy best signal (`6/5`).

**Behavioral autopsy:** the stochastic replay confirms agents still fire at
visible enemies in the no-fog Phase 4 setup: Team A primary-fire rate
`0.983`. They also move while firing: Team A move rate `0.988` and
moving-while-firing rate `0.988`. The policy therefore has active fire and
strafe behavior, but the full-env eval still produces no score separation.
Focus-fire attribution, per-agent kill concentration, and bot body/headshot
attribution remain unavailable from the replay format.

**Conclusion:** replacing/rehearsing the mini-game aim target preserves much
more isolated aim skill than standard BC, but it is falsified as a first-eval
full 3v3 transfer fix. The remaining failure is composition under full-env
combat/objective dynamics, not simple BC erasure of the synthetic aim mapping.
Do not extend `aim_target_bc_v1` to a long run without a new diagnostic or
instrumentation that explains why 90/96 synthetic aim hits still become only
scoreless 5/5 weak_basic draws.

## 2026-05-15 — Phase 4 next-experiment strategic analysis

**Question:** should the next Phase 4 experiment implement the proposed
`weak_basic_v2` opponent nerf, or should the plan change after 18+ falsified
variants and the latest `aim_target_bc_v1` result?

**Decision:** do not run `weak_basic_v2` next. It changes opponent aim noise,
bot cooldown, bot damage, round length, LR, entropy, and BC warm start, but
those axes have already been exhausted or weakened by prior falsifications:
`weak_basic_v1` still drew, reduced-fire/cooldown variants did not escape the
basin, damage variants changed score margins without producing learner score,
and LR/entropy variants produced at most transient kill edges. Running another
multi-axis opponent nerf would risk manufacturing wins without explaining why
preserved synthetic aim still fails in full 3v3.

**Implementation:** added `python/scripts/analyze_replay_combat.py`, a replay
diagnostic that reconstructs existing text replays through the C++ sim and
reports per-slot/team fire commands, visible-fire commands, damage-producing
hit deltas, kill deltas, damage, nearest-visible-target aim error, and target
distribution. It detects multi-episode replay dumps by tick rollback and resets
the sim with the replay seed plus episode index. Outputs were written to
`runs/phase4_replay_combat_diagnostics/`.

**Cross-replay autopsy:** recent stochastic replays show the same pattern.
Team A fires frequently and usually fires with a visible target, but hit
conversion is poor. Team A damage-producing hits per fire command were
`0.0096-0.0219`, while Team B converted `0.0231-0.0455`. Team A nearest-visible
aim error stayed high (`1.466-1.684` radians), and target attribution remained
diffuse instead of a reliable focus-fire policy. In the best recent probe,
`aim_target_bc_v1`, Team A converted `97/4422` fire commands into damage and
`2` kill deltas; Team B converted `201/4420` fire commands and `30` kill
deltas.

**Recommendation:** the next actual run should be instrumentation-gated
combat composition, not `weak_basic_v2`. Add these hit/aim/focus metrics to the
eval path for the next probe, then test a target-conditioned combat head or
explicit target-selection head that keeps movement/objective behavior intact
while conditioning aim/fire on a chosen enemy slot. Stop early if update 50
does not improve hit/fire, target concentration, or score over
`aim_target_bc_v1`.

**Report:** `docs/reports/phase4_next_experiment_recommendation.md`.

## 2026-05-15 — Phase 4 target-conditioned combat head probe

**Question:** can an internal target-selection head and target-conditioned
aim/fire path improve weak_basic combat composition without changing rules,
opponent, rewards, damage, round length, action space, or observation space?

**Implementation:** added opt-in Phase 4 MAPPO target conditioning:
`actor_target_selection_head` emits three enemy-slot logits, the actor
reconstructs three enemy candidate positions from the ordered team observation
batch, aim/fire heads condition on the soft selected target, and BC/PPO can add
a nearest-visible target-selection auxiliary loss. Eval now logs Team A/B
hit/fire, visible-fire rate, nearest-visible aim error, target concentration
entropy, and damage per fire command.

**Run:** config
`experiments/configs/phase4/probe/phase4_mappo_target_cond_v1.yaml`, seed
`3519994490`, weak_basic, `1000` damage, `30s` rounds, LR `1e-6`, entropy
`0.02`, aim-only checkpoint warm start, walk-and-shoot BC with mini-game
aim-target replacement. BC used `500` steps with `256` BC batch and `256`
aim-rehearsal batch to fit the interactive runtime.

**BC gate result:** BC completed `500/500` steps. Target auxiliary accuracy
rose from `0.188` to `0.829`. Post-BC weak_basic eval was still scoreless:
mean_reward `+0.534`, wins `0/50`, draws `50/50`, score `0.00/0.00`. Team A/B
hit/fire was `0.0398/0.0635`; Team A/B nearest-visible aim error was printed
as `1.300/0.624` rad. The strict gate required Team A hit/fire `>0.025` and
Team A aim error `<1.3`; hit/fire passed, aim error was just above threshold,
so the gate failed and PPO did not start.

**Conclusion:** target selection is learnable and improves Team A hit/fire
relative to the previous replay-autopsy range, but it does not reduce full-env
aim error enough or escape the scoreless weak_basic draw basin. This config is
falsified at the BC gate. Report:
`docs/reports/phase4_target_conditioned_combat_probe.md`.

## 2026-05-18 — Phase 4 target_cond_v1 required rerun

**Reason:** autonomous experiment-runner priority requested the
target-conditioned combat head run first if no ready/running target-conditioned
task existed on the board. A fresh kanban task `t_19e3b695b96` was created and
run from repo root so checkpoint paths resolved correctly.

**Preflight:** `make build-cpp`, `make py-install`,
`python/.venv/bin/python -m pytest python/tests/test_mappo_aux_aim.py -xvs`,
and a `Path(...)`-based `load_config` check passed. The documented string-form
`load_config(...)` one-liner is stale because `load_config` now expects a
`Path`.

**Run:** commit `0d22c3e`, config
`experiments/configs/phase4/probe/phase4_mappo_target_cond_v1.yaml`, seed
`3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/rwnuphfj`, checkpoint
manifest `runs/phase4_mappo_target_cond_v1/mappo/checkpoint_manifest.json`.

**BC gate result:** BC completed `500/500` steps. Target auxiliary accuracy
rose from `0.188` at step 1 to `0.828` at step 500. Post-BC eval was still
scoreless: mean_reward `-0.732`, wins `0/50`, losses `0/50`, draws `50/50`,
score `0.00/0.00`, kills `5.0/5.0`. Team A/B hit/fire was `0.0383/0.0716`.
Team A/B nearest-visible aim error was `1.308/1.129` rad. The strict BC gate
failed because Team A aim error did not satisfy `<1.300`, so PPO did not start.

**Conclusion:** the rerun confirms the earlier target-conditioned probe
outcome. Target conditioning improves Team A hit/fire above the `0.025` gate,
but not aim error or score. Per the configured falsification criterion, this
architecture is not sufficient as the next Phase 4 bottleneck fix.

## 2026-05-18 — Phase 4 weak_basic_v2 curriculum run

**Reason:** after the target-conditioned rerun failed its BC gate, Priority 2
was skipped because `v7_holdshoot_v2`/`v3` were already falsified in
`docs/reports/v7_holdshoot_failure_analysis.md`. Priority 3 required a weaker
objective-contesting opponent. The existing `weak_basic` bot had hard-coded
`±0.5` rad aim noise and no config surface for opponent-only fire cadence, so
this run added a narrow scripted opponent variant `weak_basic_v2`: same
objective movement as `weak_basic`, deterministic `±1.5` rad aim noise, and a
deterministic 60-tick primary-fire cadence. No sim rules, rewards,
obs/action spaces, replay format, or MAPPO/PPO code changed.

**Config/run:** `experiments/configs/phase4/probe/phase4_mappo_weak_basic_v2.yaml`,
seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3byh514d`, final
checkpoint `runs/phase4_mappo_weak_basic_v2/mappo/ckpt_0500.pt`, best-eval
alias update `400`, stochastic replay
`data/replays/phase4_weak_basic_v2_ckpt0500_stochastic.replay`, replay
diagnostic
`runs/phase4_replay_combat_diagnostics/phase4_weak_basic_v2_ckpt0500_stochastic.json`.

**Verification:** `make build-cpp`, `make py-install`,
`python/.venv/bin/python -m pytest python/tests/test_phase0_determinism.py
python/tests/test_env.py::test_all_valid_opponent_bots_instantiate
python/tests/test_phase4_mappo_env.py::test_invalid_opponent_bot_raises
python/tests/test_mappo_aux_aim.py -q`, Path-based config load,
`make test-cpp`, `ruff check` on touched Python files, and `git diff --check`
passed.

**Eval trajectory:** BC eval was 50/50 draws, score `0/0`, kills `7/1`,
Team A/B hit/fire `0.0678/0.4000`, Team A/B aim error `1.532/0.803`. PPO
improved the greedy combat margin early but never produced score:
update 50 `9/0` kills, update 100 `9/1`, update 150 `9/1`, update 200
`9/1`, update 250 `9/2`, update 300 `9/2`, update 350 `8/2`, update 400
`9/2`, update 450 `9/2`, update 500 `8/2`. Every eval remained `0/50` wins,
`0/50` losses, `50/50` draws, score `0.00/0.00`.

**Final eval:** update 500 mean_reward `+1.000`, wins `0/50`, losses `0/50`,
draws `50/50`, score `0.00/0.00`, kills `8.0/2.0`, Team A/B hit/fire
`0.0692/0.3714`, visible-fire `0.947/1.000`, aim error `1.521/1.409`, damage
per fire command `69.2/371.4`.

**Replay diagnostic:** the stochastic 5-episode replay was worse than greedy
eval. Team A/B damage hit per fire command was `0.0169/0.2765`, kill deltas
`4/4`, and mean nearest-visible aim error `1.557/1.447` rad. The learner fired
`8675` times versus the opponent's `434` commands, but the opponent still
matched kills through much higher shot value.

**Conclusion:** `weak_basic_v2` is a partial combat-signal improvement but not
a winning curriculum. It manufactured a greedy kill edge against a much weaker
opponent, yet still produced zero score and zero wins by update 500. The
remaining bottleneck is objective conversion/timing under combat, not merely
opponent strength. Proceeding to a simplified 1v1 combat diagnostic is
justified; more weak-opponent 3v3 tuning is unlikely to explain the scoreless
basin.

## 2026-05-18 — Phase 4 combat_1v1_v1 mini-game run

**Reason:** Priority 4 after target-conditioned combat failed the BC gate,
hold-and-shoot cap shaping was already falsified, and `weak_basic_v2` produced
a kill edge but no 3v3 score or wins. The full Phase 4 env intentionally owns
`team_size=3`, so this implemented a separate `env.mini_game: combat_1v1`
route that preserves Phase 4 actor/critic/action tensor shapes while
activating only one learner slot against one visible duel target.

**Implementation/config:** added `Phase4Combat1v1MappoEnv`, registry routing,
tests, and `experiments/configs/phase4/probe/phase4_mappo_combat_1v1_v1.yaml`.
The synthetic target drifts in aim space, has `3` HP, and respawns after kills.
Reward is direct: hit reward, kill bonus, miss/no-fire penalties, and a small
aim-error penalty. This changes only the synthetic mini-game path, not sim
rules, rewards, observation/action spaces, replay format, or MAPPO/PPO.

**Run:** seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/inuw33u7`, final
checkpoint `runs/phase4_mappo_combat_1v1_v1/mappo/ckpt_final.pt`, checkpoint
manifest `runs/phase4_mappo_combat_1v1_v1/mappo/checkpoint_manifest.json`.
No C++ replay artifact was produced because this is a Python synthetic
mini-game, not a text replay over the native simulator.

**Verification:** focused tests for `combat_1v1` and existing `aim_only`
routing passed (`9 passed`), `ruff check` on touched Python files passed,
Path-based config load passed, and `git diff --check` passed.

**Eval trajectory:** mean Team A kills per 64-decision episode rose from
`2.16` at update 20 to `10.44` at update 200. The configured success threshold
was `12`; the run did not clear it, but it was still improving at the final
checkpoint. Eval kills/score: update 20 `2.16`, 40 `2.52`, 60 `2.60`, 80
`2.06`, 100 `3.28`, 120 `6.88`, 140 `8.92`, 160 `9.26`, 180 `8.76`, 200
`10.44`.

**Conclusion:** the simplified 1v1 combat skill is learnable but not solved by
the first 200-update budget. This is a partial positive diagnostic and justifies
one continuation run before declaring the simplified combat hypothesis
exhausted.

## 2026-05-18 — Phase 4 combat_1v1_v2 continuation run

**Reason/config:** `combat_1v1_v1` ended at `10.44/12` mean kills and was still
improving, so `experiments/configs/phase4/probe/phase4_mappo_combat_1v1_v2.yaml`
continued from `runs/phase4_mappo_combat_1v1_v1/mappo/ckpt_final.pt` for 100
updates with LR `1e-4`.

**Run:** seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/0exrak0h`, final
checkpoint `runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt`, checkpoint
manifest `runs/phase4_mappo_combat_1v1_v2/mappo/checkpoint_manifest.json`.

**Result:** the simplified 1v1 combat gate cleared at update 40 and stayed
above threshold. Eval mean Team A kills/score: update 20 `11.50`, update 40
`12.68`, update 60 `12.08`, update 80 `13.16`, update 100 `12.98`. The best
eval was update 80 with `13.16` kills per 64-decision episode; final eval was
`12.98`, above the `12` success threshold.

**Conclusion:** the simplified 1v1 combat mini-game is solved enough to count
as a positive diagnostic. Phase 4's full 3v3 failure is therefore not a basic
inability to learn the duel reward under Phase 4 tensors; the next unanswered
question is transfer/composition back into full 3v3 objective play.

## 2026-05-18 — Phase 4 combat_1v1_transfer_v1 full 3v3 probe

**Reason/config:** after `combat_1v1_v2` cleared the simplified duel gate, this
tested the direct scale-back-up path: warm-start full 3v3 `weak_basic_v2` from
`runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt`, skip BC to avoid
erasing the synthetic combat mapping, and cap PPO at 50 updates.
Config: `experiments/configs/phase4/probe/phase4_mappo_combat_1v1_transfer_v1.yaml`.

**Run:** seed `3519994490`, W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/29bskhax`, checkpoint
`runs/phase4_mappo_combat_1v1_transfer_v1/mappo/ckpt_0050.pt`, stochastic
replay `data/replays/phase4_combat_1v1_transfer_v1_ckpt0050_stochastic.replay`,
replay diagnostic
`runs/phase4_replay_combat_diagnostics/phase4_combat_1v1_transfer_v1_ckpt0050_stochastic.json`.

**Result:** direct transfer failed. Update-50 eval was mean_reward `-11.000`,
wins `0/50`, losses `50/50`, draws `0/50`, score `0.00/37.00`, kills
`0.0/0.0`, Team A/B hit/fire `0.0000/0.0000`, and aim error `1.574/1.341`.
Training metrics showed no objective competence: on-point contact collapsed to
`0.000` by late updates and distance rose to `1.410`.

**Replay diagnostic:** the stochastic replay confirmed spray without
conversion. Team A fired `8602` times with visible targets but produced only
`4` damage hits (`0.00047` damage hits per fire command), `0` kill deltas, and
mean nearest-visible aim error `1.581` rad. Team B fired only `450` times and
also produced no kills in the stochastic sample, but won by uncontested
objective score.

**Conclusion:** the solved 1v1 combat mini-game does not directly transfer to
full 3v3 objective play. At this point the original priority queue is
exhausted: target-conditioned combat failed its BC gate, hold-and-shoot v2/v3
were already falsified, weak_basic_v2 manufactured a kill edge without score,
the 1v1 simplification was solved, and direct transfer from that simplified
skill collapsed objective competence. The remaining work is a new composition
design, not another run from the current hypothesis queue.

## 2026-05-18 — Phase 4 composition_rehearsal_v1 BC gate

**Reason/config:** tested opt-in multi-teacher composition rehearsal before PPO:
objective teacher `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` and combat
teacher `runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt` distilled into a
single student policy initialized from the objective checkpoint. Config:
`experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v1.yaml`,
commit `81dcfe7a4fb4a7d309c48234dd79655b5d56c578`, seed `3519994490`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/oeld227b`, output
directory `runs/phase4_mappo_composition_rehearsal_v1`, final checkpoint
`runs/phase4_mappo_composition_rehearsal_v1/mappo/ckpt_final.pt`.

**Preflight:** `make build-cpp && make py-install` passed.
`pytest tests/test_mappo_composition_rehearsal.py -xvs` passed (`5 passed`).
`pytest tests/test_mappo_aux_aim.py -xvs` passed (`13 passed`). The literal
string-path smoke command for `load_config` failed because the helper expects a
`Path`; the equivalent `Path(...)` check passed with `composition_pretrain=True`.

**BC gate:** failed after 1000 composition rehearsal steps, so PPO was skipped.
Objective retention passed: on-point `0.68278`, wins/losses/draws `0/0/50`,
score `0.00/0.00`, kills `0.0/0.0`. Combat retention passed: mean kills
`12.78` per 64-decision episode. Full 3v3 diagnostic failed on hit/fire:
Team A hit/fire `0.01676` versus the `0.02` gate, aim error `1.18534` rad,
wins/losses/draws `0/0/50`, score `0.00/0.00`, kills `0.0/0.0`. Team B full
diagnostic hit/fire was `0.32222` with aim error `1.50110` rad.

**PPO trajectory:** none. The BC gate failed and the run exited before PPO
updates or replay generation.

**Conclusion:** falsified at the BC stage. Composition rehearsal as implemented
preserved objective occupancy and the simplified 1v1 combat kill count, but did
not preserve enough full 3v3 combat conversion to clear the hit/fire gate.
Escalate to human review for the next strategy decision.

## 2026-05-18 — Phase 4 composition_rehearsal_v1_lowgate PPO

**Reason/config:** tested Option 1 from the composition rehearsal follow-up:
copy v1 to
`experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v1_lowgate.yaml`,
lower only the full 3v3 hit/fire BC gate from `0.02` to `0.015`, and
warm-start from
`runs/phase4_mappo_composition_rehearsal_v1/mappo/ckpt_final.pt`. Config/setup
commit at launch: `e60dfbe1b8873c5c4608091fb952f91a20883ad3`, seed
`1779134701`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/7bieksp0`, output
directory `runs/phase4_mappo_composition_rehearsal_v1_lowgate`, final
checkpoint
`runs/phase4_mappo_composition_rehearsal_v1_lowgate/mappo/ckpt_final.pt`.
No replay artifact was produced by this training command.

**Preflight:** `make build-cpp && make py-install` passed.
`cd python && .venv/bin/pytest tests/test_mappo_composition_rehearsal.py -xvs`
passed (`5 passed`). `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py
-xvs` passed (`13 passed`). An initial module-entry retry from `python/`
failed before training because root-relative checkpoint paths resolved under
`python/runs`; the successful run used the root console script.

**BC gate:** passed after the warm-started 1000-step composition rehearsal:
objective on-point `0.518` > `0.250`, objective losses `0` <= `0`, combat
kills `12.60` >= `12.00`, full 3v3 Team A hit/fire `0.0152` > lowered gate
`0.0150`, full 3v3 aim error `1.161` < `1.550`.

**PPO trajectory:** ran all 500 updates. Eval stayed draw-only throughout:
update 50 `0W/0L/50D`, score `0.00/0.00`, kills `0.0/3.0`, Team A hit/fire
`0.0140`, aim error `1.094`; update 250 `0W/0L/50D`, score `0.00/0.00`,
kills `0.0/0.0`, hit/fire `0.0135`, aim error `1.218`; update 500
`0W/0L/50D`, score `0.00/0.00`, kills `1.0/0.0`, hit/fire `0.0140`, aim
error `1.137`. The checkpoint manifest selected update 350 as best eval
(`0W/0L/50D`, score `0.00/0.00`, kills `1.0/2.0`, mean reward about `-1.0`).

**Conclusion:** the lowered gate allows PPO to run, but PPO did not produce
objective scoring or wins. Evidence supports treating Option 1 as BC-gate
pass / PPO outcome insufficient rather than a Phase 4 gate clear.

## 2026-05-18 — Phase 4 composition_rehearsal_v2_2000 BC gate

**Reason/config:** tested Option 2 from the composition rehearsal follow-up:
copy v1 to
`experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v2_2000.yaml`,
extend `composition_pretrain_steps` from `1000` to `2000`, keep the original
full 3v3 hit/fire gate `0.02`, and start from the objective teacher checkpoint
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` rather than the v1
composition checkpoint. Config/setup commit at launch:
`72a76188bfcf177098a76686d0ca8f8fa27dbb7d`, seed `1779134702`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/dv1wzk0y`, output
directory `runs/phase4_mappo_composition_rehearsal_v2_2000`, final checkpoint
`runs/phase4_mappo_composition_rehearsal_v2_2000/mappo/ckpt_final.pt`. No
replay artifact was produced by this training command.

**Preflight:** `make build-cpp && make py-install` passed.
`cd python && .venv/bin/pytest tests/test_mappo_composition_rehearsal.py -xvs`
passed (`5 passed`). `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py
-xvs` passed (`13 passed`).

**BC gate:** failed after 2000 composition rehearsal steps, so PPO was skipped.
Objective on-point narrowly passed: `0.264` > `0.250`, but objective losses
failed: `50` > `0`. Combat retention passed with `13.00` kills >= `12.00`.
Full 3v3 hit/fire failed: `0.0173` < original gate `0.0200`. Full 3v3 aim
error passed: `1.295` < `1.550`.

**PPO trajectory:** none. The checkpoint manifest has no best eval update and
aliases `ckpt_final.pt` to `ckpt_last.pt` because no PPO eval ran.

**Conclusion:** extending rehearsal to 2000 steps did not clear the original
BC gate and also damaged objective-match outcome retention. Option 2 is
falsified at BC, not blocked.

## 2026-05-18 — Phase 4 cap_duel_v1 and transfer

**Reason/config:** implemented Strategy 2 from the Phase 4 strategic proposal:
a Phase 4-compatible `cap_duel` mini-game with one active learner, one
scripted recontesting enemy near the objective, and score ticks only when the
learner is on point while the enemy is dead or displaced. Probe config:
`experiments/configs/phase4/probe/phase4_mappo_cap_duel_v1.yaml`, warm-started
from objective checkpoint `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`.
Transfer config:
`experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml`,
warm-started from solved cap-duel checkpoint
`runs/phase4_mappo_cap_duel_v1/mappo/ckpt_0075.pt`. Seed `3519994490`. Base
commit at launch: `d58af34ac731bd858166ba8a645f74d4658103f7`; implementation
commit: this changeset.

**Run:** cap-duel W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/jicvzf48`, local
checkpoint `python/runs/phase4_mappo_cap_duel_v1/mappo/ckpt_0075.pt`. Transfer
W&B `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/2up9mfvb`, final
checkpoint
`python/runs/phase4_mappo_cap_duel_transfer_v1/mappo/ckpt_final.pt`, replay
`replays/phase4_cap_duel_transfer_v1_final.replay`. The cap-duel mini-game run
does not emit simulator replay artifacts.

**Preflight/tests:** `make build-cpp && make py-install` passed.
`cd python && .venv/bin/pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py`
passed (`9 passed`).

**Cap-duel probe:** solved before the 300-update falsification limit. Update
25 eval was still zero-signal (`0W/0L/50D`, score `0.00/0.00`, kills
`0.0/0.0`). Update 50 eval solved: `50W/0L/0D`, mean reward `+5.130`, score
`12.00/0.00`, kills `1.0/0.0`, mean final tick `24.1`. Update 75 remained
solved: `50W/0L/0D`, mean reward `+5.202`, score `12.00/0.00`, kills
`1.0/0.0`, mean final tick `17.5`. The run was intentionally interrupted
after update 75 because the probe had met the "solved or 300 updates" stop
condition.

**Transfer probe:** falsified at update 50 on full `weak_basic_v2` 3v3.
Final eval: mean reward `-11.000`, `0W/50L/0D`, score `0.00/37.00`, kills
`0.0/0.0`, Team A/B hit-fire `0.0017/0.0444`, visible fire `1.000/1.000`,
aim error `1.564/1.550`, target entropy `0.856/0.674`, damage/fire
`1.7/44.4`. Recent training rollout on-point contact was `0.001`, below the
`0.25` falsification threshold.

**Conclusion:** cap-duel is learnable and gives a clean middle-rung signal,
but the solved synthetic skill did not transfer to full 3v3. Strategy 2 is
mini-game positive / transfer falsified, not blocked.

## 2026-05-18 — Phase 4 focus_fire_v1 target conditioning

**Reason/config:** implemented Strategy 3 from the Phase 4 strategic proposal:
team-level focus-fire target conditioning with `team_focus_low_hp` labels,
explicit no-target class, focus-fire auxiliary loss, fallback/same-target
training metrics, and eval-time same-target fraction plus target-selection
entropy. Probe config:
`experiments/configs/phase4/probe/phase4_mappo_focus_fire_v1.yaml`,
warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, full `weak_basic_v2` 3v3,
seed `3519994490`. Base commit at launch:
`6e25e724f64d0e94d8700ed55fd4e9357c039a9a`; implementation commit:
`c015256d` (`Add Phase 4 focus-fire target conditioning`).

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/50wfu032`, final
checkpoint
`python/runs/phase4_mappo_focus_fire_v1/mappo/ckpt_final.pt`, replay
`replays/phase4_focus_fire_v1_final.replay`. An earlier W&B run
`8ovasyp5` was aborted during BC after it exposed a no-target mask bug; it is
not evidence.

**Preflight/tests:** `make build-cpp && make py-install` passed.
`cd python && pytest tests/test_mappo_focus_fire.py -xvs` passed (`6 passed`).
`cd python && pytest tests/test_mappo_composition_rehearsal.py tests/test_mappo_aux_aim.py -x`
passed (`18 passed`).

**BC:** after the no-target mask fix, BC pretrain completed normally:
step 500 loss `0.0017`, continuous loss `0.0014`, binary loss `0.0010`,
target-selection accuracy `1.000`. BC eval was draw-only but already above
the requested hit/fire threshold: mean reward `-0.362`, `0W/0L/50D`, score
`0.00/0.00`, Team A/B kills `6.0/1.0`, hit/fire `0.0468/0.3816`,
same-target fraction `1.000`, focus entropy `0.000`.

**PPO trajectory:** update 50 did not trigger the stop condition because
objective contact had not collapsed despite no score/wins: recent training
on-point `0.751`, eval `0W/0L/50D`, score `0.00/0.00`, kills `5.0/1.0`,
hit/fire `0.0410`, same-target fraction `1.000`, focus entropy `0.000`.
Update 100 met the focus-fire metric thresholds but still produced no scoring
window: mean reward `-1.000`, `0W/0L/50D`, score `0.00/0.00`, kills
`5.0/1.0`, Team A/B hit/fire `0.0417/0.3718`, visible fire `0.949/1.000`,
aim error `1.533/1.537`, nearest-aim target entropy `1.062/0.593`,
same-target fraction `1.000`, focus entropy `0.000`, damage/fire
`41.7/371.8`. Training labels stayed concentrated (`same_tgt` generally
`0.55-0.87`, focus entropy `0.31-0.69`) with fallback rate around `0.64-0.67`.

**Conclusion:** Strategy 3 fixed the measured target-selection concentration
objective and reached the requested update-100 focus metrics, but it did not
create score or win conversion in full 3v3. Phase 4 gate is not cleared;
result is done / behavior insufficient, not blocked.

## 2026-05-18 — Phase 4 cap-duel to focus-fire transfer probe

**Reason/config:** tested whether the solved cap-duel checkpoint's kill-then-cap
timing transfers better when full 3v3 training also uses the focus-fire target
conditioning path. Probe config:
`experiments/configs/phase4/probe/phase4_mappo_cap_duel_focus_fire_v1.yaml`,
warm-started from
`runs/phase4_mappo_cap_duel_v1/mappo/ckpt_0075.pt`, full `weak_basic_v2`
3v3, seed `3519994490`, target-conditioned combat enabled with
`team_focus_low_hp` labels and auxiliary coefficient `0.5`. Base commit at
launch: `46e9c959d1d5c33646f4c6e095216c5249149f9a`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/8pgwe3on`, update-50
checkpoint
`python/runs/phase4_mappo_cap_duel_focus_fire_v1/mappo/ckpt_0050.pt`, replay
`replays/phase4_cap_duel_focus_fire_v1_update50.replay`. The run was
intentionally interrupted immediately after the update-50 eval matched the
falsification rule, so no `ckpt_final.pt` alias was written.

**Preflight:** YAML parse passed. Warm-start compatibility check loaded the
cap-duel checkpoint with `strict=False` and found no unexpected keys; the only
missing weights were the expected new target-conditioning layers:
`actor_target_selection_head.{weight,bias}` and
`actor_target_condition.0.{weight,bias}`. No BC stage was configured for this
probe, so PPO started directly after the warm start.

**PPO result:** update 50 falsified the hypothesis. Recent rollout metrics had
objective contact collapsed: `onpt=0.001`, reward `+0.000/0.000`, Team A
same-target fraction `0.749`, focus entropy `0.562`, fallback rate `0.667`.
The update-50 eval was mean reward `-11.000`, `0W/50L/0D`, score
`0.00/37.00`, kills `0.0/0.0`, Team A/B hit-fire `0.0022/0.0556`, visible
fire `1.000/1.000`, aim error `1.565/1.625`, nearest-aim target entropy
`0.862/0.566`, same-target fraction `0.981/0.000`, focus entropy
`0.036/0.000`, damage/fire `2.2/55.6`.

**Conclusion:** cap-duel warm-start plus focus-fire conditioning still
collapses in full `weak_basic_v2` 3v3. The stop condition was met at update 50:
zero score, zero wins, on-point below `0.25`, and hit/fire below `0.02`.
Result is falsified / done, not blocked.

## 2026-05-18 — Phase 4 explicit combat/objective mode-gated probe

**Reason/config:** tested the hypothesis that full 3v3 transfer collapses
because the actor lacks an explicit fight-vs-cap decision. Added an
`actor_mode_head` and ran
`experiments/configs/phase4/probe/phase4_mappo_mode_gated_v1.yaml`, full
`weak_basic_v2` 3v3, seed `3519994490`, warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`. The config enables
`mode_gated_combat: true`, `mode_aux_coef: 0.3`, and compatible team-focus
target conditioning with `target_selection_aux_coef: 0.3`. Implementation
base before commit: `cd8cbee789cf1896ebaa64bb15be6b620048fdcf`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/u8361ulr`, final
checkpoint
`python/runs/phase4_mappo_mode_gated_v1/mappo/ckpt_final.pt`, update-50
checkpoint
`python/runs/phase4_mappo_mode_gated_v1/mappo/ckpt_0050.pt`. No replay was
produced by this training run.

**Preflight/tests:** `make build-cpp` passed. `make py-install` passed.
`cd python && .venv/bin/pytest tests/test_mappo_mode_gate.py -xvs` passed
(`4 passed`). `cd python && .venv/bin/pytest
tests/test_mappo_composition_rehearsal.py tests/test_mappo_aux_aim.py
tests/test_mappo_focus_fire.py -x` passed (`24 passed`).

**BC:** warm-start loaded with `strict=False`; the new missing
`actor_mode_head.{weight,bias}` entries were accepted by the new-head
allowlist. Walk-and-shoot BC reached step 500 with loss `0.0045`, continuous
loss `0.0025`, binary loss `0.0025`, mode accuracy `1.000`, target-selection
accuracy `1.000`. BC eval remained poor: mean reward `-11.000`, `0W/50L/0D`,
score `0.00/1.93`, Team A/B hit-fire `0.0145/0.2907`, mean combat mode
probability `0.892`, mode accuracy `0.988`.

**PPO result:** update 50 was ambiguous, so the run continued to update 100:
eval mean reward `-1.000`, `0W/0L/50D`, score `0.00/0.00`, kills `3.0/3.0`,
Team A/B hit-fire `0.0151/0.2791`, recent rollout on-point `0.519`, mean
combat probability `0.885`, mode accuracy `0.984`, intentional-fire fraction
`0.885`, objective-focus fraction `0.046`. Final update 100 remained draw-only:
mean reward `-1.000`, `0W/0L/50D`, score `0.00/0.00`, kills `3.0/3.0`,
Team A/B hit-fire `0.0157/0.2791`, visible fire `0.953/1.000`, aim error
`1.562/1.129`, same-target fraction `0.959`, focus entropy `0.077`, mean
combat probability `0.887`, mode accuracy `0.981`, intentional-fire fraction
`0.887`, objective-focus fraction `0.046`.

**Conclusion:** explicit mode gating did not clear Phase 4. It preserved enough
objective contact to avoid update-50 falsification and produced kills, but it
collapsed into high combat-mode probability and still produced zero score and
zero wins. Result is not cleared / evidence insufficient for the hypothesis,
not blocked.

## 2026-05-19 — Phase 4 majority-on-point curriculum smoke

**Reason/config:** implemented an opt-in majority-on-point reward curriculum
to test whether a dense objective-advantage signal can bridge the observed
gap from kill edge to scoring. Added:
`experiments/configs/phase4/probe/phase4_mappo_majority_advantage_smoke.yaml`
and
`experiments/configs/phase4/probe/phase4_mappo_majority_advantage_noanneal_diagnostic.yaml`.
Both use `weak_basic_v2`, seed `3519994490`, warm-start from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, walk-and-shoot BC, and
real objective constants. Base commit before local changes:
`6c7785f82727d71466cc63d564a611ea7f8327b1`.

**Implementation notes:** the simulator scoring rule was not changed. The
new reward term is under `env.reward.majority_on_point_coef`, with
`majority_on_point_anneal_updates` owned by the trainer schedule and final
eval forcing alpha to zero. The old unconditional `on_point_shaping_coef` is
off in both probe configs. Also fixed the walk-and-shoot BC aim target so
action index 2 trains relative `aim_delta` from current `own_aim_unit`, not
an absolute enemy angle.

**Verification:** CMake configure/build for Python 3.13 passed; the extension
was copied from the MSVC `Release/` output directory next to the package for
native Windows import. Focused Python checks passed:
`py -3.13 -m pytest tests/test_reward.py tests/test_phase4_mappo_env.py
tests/test_mappo_team_spirit_ramp.py tests/test_mappo_bc_freeze.py
tests/test_mappo_public_api.py tests/test_bindings_obs.py
tests/test_obs_manifest.py -q` (`119 passed`). Actor/critic leak and obs
contract C++ checks passed:
`ctest --test-dir build -C Release -R
"ActorLeak|ActorObs|CriticObs|ObsDims|ObsUtils" --output-on-failure`
(`44 passed`).

**No-anneal diagnostic run:** local W&B-disabled run; no W&B URL, no replay
produced. Config:
`experiments/configs/phase4/probe/phase4_mappo_majority_advantage_noanneal_diagnostic.yaml`.
Final checkpoints under
`python/runs/phase4_mappo_majority_advantage_noanneal_diagnostic/mappo/`.
Best eval was update 15, mean reward `+0.948`, `0W/0L/20D`, score
`0.00/0.00`, kills `8.0/2.0`. Update 25 final eval remained scoreless but
showed the new diagnostics are useful: Team A/B majority-on-point seconds
`34.00/15.30`, uncontested seconds `0.00/0.10`, alive-edge-no-score
`29.50/8.00`, cap-progress gain `2.0` ticks.

**Annealed smoke run:** local W&B-disabled run; no W&B URL, no replay
produced. Config:
`experiments/configs/phase4/probe/phase4_mappo_majority_advantage_smoke.yaml`.
Final checkpoints under
`python/runs/phase4_mappo_majority_advantage_smoke/mappo/`. Alpha annealed
from `0.2` to `0.0` by update 50, and eval used real reward with the
majority term disabled. Best eval was update 10, mean reward `+0.948`,
`0W/0L/20D`, score `0.00/0.00`, kills `8.0/2.0`. Later evals showed
objective pressure did move but not in the desired direction: update 40 had
Team A/B majority seconds `31.00/16.80`, uncontested seconds `0.00/2.50`,
cap-progress gain `76.0`; update 50 real-reward eval was `0W/20L/0D`, score
`0.00/2.90`, Team A/B majority seconds `31.00/16.80`, uncontested seconds
`0.00/10.90`, cap-progress gain `239.0`.

**Conclusion:** majority shaping produced nonzero majority windows and cap
progress diagnostics, so the signal is live. It did not produce Team A score,
and the annealed run exposed a sharper issue: Team A can own alive/majority
windows while failing to create uncontested capture/scoring time, then Team B
converts uncontested time when alpha is gone. The next design axis should be
explicitly curriculum-only contest clearing/capture timing, or actor
information/coordination, not another plain reward-scale tweak. Phase 4 is
not cleared.

## 2026-05-19 — Phase 4 objective timing fixed-easy long diagnostic

**Reason/config:** tested the "episode too short / capture too slow" hypothesis
directly by adding config-gated objective timing support with canonical defaults
unchanged. The diagnostic config was
`experiments/configs/phase4/probe/phase4_mappo_objective_timing_easy_long.yaml`,
seed `3519994490`, commit
`6c7785f82727d71466cc63d564a611ea7f8327b1`. It used `weak_basic_v2`,
warm-started from `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`,
ran walk-and-shoot BC for 500 steps, and trained for 250 updates with
60-second rounds, objective unlock `5s`, capture `2s`, and majority
advantage shaping annealed to zero by the final eval.

**Implementation notes:** objective unlock/capture ticks are now explicit
`MatchConfig` fields exposed through the Python bindings and config loader.
The canonical defaults remain 15s unlock and 8s capture. Runtime trainer
scheduling can push timing to sync or async vector envs, and eval logs the
objective timing used. W&B required mode was added for long experiment configs
so auth/network failures block instead of silently falling back to local-only
tracking.

**Verification:** CMake configure and build passed for Python 3.13. The
extension was copied from the MSVC `Release/` output directory next to the
Python package for native Windows import. Focused Python checks passed:
`py -3.13 -m pytest tests/test_reward.py tests/test_phase4_mappo_env.py
tests/test_mappo_team_spirit_ramp.py tests/test_mappo_bc_freeze.py
tests/test_mappo_public_api.py tests/test_bindings_obs.py
tests/test_obs_manifest.py -q` (`126 passed`, 2 pytest config warnings).
C++ objective/determinism/obs checks passed:
`ctest --test-dir build -C Release -R
"Objective|Determinism|GoldenReplay|ActorLeak|ActorObs|CriticObs|ObsDims|ObsUtils"
--output-on-failure` (`57 passed`). W&B preflight succeeded with run
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/5ftjfv8h`.

**Run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/qpm9k6an`, runtime
about 27.9 minutes. Checkpoints were written under
`python/runs/phase4_mappo_objective_timing_easy_long/mappo/`, including
`ckpt_0250.pt`, `ckpt_last.pt`, `ckpt_best_eval.pt`, `ckpt_final.pt`, and
`checkpoint_manifest.json`. No replay artifact was produced by this training
run.

**BC/result:** BC eval remained draw-only despite the shortened objective:
mean reward `+0.942`, `0W/0L/50D`, score `0.00/0.00`, Team A/B kills
`8.0/0.0`, majority seconds `41.60/0.00`, uncontested seconds `0.00/0.00`,
cap-progress gain `0.0`.

**PPO result:** all evals at updates 25 through 250 were draw-only and
scoreless. Final eval at update 250: mean reward `-0.356`, `0W/0L/50D`,
score `0.00/0.00`, objective timing `5.00/2.00`, Team A/B kills `1.0/3.0`,
Team A/B hit-fire `0.0217/0.4186`, aim error `1.231/1.908`, majority seconds
`12.40/20.00`, uncontested seconds `0.00/0.00`, alive-edge-no-score seconds
`8.00/16.00`, cap-progress gain `0.0`. The manifest selected the BC checkpoint
as best eval (`best_eval_update_idx: 0`, score `+0.942`), not any PPO update.

**Decision:** falsified / not cleared. The fixed-easy objective did not convert
majority or kill advantage into uncontested capture, score, or wins by the
250-update stop point, so objective unlock/capture duration is not sufficient
as the primary blocker. Per the run strategy, do not spend another long run on
the timing curriculum or 120s canonical control until the no-uncontested-time
failure is addressed. The next design axis should focus on contest clearing
and coordination: actor information/intent, objective entry/retreat behavior,
or a curriculum that creates gradient specifically for clearing the last
contester rather than merely shortening capture.

## 2026-05-19 — Phase 4 current self-play BC probe and Team B action-frame fix

**Reason/config:** implemented and tested the simplest Phase 4 current-vs-current
self-play path: all six Ranger slots are controlled by the current MAPPO
policy, with flat actor observations and per-agent centralized critic views.
The primary W&B probe used
`experiments/configs/phase4/probe/phase4_mappo_current_selfplay_bc_probe.yaml`,
seed `3519994490`, base commit
`6c7785f82727d71466cc63d564a611ea7f8327b1` with uncommitted self-play/action
boundary changes in the working tree. The config ran 300 steps of
`walk_and_shoot` BC, then 50 PPO updates with canonical 60s rounds and
15s/8s objective timing. A longer sibling config,
`experiments/configs/phase4/probe/phase4_mappo_current_selfplay_long.yaml`,
now uses the same BC bootstrap for future 250-update runs.

**Structural finding:** Phase 4 actor observations are team-relative, but
learned Team B movement was entering the simulator without conversion back to
world frame. This made BC move Team A toward objective while Team B moved in
the wrong world direction. Fixed the action boundary so learned actions for
slots 3-5 negate `move_x/move_y` before entering the sim; `aim_delta` is left
unchanged because it is a relative angular delta. The same conversion was
applied to the Phase 11 current/snapshot self-play wrapper, and replay dumping
now writes Phase 4 self-play all-six policy actions in world frame.

**Pre-fix W&B probe:** run
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/yajtva64`. It proved
the BC bootstrap could produce one-sided objective progress but exposed the
action-frame bug: final eval was `0W/0L/20D`, score `0.00/0.00`, kills
`0.0/0.0`, Team A/B hit-fire `0.0000/0.0000`, Team A majority/uncontested
seconds `16.20/16.20`, Team B `0.00/0.00`, cap-progress gain `163.0`.

**Post-fix W&B probe:** run
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/naxqnky7`. Checkpoints
under `python/runs/phase4_mappo_current_selfplay_bc_probe/mappo/`; manifest
selected update 50 as best eval. BC eval after the fix already had combat:
draw-only, score `0.00/0.00`, hit-fire `0.0161/0.0239`. Final update-50 eval:
mean reward `-0.886`, `0W/0L/20D`, score `0.00/0.00`, Team A/B kills
`3.0/1.0`, hit-fire `0.0217/0.0228`, visible-fire `0.956/0.956`, aim error
`1.019/1.089`, majority seconds `1.00/3.90`, uncontested seconds `0.00/3.60`,
cap-progress gain `106.0`. Rollout metrics showed agents reaching point:
`self_on_point_fraction` peaked at `0.555` at update 30 and was `0.196` at
update 50.

**Replay artifact:** stochastic replay
`data/replays/phase4_current_selfplay_bc_probe_ckpt0050_stochastic.replay`
from `ckpt_final.pt`, seed `3519994490`, 600 decisions. Combat analyzer
summary for that replay: final state score `0.00/0.00`, kills `0/1`;
Team A/B fire commands `1797/1796`, visible-fire rates `0.955/1.000`,
damage hits `21/25`, total damage `21000/25000` centi-HP, Team B kill deltas
`1`, mean nearest-visible aim error `1.331/1.484` rad.

**Verification:** focused checks passed:
`py -3.13 -m pytest tests/test_phase4_current_selfplay.py
tests/test_phase4_mappo_env.py tests/test_phase11_current_selfplay.py
tests/test_mappo_loss_mask.py tests/test_train_dispatch.py -q`
(`53 passed`, 2 pytest config warnings). Replay-specific checks passed:
`py -3.13 -m pytest
tests/test_phase4_checkpoint_replay_dump.py::test_dump_replay_supports_phase4_current_selfplay_checkpoint
tests/test_phase4_current_selfplay.py tests/test_phase11_current_selfplay.py -q`
(`14 passed`, 2 pytest config warnings).

**Decision:** self-play plumbing is working and materially improves the Phase 4
diagnostic surface: both teams can reach/contest point and combat now produces
damage/kills. Phase 4 is not cleared: current self-play still produced only
draws and no score conversion by update 50. The next useful run is the
250-update BC self-play long config or a 100-150 update intermediate run,
judged by score/cap conversion, sustained on-point fraction, hit-fire, damage,
kills, and replay inspection rather than self-play win rate.

## 2026-05-19 — Phase 4 self-play long runs and anchor-transfer check

**Reason/configs:** continued the current-vs-current Phase 4 self-play path
with longer W&B runs and explicit gate verification. Base commit before local
changes remains `6c7785f82727d71466cc63d564a611ea7f8327b1`; all results below
also include the uncommitted self-play/action-frame/curriculum working-tree
changes. Primary configs were
`experiments/configs/phase4/probe/phase4_mappo_current_selfplay_long.yaml`
and
`experiments/configs/phase4/probe/phase4_mappo_current_selfplay_curriculum_long.yaml`,
seed `3519994490`.

**Evaluator/replay infrastructure:** extended
`python/scripts/eval_mappo_matrix.py` so a six-agent Phase 4 current-self-play
checkpoint can be evaluated as a three-agent Team A or Team B policy against
scripted anchor bots. Matrix rows now include the full eval-stat dictionary,
including combat/objective diagnostics. Fixed a runtime objective-timing setter
bug where configs using `objective_*_seconds` could later also receive
`objective_*_ticks` from the curriculum scheduler, causing reset-time config
validation failures. Replay artifacts:
`data/replays/phase4_current_selfplay_long_ckpt0225_stochastic.replay` and
`data/replays/phase4_current_selfplay_curriculum_ckpt0225_greedy.replay`.

**Canonical self-play long run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/t9rpy6pe`, config
`phase4_mappo_current_selfplay_long.yaml`, checkpoints under
`python/runs/phase4_mappo_current_selfplay_long/mappo/`. The manifest selected
update 225 as best eval: `0W/0L/50D`, score `0.00/3.47`, kills `2.0/5.0`.
Final update 250 was still nonzero but lower: score `0.00/2.20`, kills
`1.0/6.0`. This was the first credible nonzero self-play score signal, but it
was Team B-heavy and self-play win/loss/draw counts remained draw-only by
construction.

**Anchor transfer for canonical self-play:** matrix artifact
`python/runs/phase4_mappo_current_selfplay_long/mappo/anchor_eval_ckpt0225.json`.
As Team A, `ckpt_final.pt` drew `50/50` vs `noop` with score `0.00/0.00`,
lost `50/50` vs `weak_basic_v2` with score `0.00/5.97`, and lost `50/50` vs
`basic` with score `0.00/29.37`. As Team B, artifact
`anchor_eval_ckpt0225_team_b.json`, it also drew vs `noop` and lost all games
to `weak_basic_v2`/`basic` while scoring zero. A 10-episode checkpoint scan
(`anchor_scan_team_a_10ep.json`) found no saved checkpoint that scored against
`noop`; all had only about `1.4s` uncontested time and zero score.

**Curriculum self-play long run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/78b07kac`, config
`phase4_mappo_current_selfplay_curriculum_long.yaml`, output
`python/runs/phase4_mappo_current_selfplay_curriculum_long_v2/mappo/`.
This added warm-start from `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`,
5s/2s -> 15s/8s objective timing curriculum, and majority-on-point advantage
shaping annealed to zero by update 300. The first attempt
(`dogvblxb`) exposed the setter bug above and is not usable. The restarted run
completed 300 updates. The manifest selected update 225 as best eval:
`0W/0L/50D`, score `6.53/0.00`, kills `0.0/2.0`. The strongest canonical
self-play eval observed in logs was update 200: score `6.90/0.00`, Team A
uncontested `15.50s`, cap-progress gain `252` ticks. Final update 300 had the
curriculum fully annealed back to canonical timing/reward and regressed to
score `0.00/0.00`, Team A/B majority seconds `0.80/11.00`, Team A uncontested
`0.80s`, cap-progress gain `19`.

**Anchor transfer for curriculum self-play:** matrix artifact
`python/runs/phase4_mappo_current_selfplay_curriculum_long_v2/mappo/anchor_eval_key_ckpts_team_a.json`
evaluated checkpoints 200, 225, 275, and 300 against `noop`, `weak_basic_v2`,
and `basic` for 50 episodes each. All four checkpoints drew vs `noop` with
zero score; each had only `1.4s` Team A uncontested time and `12-13`
cap-progress ticks. All four lost every game to `weak_basic_v2` and `basic`.
Against `weak_basic_v2`, Team B scored about `31.4-31.7` while Team A scored
zero, despite Team A logging about `15-16s` majority time and zero uncontested
time.

**Replay inspection:** the curriculum greedy replay from `ckpt_0225.pt` was
scoreless (`0.00/0.00`, no kills). Combat was symmetric and continuous:
Team A/B fire commands `1800/1800`, damage hits `29/29`, total damage
`29000/29000` centi-HP, visible-fire rate `1.000/1.000`, mean aim error
`1.663/1.617` rad. Note: the current replay header does not encode objective
timing overrides, so timing-curriculum replays are useful for action/combat
inspection but not perfect reproduction of non-canonical eval timing.

**Verification:** focused checks passed:
`py -3.13 -m pytest tests/test_phase4_mappo_env.py
tests/test_mappo_team_spirit_ramp.py tests/test_mappo_matrix_eval.py
tests/test_phase4_current_selfplay.py -q` (`55 passed`, 2 pytest config
warnings).

**Decision:** not cleared. The curriculum produced real self-play objective
conversion and is a stronger signal than the canonical self-play run, but the
behavior does not transfer to anchor bots and is not stable after annealing.
The structural failure is now sharper: policies can score in current-vs-current
self-play under some timing/curriculum states, but against anchors they do not
hold point long enough even against `noop`, and against `weak_basic_v2` they
create majority windows without any uncontested time while the bot converts
uncontested control. Next work should not be "just run longer" on plain
current self-play; it should introduce anchor mixing or an explicit hold/capture
behavior objective so the learned policy has to transfer outside the
self-play opponent distribution.

## 2026-05-19 - Phase 4 anchor-mixed current self-play long run

**Reason/config:** tested the simplest anchor-mixed Phase 4 self-play path after
the curriculum run showed non-transferable current-vs-current scoring. Config
`experiments/configs/phase4/probe/phase4_mappo_current_selfplay_anchor_mix_long.yaml`,
seed `3519994490`, base commit `6c7785f82727d71466cc63d564a611ea7f8327b1`
plus the uncommitted Phase 4 self-play/action-frame/curriculum/eval changes.
The run warm-started from
`python/runs/phase4_mappo_current_selfplay_curriculum_long_v2/mappo/ckpt_0200.pt`
and used schedule weights current `0.4`, anchor `0.6`, anchor bot
`weak_basic_v2`, objective timing `10s/5s -> 15s/8s`, and majority advantage
shaping `0.05 -> 0`.

**W&B/run:** W&B
`https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/dvolnkls`, output under
`python/runs/phase4_mappo_current_selfplay_anchor_mix_long/mappo/`. The trainer
completed 300 updates. Manifest:
`best_eval_update_idx=75`, `best_eval_score=2.94e-07`, best eval stats
`0W/34L/16D`, score `0.00/25.07`, kills `4.72/0.64`. Final canonical eval at
update 300 was `0W/34L/16D`, score `0.00/13.08`, kills `5.7/1.0`, Team A/B
majority seconds `15.98/18.56`, uncontested seconds `0.00/18.56`, cap-progress
gain `161.8`.

**Observed self-play trajectory:** anchor mixing reduced Team B score compared
with some earlier checkpoints but never produced a Team A canonical win. The
best-looking Team A scoring point was update 125 canonical eval: `0W/35L/15D`,
score `4.23/15.19`, kills `5.8/1.2`, Team A/B majority seconds `21.36/26.87`,
uncontested seconds `6.63/20.86`. Later checkpoints reduced Team B score to
about `8.88-13.08` but Team A score returned to zero.

**Anchor transfer:** matrix artifact
`python/runs/phase4_mappo_current_selfplay_anchor_mix_long/mappo/anchor_eval_key_ckpts_team_a.json`
evaluated checkpoints 25, 75, 125, 250, and 300 against `noop`,
`weak_basic_v2`, and `basic` for 50 episodes each. Every checkpoint drew
`50/50` vs `noop` with score `0.00/0.00`. Every checkpoint lost `50/50` to
`weak_basic_v2` and `basic` with Team A score `0.00`. Against `weak_basic_v2`,
opponent score improved from `28.13` at update 25 to `21.53` at update 300, but
that is still no gate evidence because Team A never scored or won.

**Replay inspection:** replay artifact
`data/replays/phase4_current_selfplay_anchor_mix_ckpt0125_greedy.replay` from
update 125, seed `3519994490`, 600 decisions. Combat analyzer summary: final
score `0.00/0.00`, kills `2/2`, Team A/B fire commands `1800/1800`, visible
fire rate `0.956/0.956`, damage hits `34/32`, total damage `34000/32000`
centi-HP, mean nearest-visible aim error `1.207/1.269` rad. This replay did
not show stable capture conversion; it looked like symmetric combat pressure
with scoreless objective play.

**Verification:** focused checks passed:
`py -3.13 -m pytest tests/test_phase4_current_selfplay.py
tests/test_mappo_loss_mask.py tests/test_phase4_mappo_env.py
tests/test_mappo_team_spirit_ramp.py tests/test_mappo_matrix_eval.py -q`
(`65 passed`, 2 pytest config warnings).

**Decision:** not cleared. The run is a useful diagnostic success but not a
Phase 4 success. Anchor mixing in this form did not produce transferable
hold/capture behavior; it mostly preserved a Team A kill edge while Team B
still converted uncontested objective time. The next experiment should change
the learning target more directly: add explicit hold/capture pressure or staged
capture drills, and evaluate against anchors before spending another long run
on current-vs-current self-play.

## 2026-05-21 — Phase 4 composition rehearsal v2_2000 (iteration 1, skipped)

**Config:** `../experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v2_2000.yaml`
**Git commit:** `cbf296be5f4d6e9b7f998a11e3f48f032238b869`  **Seed:** `1779134702`
**W&B:** not launched  **Output:** not created
**Gate status:** SKIPPED (prerequisite checkpoint missing)
**Gate reason:** Combat teacher checkpoint
`runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt` is not present on this
machine. The May-18 journal documents this checkpoint being produced at
`13.16/12` mean kills, but the artifact is not in `python/runs/` (or anywhere
under the working tree). The composition rehearsal config cannot launch
without it, and Decision Rules list new teacher checkpoints as "out of scope
without explicit user approval."
**Failing checks (if any):** n/a (run not launched)
**Manifest summary:** n/a
**Anchor transfer:** n/a
**Decision:** falling back to anchor mix v2 (Run 2) per the goal's documented
fallback path; surfacing the missing combat teacher to the user at the end of
the iteration.

## 2026-05-21 — Phase 4 current self-play anchor mix v2 long (iteration 2)

**Config:** `../experiments/configs/phase4/probe/phase4_mappo_current_selfplay_anchor_mix_v2_long.yaml`
**Git commit:** `cbf296be5f4d6e9b7f998a11e3f48f032238b869` (plus working-tree
patch to `python/train/mappo_matrix_eval.py` that fixes a
`require_learner='mappo'` runtime resolution regression — the embedded matrix
eval was passing a config without phase/learner routing, so
`resolve_runtime_env_factory` fell through to legacy phase 2 = `ppo_recurrent`
and raised. Patch injects `learner: {kind: mappo}` and defaults
`env.kind=mappo_match`; verified by `tests/test_mappo_matrix_eval.py`
embedded-path cases passing, by `check_import_boundaries`, and by a smoke
matrix-eval CLI invocation against `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`).
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/zd27z2qv
**Output:** `runs/phase4_mappo_current_selfplay_anchor_mix_v2_long/`
**Gate status:** NOT_CLEARED
**Gate reason:** Objective checks did not meet configured thresholds.
**Failing checks:**
- `selfplay_canonical_score` (`canonical_eval/mean_score_a`): max `0.740` < `6.0`
- `anchor_vs_weak_basic_v2_score` (`matrix/anchor/weak_basic_v2/mean_score_a`):
  max `0.000` < `3.0`
- `anchor_vs_weak_basic_v2_any_win` (`matrix/anchor/weak_basic_v2/wins`):
  max `0` < `5`
- `aim_error_ceiling` (`canonical_eval/team_a_aim_error_rad`): min `2.491` > `1.5`
- Passing: `anchor_vs_noop_no_loss` (`matrix/anchor/noop/loss_rate`)
  = `0.000` <= `0.5`.

**Manifest summary:** `best_eval_update_idx=275`, best-eval stats
`0W/28L/22D`, score `0.00/13.33`, kills `0.0/0.0`. `ckpt_final.pt` aliases
`ckpt_best_eval.pt`. Highest canonical Team A score observed in any
eval across the run was `0.74` at update `125`; final canonical eval at
update `300` was `0.00/16.07`. Self-play eval converted slightly more
(score `0.41` at update `50`, `2.76` at update `175`) but canonical never
exceeded `0.74`.

**Anchor transfer (matrix eval, 50 episodes each, ckpts 125/200/final):**
- vs `noop`: **50/50 wins, Team A score `3.00`** for all three checkpoints
  (`loss_rate=0.00`). First anchor-mix run to actually beat `noop` rather
  than draw; the policy can score when the opponent does not contest.
- vs `weak_basic_v2`: **50/50 losses, Team A score `0.00`, Team B score
  ~`46.5`** for all three checkpoints. Same pattern as the v1
  anchor-mix run.
- vs `basic`: **50/50 losses, Team A score `0.00`, Team B score ~`48.9`**
  for all three checkpoints. Slightly stronger Team B convergence than
  `weak_basic_v2`, consistent with the harder bot.

**Replay artifacts:**
- `data/replays/phase4_current_selfplay_anchor_mix_v2_long_ckpt_final_stochastic.replay`
- `data/replays/phase4_current_selfplay_anchor_mix_v2_long_ckpt_final_greedy.replay`

**Decision:** stopping the loop. Two of the documented Phase 4 probe paths
are now exhausted: Run 1 (composition rehearsal v2_2000) cannot launch on
this machine because the combat teacher checkpoint is missing, and Run 2
(anchor mix v2 long) hit the same falsification trip as the
`curriculum_long_v2` run — canonical Team A score does not reach `6.0` by
update `200`, and matrix transfer to `weak_basic_v2`/`basic` is `0/50`
wins with `0.00` Team A score. Per Decision Rules ("another knob tweak
is unlikely to help") and the Stop Conditions ("If both Run 1 and Run 2
fail, stop the loop and report back"), recommend the user either:
(a) recreate the missing combat teacher checkpoint and re-attempt the
composition rehearsal path; (b) escalate to Strategy 2 (cap-duel) or
Strategy 3 (focus-fire) from the May-18 proposal, both of which require
code changes that are listed as out of scope without explicit user
approval. The noop-only positive transfer (`3.00` Team A score, full
sweep) is a new diagnostic data point: anchor mixing did teach the policy
to convert when the opponent does not contest, but the missing piece is
still combat conversion against a fire-back opponent.

## 2026-05-21 — Phase 4 cap_duel self-play v1 (iteration 1)

**Config:** `../experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v1.yaml`
**Git commit:** `c562fcf7b571b837b167e6195eecf2297fc8c0f9`  **Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/64zvtdgr  **Output:** `runs/phase4_mappo_cap_duel_selfplay_v1/`
**Gate status:** HUMAN_INSPECTION_REQUIRED
**Gate reason:** Objective checks passed; awaiting human subjective review.
**Failing checks (if any):** none. Passing checks: `cap_duel_score=9.02 >= 6.0`, `cap_duel_kills=56.0 >= 5.0`, `cap_duel_wins=32 >= 25`.
**Manifest summary:** `best_eval_update_idx=225`, best eval reward `+4.747`, `28W/11L/11D`, score `7.66/3.76`, mean kills `1.02/0.76`. Highest eval score was `9.02` at update `150`; highest eval wins was `32/50` at updates `50` and `175`.
**Replay artifacts:** `../data/replays/phase4_cap_duel_selfplay_v1_ckpt_final_greedy.replay`, `../data/replays/phase4_cap_duel_selfplay_v1_ckpt_final_stochastic.replay`.
**Gate artifacts:** `runs/phase4_mappo_cap_duel_selfplay_v1/evidence.json`, `runs/phase4_mappo_cap_duel_selfplay_v1/gate_decision.json`.
**Decision:** continuing to Stage 3 transfer probe per GOAL_INSTRUCTIONS.md Stage 2 branching. Subjective replay question remains: approve/reject whether the greedy cap_duel replay shows kill/displace-then-hold behavior rather than just trading fire.

## 2026-05-21 - Phase 4 cap_duel transfer v1 (iteration 1)

**Config:** `../experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml`
**Git commit:** `c562fcf7b571b837b167e6195eecf2297fc8c0f9`  **Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/s02dhwwd  **Output:** `runs/phase4_mappo_cap_duel_transfer_v1/`
**Gate status:** NOT_REACHED
**Gate reason:** Transfer probe was stopped after the update-50 eval per `GOAL_INSTRUCTIONS.md` Stage 3 Decision Rules: "Transfer eval still scores 0 by update 50 -> escalate." The run was intentionally stopped before final checkpoint, replay dump, matrix eval, evidence build, or phase-gate invocation.
**Failing checks (if any):** No gate artifact was produced. The update-50 eval would fail the configured transfer gate on `weak_basic_v2_score` (`eval/mean_score_a=0.00 < 3.0`) and `weak_basic_v2_wins` (`eval/wins=0 < 5`). `hit_fire_floor` would also fail at update 50 (`eval/team_a_hit_fire=0.0145 < 0.04`).
**Manifest summary:** no final manifest. Update 25 eval: `0W/0L/50D`, score `0.00/0.00`, kills `0.0/2.0`, hit_fire `0.0161/0.2111`, majority seconds `0.00/56.00`. Update 50 eval: `0W/0L/50D`, score `0.00/0.00`, kills `0.0/2.0`, hit_fire `0.0145/0.2111`, majority seconds `0.00/56.00`, damage from fire `14.5/211.1`.
**Anchor transfer (Stage 3 only):** not run because the update-50 zero-score escalation rule fired before final checkpoint and matrix eval.
**Replay artifacts:** none for Stage 3; replay dump was not run because the transfer gate was not reached.
**Decision:** stopping this cap_duel transfer loop and reporting back. Stage 2 solved the cap_duel objective gate objectively, but the skill did not survive transfer into the full 3v3 reward gradient against `weak_basic_v2`. Recommended next escalation is Strategy 3 focus-fire target conditioning, or a composition rehearsal that uses the cap_duel teacher as the available combat/objective teacher. Both require explicit approval before additional code-level or training-plan changes.

## 2026-05-21 — Phase 4 cap_duel rollout inspection

**Config:** ../experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v1.yaml
**Checkpoint:** runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt
**Git commit:** `e9d5b5026e51bdefb4d187fd90b9bff5f4b6f202` (plus working-tree
patches: additive `_make_info` diagnostic keys + focused test in
`python/envs/phase4_cap_duel_mappo.py` and
`python/tests/test_phase4_cap_duel_mappo.py`; new
`python/scripts/inspect_cap_duel_rollout.py`; replay-loader bug fixes in
`src/viewer/src/replay_loader.cpp` that propagate `fog`, `cover=`, and
`walls=` into the playback `MatchConfig`).
**Seed:** base `3519994490`, 10 episodes per mode at seeds 3519994490..3519994499.
**W&B:** n/a (diagnostic only, no training launched).
**Output:** runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/
**Gate status:** DIAGNOSTIC
**Gate reason:** Stage A of `GOAL_INSTRUCTIONS.md` — answer the
`HUMAN_INSPECTION_REQUIRED` subjective gate from the cap_duel selfplay
v1 run, which could not be answered from the standard Phase 4 replay
viewer because cap_duel is a Python-only mini-game whose world the
canonical C++ Phase 4 sim cannot reconstruct from the action stream
alone.

**Greedy results:** wins A/B/Draws 9/1/0, mean Team A score 10.70 / 12,
mean Team B score 1.40. Total score events 107; kill_then_hold 107
(`kill_then_hold_ratio=1.000`), displace_then_hold 0, accidental 0.
10 kills, 32 hits, 99 fire decisions. Zero "forbidden" score ticks
(self on point and enemy alive on point simultaneously).
**Stochastic results:** wins A/B/Draws 9/1/0, mean Team A score 10.90,
mean Team B score 1.30. Total score events 109; kill_then_hold 109
(`kill_then_hold_ratio=1.000`), displace_then_hold 0, accidental 0.
12 kills, 36 hits, 71 fire decisions.
**Inspection artifacts:** runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_greedy.json,
runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_stochastic.json.

**Single-seed anomaly observed:** the very first inspector run on seed
`3519994490` alone (1 episode) was an outlier self-play loss — Team B
won, learner died at step 37, 2 hits in 42 decisions, 0 score events.
That seed-alone picture matches the viewer-observed "spinning, random
fire, never approaches point." Aggregating 10 episodes shows the
single seed was unrepresentative; the policy wins 9/10 with deliberate
kill-then-hold scoring.

**Verdict:** the cap_duel selfplay v1 checkpoint actually learned
kill-then-hold. 100% of the 216 aggregated score events across greedy
and stochastic are attributable to enemy-dead-at-score or a kill within
the `enemy_recontest_delay=12` window. None is accidental
displacement. The viewer-observed pathology was a combination of the
canonical Phase 4 sim being the wrong rendering target for cap_duel
replays (cap_duel agents spawn within `point_radius=0.18` of origin in
the cap_duel world; the viewer renders them against canonical Phase 4
spawn positions and the canonical objective location) and the user
having sampled a near-anomalous seed in inspection.

**Decision:** Stage A complete. Recommend Stage B-1 (composition
rehearsal with the cap_duel checkpoint as combat teacher) per
`GOAL_INSTRUCTIONS.md`. Awaiting user approval to launch.

## 2026-05-21 — Phase 4 cap_duel v2 retrain (engineered quirks removed)

**Config:** ../experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v2.yaml
**Git commit:** `e9d5b5026e51bdefb4d187fd90b9bff5f4b6f202` (plus
working-tree env changes: additive `knockback_magnitude`,
`spawn_distance`, `respawn_at_spawn_position` knobs on
`Phase4CapDuelMappoEnv` with v1-preserving defaults; four focused tests;
new v2 yaml).
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/l9890sl6
**Output:** runs/phase4_mappo_cap_duel_selfplay_v2/
**Gate status:** DIAGNOSTIC (cap_duel duel-gate thresholds pass; no
formal phase_gate.cli run yet — this entry documents the retrain
itself; gate invocation is a follow-up).

**Reason:** the prior cap_duel v1 inspection found that the v1 win
mechanism depended on three engineered quirks not present in the
canonical Phase 4 3v3 sim: spawn-on-point, shot knockback (~0.693
units per hit), and respawn-on-point. Sim-side check confirmed
`apply_damage_buffer` in `src/sim/src/internal/sim_combat.cpp` writes
HP only, no position. v2 removes these quirks: agents spawn at
`spawn_distance=0.4` on opposite sides of origin, knockback is 0,
respawn restores each agent to its initial spawn position. Same
warm-start (`phase4_mappo_basic_v6_5`), same training knobs as v1.

**Pre-launch checks:** focused pytest suite (61 passed), import-boundary
check passed, hand-coded "walk to origin + aim + fire" policy wins
20/20 self-play episodes in 15 decisions each → v2 env is solvable.

**Training:** 250 PPO updates after 500-step BC pretrain. Best eval at
update 225: mean_reward `+3.979`, `33W/2L/15D`, score `10.30/2.46`,
kills `1.0/0.4`. Final eval at update 250: `37W/3L/10D`, score
`10.54/1.40`. Both clear the duel-gate thresholds
(`cap_duel_score >= 6.0`, `cap_duel_kills >= 5.0` per-eval total
~50, `cap_duel_wins >= 25`).

**Inspector results (10 episodes per mode):**
- Greedy: 8W/0L/2D, mean A score 9.30, kill_then_hold 81/93 (87.1%),
  accidental 12 (one outlier ep where A held the point against a B
  policy that hovered just outside the point — not a v1-style
  spawn-on-point exploit).
- Stochastic: 10W/0L/0D, mean A score 11.90, kill_then_hold 119/119
  (100%), accidental 0. 15 kills, 47 hits, 218 fires (~21.6% hit rate).
- Combined: 200/212 = **94.3% kill_then_hold** across both modes.

**Artifacts:** runs/phase4_mappo_cap_duel_selfplay_v2/mappo/diagnostics/inspect_{greedy,stochastic}.json,
four cap_duel-native HTML viewers under the same dir.

**Decision:** v2 cap_duel is an honest combat teacher under canonical-
sim-compatible rules. Recommend Stage B-1 (composition rehearsal with
this v2 ckpt as combat teacher) per the v2 hand-off note. Config-only
next move; no code changes required.

## 2026-05-21 — Phase 4 composition rehearsal cap_duel v2 (iteration 1)

**Config:** ../experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2.yaml
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` (plus working-tree
config/test additions for the cap_duel v2 composition-rehearsal config and a
focused rehearsal-env test).
**Seed:** `1779134702`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ao4hy6fa
**Output:** python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2/
**Gate status:** NOT_REACHED (pre-PPO composition kill-switch)
**Gate reason:** the post-composition full 3v3 hit/fire check collapsed below
the required `0.04` floor before PPO could start. The phase gate was not
invoked because the documented pre-PPO kill-switch fired.

**Pre-launch verification:** `py -3.13 -m pytest
tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py
tests/test_phase4_mappo_env.py tests/test_phase4_current_selfplay.py
tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py
tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py -q`
passed (`87 passed`). `py -3.13 -m scripts.check_import_boundaries` passed.
The new focused composition-rehearsal test verifies that the rehearsal combat
env constructs `mini_game: cap_duel`, honors the v2 knobs, produces the
expected actor tensor shape, and runs a finite one-step BC loss.

**Composition pretrain:** ran all 2000 steps. Final logged loss `0.0782`
(`move_loss=0.0000`, `aim_loss=0.0003`, `fire_loss=0.0779`).

**Post-BC diagnostics:** `composition_gate passed=False`; objective on-point
`0.131 < 0.250`, objective losses `50 > 0`, combat kills `4.26 < 12.00`,
full-env Team A hit/fire `0.0047 < 0.0400`, full-env aim error
`1.387 < 1.550`.

**Anchor transfer artifact:** matrix eval still ran after PPO was skipped.
`python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2/mappo/matrix_eval.json`
shows draw-only vs `noop` (`0.00/0.00` score), `50/50` losses vs
`weak_basic_v2` with Team A score `0.00`, and `50/50` losses vs `basic` with
Team A score `0.00`.

**Replay artifacts:** none. PPO was skipped before producing a useful full-3v3
checkpoint for the Stage 1 subjective replay question.

**Decision:** apply the single documented Stage 1 fallback:
`composition_pretrain_steps: 2000 -> 4000`. The cap_duel v2
`mini_game_config` block was verified field-for-field against
`phase4_mappo_cap_duel_selfplay_v2.yaml` before launching the fallback.

## 2026-05-21 — Phase 4 composition rehearsal cap_duel v2 4000-step fallback (iteration 2)

**Config:** ../experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` (plus working-tree
config/test/doc additions for this goal).
**Seed:** `1779134702`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ssmkgzua
**Output:** python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2_4000/
**Gate status:** NOT_REACHED (pre-PPO composition kill-switch)
**Gate reason:** the single allowed 4000-step fallback still failed the
post-composition full 3v3 hit/fire floor, so PPO was skipped and the Phase 4
anchor-transfer phase gate was not invoked.

**Pre-launch verification:** after adding the fallback config,
`py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py
tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py
tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py
tests/test_mappo_composition_rehearsal.py tests/test_mappo_pretrain_hooks.py
tests/test_mappo_team_spirit_ramp.py -q` passed (`88 passed`).
`py -3.13 -m scripts.check_import_boundaries` passed. The configured
`tests/test_phase4_checkpoint_replay_dump.py` path is not present in this
checkout; the current equivalent smoke suite
`py -3.13 -m pytest tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q`
passed (`11 passed`).

**Composition pretrain:** ran all 4000 steps. Final logged loss `0.0550`
(`move_loss=0.0000`, `aim_loss=0.0002`, `fire_loss=0.0548`).

**Post-BC diagnostics:** `composition_gate passed=False`; objective on-point
`0.203 < 0.250`, objective losses `50 > 0`, combat kills `3.04 < 12.00`,
full-env Team A hit/fire `0.0116 < 0.0400`, full-env aim error
`1.555 > 1.550`.

**Anchor transfer artifact:** matrix eval ran after PPO was skipped.
`python/runs/phase4_mappo_composition_rehearsal_cap_duel_v2_4000/mappo/matrix_eval.json`
shows draw-only vs `noop` (`0.00/0.00` score, Team A kills `1.0`), `50/50`
losses vs `weak_basic_v2` with Team A score `0.00`, and `50/50` losses vs
`basic` with Team A score `0.00`.

**Replay artifacts:** none. The run stopped before PPO and before a meaningful
full-3v3 checkpoint for the subjective replay question.

**Decision:** stop the Stage 1 loop. Both the required 2000-step composition
rehearsal and its single allowed 4000-step fallback failed before PPO because
combat did not survive into the full 3v3 post-BC eval. The cap_duel v2 teacher
is honest in isolation, but this composition-rehearsal path did not bind the
teacher's kill-then-hold behavior into the full weak_basic_v2 distribution.
Recommended next move is user decision on a code-scope escalation such as a
distillation anchor during PPO or Strategy 3 focus-fire target conditioning;
another config-only rehearsal-length tweak is outside the documented rules.

## 2026-05-21 — Phase 4 PPO-time cap_duel v2 distillation anchor

**Config:** ../experiments/configs/phase4/probe/phase4_mappo_cap_duel_distill_anchor_v1.yaml
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` (plus working-tree
trainer/test/config additions for the PPO-time cap_duel distillation anchor).
**Seed:** `1779134702`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/c8yyqsdd
**Output:** python/runs/phase4_mappo_cap_duel_distill_anchor_v1/
**Gate status:** NOT_CLEARED
**Gate reason:** objective checks failed. `weak_basic_v2_score=0.0 < 3.0`,
`weak_basic_v2_wins=0 < 5`, `hit_fire_floor=0.0168918919 < 0.04`, and
`anchor_vs_basic_score=0.0 < 1.0`. The noop no-loss check passed.

**Implementation:** added a PPO-time cap_duel distillation anchor that loads the
frozen cap_duel v2 teacher, rolls a cap_duel batch every PPO update, and adds
aim/fire imitation loss to the learner update under new `distill/*` W&B metrics.
No C++, sim rules, reward functions, observation/action spaces, replay format,
existing metric schema, or gate thresholds were changed.

**Pre-launch verification:** `py -3.13 -m pytest
tests/test_phase4_cap_duel_mappo.py tests/test_phase4_mappo_env.py
tests/test_mappo_matrix_eval.py tests/test_mappo_composition_rehearsal.py
tests/test_mappo_pretrain_hooks.py tests/test_mappo_team_spirit_ramp.py
tests/test_cap_duel_distill.py -q` passed (`81 passed`). `py -3.13 -m
scripts.check_import_boundaries` passed. `git diff --check` passed.

**Training:** warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`, teacher
`runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt`. The distillation
anchor was active with finite metrics throughout the run. Final W&B summary
reported `distill/loss=1.2836`, `distill/aim_loss=0.57077`,
`distill/fire_loss=0.71284`, `distill/scaled_loss=0.06418`,
`distill/active_samples=256`, `distill/teacher_fire_prob=0.80877`,
`distill/student_fire_prob=0.48559`, and `distill/fire_agreement=0.19141`.

**Early stop:** the configured update-50 kill rule fired:
`team_a_hit_fire=0.0168918919 < 0.04` and `mean_score_a=0.0 <= 0.0`.
Update-25 eval was also below target (`team_a_hit_fire=0.0165`, score
`0.00/23.33`, `0W/50L/0D`), so the anchor did not produce an early full-3v3
combat transfer signal.

**Matrix transfer artifact:** `python/runs/phase4_mappo_cap_duel_distill_anchor_v1/mappo/matrix_eval.json`
shows draw-only vs `noop` (`0.00/0.00`, `0W/0L/50D`), `50/50` losses vs
`weak_basic_v2` (`0.00/23.33`), and `50/50` losses vs `basic` (`0.00/37.00`,
Team B kills `30.0`).

**Artifacts:** evidence and decision files are under
`python/runs/phase4_mappo_cap_duel_distill_anchor_v1/`:
`evidence.json`, `gate_decision.json`, `launch.log`, plus
`mappo/early_stop_decision.json`, `mappo/checkpoint_manifest.json`, and
`mappo/transfer_summary.{json,md}`. Replay dumps:
`data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_stochastic.replay`.
Viewer command:
`xushi2-viewer --replay data/replays/phase4_cap_duel_distill_anchor_v1_ckpt_final_greedy.replay`.

**Post-run verification:** `py -3.13 -m pytest
tests/smoke/test_phase_checkpoint_replay_dump_smoke.py -q` passed
(`11 passed`). `phase_gate.cli` wrote `gate_decision.json` with status
`NOT_CLEARED`.

**Decision:** stop this PPO-time cap_duel distillation-anchor path. It proved
the teacher hook can run and log cleanly during PPO, but it did not raise
full-3v3 hit/fire above the `0.04` floor or produce any Team A scoring. Per the
goal instructions, the recommended next move is Strategy 3: explicit focus-fire
target conditioning, with user approval before adding any new actor-head or
action-space-facing machinery.

## 2026-05-22 — Phase 4 cap_duel v2 focus-fire transfer probe

**Config:**
`experiments/configs/phase4/probe/phase4_mappo_cap_duel_v2_focus_fire_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/j90y7l1i
**Output:** `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/`
**Gate status:** `NOT_CLEARED`

**Reason:** a 2026-05-22 audit reconciled the latest "Strategy 3" recommendation
with existing focus-fire work. `phase4_mappo_focus_fire_v1` already implemented
the shared low-HP target-conditioning machinery and fixed measured target
concentration, but did not score. `phase4_mappo_cap_duel_focus_fire_v1` used the
older cap-duel v1 checkpoint, so it did not directly test the honest cap-duel v2
teacher. This run tested the one remaining config-only combination: cap-duel v2
checkpoint plus existing focus-fire machinery, with no new actor head or
action-space-facing target field.

**Training:** warm-started from
`runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt` into full 3v3
against `weak_basic_v2`, with `target_conditioned_combat: true`,
`target_selection_aux_mode: team_focus_low_hp`, and total `100` updates. A
subagent-launched first attempt crashed at update 30 due to a closed console
print (`OSError: [Errno 22] Invalid argument`) and produced no eval; it is not
gate evidence. The clean relaunch is W&B `j90y7l1i`.

**Result:** update 50 already matched the falsification rule: `0W/50L/0D`,
score `0.00/35.73`, Team A hit/fire `0.0000`, recent training `onpt=0.000`.
The run continued to update 100 and remained negative: `0W/50L/0D`, score
`0.00/35.93`, Team A hit/fire `0.0017`, same-target fraction `0.817`, focus
entropy `0.350`.

**Matrix transfer:** `python/runs/phase4_mappo_cap_duel_v2_focus_fire_v1/mappo/matrix_eval.json`
shows draw-only vs `noop` (`0.00/0.00`), `50/50` losses vs `weak_basic_v2`
(`0.00/35.73`), and `50/50` losses vs `basic` (`0.00/37.00`, Team B kills
`30.0`).

**Artifacts:** `evidence.json`, `gate_decision.json`, `checkpoint_manifest.json`,
`transfer_summary.{json,md}`, and two replay dumps:
`data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_stochastic.replay`.
Viewer command:
`xushi2-viewer --replay data/replays/phase4_cap_duel_v2_focus_fire_v1_ckpt_final_greedy.replay`.

**Verification:** focused pytest suite passed (`43 passed`),
`scripts.check_import_boundaries` passed, replay dump smoke passed (`11 passed`),
and `phase_gate.cli` wrote `NOT_CLEARED`.

**Decision:** stop this config-only focus-fire path. The honest cap-duel v2
teacher plus existing focus-fire machinery did not recover full-3v3 combat,
scoring, or objective conversion. Recommended next move is a new active design
plan for a different structural intervention; do not retry plain focus-fire,
cap-duel v1 focus-fire, cap-duel v2 focus-fire config-only variants,
composition-rehearsal length variants, or distillation coefficient-only
variants.

## 2026-05-22 — Phase 4 full-env teacher rehearsal implementation

**Status:** implementation ready; probe not yet launched in this entry.

**Reason:** after cap-duel v2 composition, PPO-time distillation, and cap-duel
v2 focus-fire all failed to bind combat into full weak_basic_v2 3v3, a new
active design plan was added:
`docs/plans/active/2026-05-22-phase4-full-env-teacher-rehearsal-design.md`.
The mechanism is an opt-in pre-PPO rehearsal stage that trains movement,
aim/fire, and optional target-selection labels in the actual full Phase 4
environment using only actor-visible observations.

**Implementation:** added `python/train/full_env_rehearsal.py`, wired
`maybe_run_full_env_rehearsal(...)` into the pretrain chain before
composition/BC, and added the first probe config:
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`.
The new gate writes `full_env_rehearsal_gate.json` before PPO. If the gate
returns `NOT_REACHED`, PPO must not launch.

**Scope:** no C++, sim rules, reward formulas, observation/action schemas,
replay format, existing W&B metric schema, or phase-gate thresholds changed.
The scripted rehearsal label function accepts only actor observations and
`MappoConfig`; it does not accept env/sim/critic/info state.

**Verification:** `py -3.13 -m pytest tests/test_full_env_rehearsal.py
tests/test_mappo_pretrain_hooks.py -q` passed (`9 passed`).
`py -3.13 -m pytest tests/test_mappo_pretrain_hooks.py
tests/test_mappo_focus_fire.py tests/test_mappo_aux_aim.py
tests/test_phase7_partial_obs.py -q` passed (`31 passed`). Broader focused
suite with Phase 4 env and matrix eval passed (`70 passed`).
`py -3.13 -m scripts.check_import_boundaries` passed.

**Next:** launch
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`
once with W&B enabled. Treat pre-PPO gate failure as `NOT_REACHED`, not
blocked.

## 2026-05-22 — Phase 4 full-env teacher rehearsal probe

**Config:**
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/xi4zc1cu
**Output:** `python/runs/phase4_mappo_full_env_rehearsal_v1/`
**Pre-PPO gate status:** `NOT_REACHED`

**Training:** warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` and ran the new
full-env scripted rehearsal stage for 2000 supervised steps. The rehearsal
objective converged mechanically: step 1 loss `2.1260` (`move=0.1225`,
`aim=0.1533`, `fire=1.3187`, `target=1.0630`) to step 2000 loss `0.0012`
(`move=0.0007`, `aim=0.0002`, `fire=0.0003`, `target=0.0001`).

**Gate result:** the pre-PPO full-env gate failed with Team A hit/fire
`0.0005556 < 0.04`, objective on-point `0.015 < 0.25`, `50/50` losses,
`0` wins, and score `0.00/37.00`. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/full_env_rehearsal_gate.json`.
Because the gate returned `NOT_REACHED`, PPO was intentionally skipped and no
post-PPO phase-gate decision was produced.

**Artifacts:** checkpoint manifest under
`python/runs/phase4_mappo_full_env_rehearsal_v1/mappo/`, plus replay dumps:
`data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_full_env_rehearsal_v1_ckpt_final_stochastic.replay`.
Viewer command:
`xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v1_ckpt_final_greedy.replay`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-result.md`.

**Verification:** implementation/focused suites passed (`9 passed`,
`31 passed`, `70 passed`), import boundary check passed, and replay dump smoke
passed (`11 passed`).

**Decision:** stop this full-env scripted rehearsal v1 path before PPO. The
teacher labels are learnable by supervised loss, but they do not produce enough
full-env hit/fire or objective pressure to justify PPO. Next work should audit
replay/label quality and design a bounded v2; do not rerun v1, extend length
only, weaken the gate, or force PPO after the failed pre-PPO gate.

## 2026-05-22 — Phase 4 full-env teacher rehearsal v2

**Config:**
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/mbqehspr
**Output:** `python/runs/phase4_mappo_full_env_rehearsal_v2/`
**Pre-PPO gate status:** `NOT_REACHED`

**Audit/fix:** the v1 replay analyzer showed Team A firing continuously at
visible enemies but with very large mean nearest-visible aim error (`2.5406`
rad greedy). The bug was in the scripted rehearsal teacher: actor aim vectors
are `(sin theta, cos theta)`, but the target angle used `atan2(y, x)`. V2 fixes
the teacher to use `atan2(x, y)` for the target vector and adds a regression
test for the convention.

**Training:** warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` and ran the corrected
full-env rehearsal stage for 2000 supervised steps. The objective again
converged: step 1 loss `2.2337` (`move=0.1387`, `aim=0.2181`,
`fire=1.3455`, `target=1.0629`) to step 2000 loss `0.0012`
(`move=0.0007`, `aim=0.0003`, `fire=0.0002`, `target=0.0001`).

**Gate result:** the pre-PPO full-env gate still failed with Team A hit/fire
`0.0 < 0.04`, objective on-point `0.01 < 0.25`, `50/50` losses, `0` wins,
and score `0.00/37.00`. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/full_env_rehearsal_gate.json`.
Because the gate returned `NOT_REACHED`, PPO was intentionally skipped.

**Replay diagnostics:** v2 reduced Team A mean nearest-visible aim error to
`0.6073` rad greedy and `0.6376` rad stochastic, but both replays still had
`0` Team A damage and `0` kills while firing nearly every decision. This
confirms the aim convention bug was real, but not sufficient to make scripted
full-env rehearsal viable.

**Artifacts:** checkpoint manifest under
`python/runs/phase4_mappo_full_env_rehearsal_v2/mappo/`, plus replay dumps:
`data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay` and
`data/replays/phase4_full_env_rehearsal_v2_ckpt_final_stochastic.replay`.
Viewer command:
`xushi2-viewer --replay data/replays/phase4_full_env_rehearsal_v2_ckpt_final_greedy.replay`.
Result doc:
`docs/plans/archive/2026-05-22-phase4-full-env-rehearsal-v2-result.md`.

**Verification:** full-env/pretrain tests passed (`10 passed`), broader
focused suite passed (`71 passed`), import boundary check passed, replay dump
smoke passed (`11 passed`), and replay combat analyzer completed for both v2
replays.

**Decision:** stop the corrected full-env scripted rehearsal v2 path before
PPO. Do not extend v2 length or force PPO. The next design needs a
higher-fidelity full-env teacher that accounts for shooting geometry and
objective timing, or a separate scripted-action diagnostic proving that the
teacher can hit and hold in the same full-env distribution before training a
neural policy against its labels.

## 2026-05-22 — Phase 4 direct full-env teacher diagnostic

**Status:** offline diagnostic complete; not Phase 4 gate evidence.
**Config source:**
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** none; this was not a training run.

**Reason:** v2 fixed the aim convention and improved neural replay aim error,
but still failed the pre-PPO gate. The next question was whether the teacher
itself can create useful full-env behavior before using it as supervised data.

**Implementation:** added `python/scripts/diagnose_full_env_teacher.py` and
`python/tests/test_full_env_teacher_diagnostic.py`. The script can run the
actor-observation-only rehearsal teacher or a C++ scripted bot teacher directly
through `Phase4MappoEnv`, writing JSON summaries without PPO.

**Actor-observation teacher diagnostic:** `actor_obs_scripted` vs
`weak_basic_v2`, 50 episodes, produced `0` wins, `0` losses, `50` draws,
score `0.00/0.00`, Team A hit/fire `0.0`, visible-fire rate `1.0`, and
objective_on_point `0.0`. Artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v2/scripted_teacher_diagnostic.json`.
The same teacher against `noop` won `10/10` with score `37.0/0.0` and
objective_on_point `0.9333`, so basic objective movement works when
uncontested.

**Full-state teacher bound:** `cpp_basic` vs `weak_basic_v2`, 10 episodes,
produced `10` wins, score `12.70/0.00`, Team A hit/fire `0.0917`, visible-fire
rate `1.0`, and objective_on_point `0.8667`. Artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v2/cpp_basic_teacher_diagnostic.json`.

**Interpretation:** the actor-observation-only v2 teacher is not viable as a
training target. It can walk to point, but cannot create contested damage or
majority pressure. The full-state `cpp_basic` contrast proves the full Phase 4
wrapper can support hit/hold behavior, but it uses information not present in
the current flat actor observation. Phase 4 flat actor obs still exposes only a
counterpart enemy through `visible_enemy_1v1`; `cpp_basic` can choose the
nearest visible enemy across all enemy slots.

**Verification:** diagnostic/full-env tests passed (`9 passed`), broader
focused suite passed (`74 passed`), and import boundary check passed.

**Decision:** do not rerun or extend actor-observation full-env rehearsal v1/v2.
Next work should be a bounded v3 design around a higher-fidelity teacher with a
preflight diagnostic: either privileged training-time imitation from
`cpp_basic` with explicit no-inference-leak tests, or an approved move to an
existing wider multi-enemy actor observation path. Do not change sim rules,
reward formulas, replay format, action semantics, or phase-gate thresholds.

## 2026-05-22 — Phase 4 full-env rehearsal v3 cpp_basic preflight

**Status:** implementation ready; W&B training not yet launched.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`

**Implementation:** extended `python/train/full_env_rehearsal.py` with an
opt-in `teacher: cpp_basic` path. This collects C++ scripted `basic` actions
from the current Phase 4 env state as privileged training-time labels for
movement, aim, and fire. The actor inference path remains unchanged and still
consumes actor observations only. The v3 config disables target-selection
conditioning to avoid mixing the prior focus-fire head with full-state
movement/aim/fire labels.

**Preflight diagnostic:** ran
`py -3.13 -m scripts.diagnose_full_env_teacher --config ../experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml --episodes 10 --seed 3519994490 --teacher cpp_basic --output runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/cpp_basic_teacher_diagnostic.json`.
Result: `10W/0L/0D`, score `12.70/0.00`, Team A hit/fire `0.0917`,
objective_on_point `0.8667`.

**Verification:** `py -3.13 -m pytest tests/test_full_env_rehearsal.py
tests/test_full_env_teacher_diagnostic.py tests/test_mappo_pretrain_hooks.py -q`
passed (`16 passed`). Broader focused suite with focus/fire/leak-adjacent,
Phase 4 env, and matrix eval tests passed (`76 passed`). Import boundary check
passed. Config smoke confirmed `obs_dim=31`, `action_dim=6`, and
`target_selection_dim=0`.

**Decision:** ready for one bounded v3 experiment run with W&B enabled. The run
must respect the existing pre-PPO `full_env_rehearsal_gate.json`: if it returns
`NOT_REACHED`, stop and do not force PPO. If it passes and PPO starts, run only
the configured 100 updates and then require normal Phase 4 gate/matrix/replay
evidence. Do not treat the privileged teacher diagnostic itself as gate
evidence.

## 2026-05-22 — Phase 4 full-env rehearsal v3 cpp_basic run

**Config:**
`experiments/configs/phase4/probe/phase4_mappo_full_env_rehearsal_v3_cpp_basic.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/3kfkr7r2
**Output:** `python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/`
**Pre-PPO gate status:** `NOT_REACHED`

**Training:** warm-started from
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` and ran 2000 supervised
full-env rehearsal steps using `teacher: cpp_basic`. The supervised objective
converged from step 1 loss `0.9507` (`move=0.1399`, `aim=0.0953`,
`fire=0.7156`) to step 2000 loss `0.0264` (`move=0.0050`, `aim=0.0211`,
`fire=0.0002`).

**Gate result:** the pre-PPO gate failed with Team A hit/fire
`0.0061111111 < 0.04`, objective_on_point `0.0166666667 < 0.25`, `50/50`
losses, `0` wins, and score `0.00/37.00`. Gate artifact:
`python/runs/phase4_mappo_full_env_rehearsal_v3_cpp_basic/mappo/full_env_rehearsal_gate.json`.
Because the gate returned `NOT_REACHED`, PPO was intentionally skipped.

**Matrix eval:** `ckpt_final.pt` drew `50/50` vs `noop` with score
`0.00/0.00`, lost `50/50` vs `weak_basic_v2` with score `0.00/37.00`, and
lost `50/50` vs `basic` with score `0.00/37.00`.

**Replay diagnostics:** replay dumps were written to
`data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_greedy.replay`
and
`data/replays/phase4_full_env_rehearsal_v3_cpp_basic_ckpt_final_stochastic.replay`.
Greedy analysis produced Team A hit/fire `0.0061111111`, `11000` centi-HP
damage, `0` kills, and score `0.00/37.00`. Stochastic analysis produced Team A
hit/fire `0.0116731518`, `21000` centi-HP damage, `1` kill, and score
`0.00/35.10`.

**Verification:** focused v3 tests passed (`16 passed`), broader focused suite
passed (`76 passed`), import boundary check passed, direct `cpp_basic` teacher
diagnostic passed (`10W/0L/0D`), and replay dump smoke passed (`11 passed`).

**Decision:** stop this v3 privileged full-env rehearsal path before PPO. Do
not retry by only increasing rehearsal length, weakening the gate, or forcing
PPO. The direct `cpp_basic` teacher succeeds, but the unchanged flat actor
observation/policy does not retain enough contested hit/fire or objective
pressure after imitation. Next work should make an explicit actor
information/capacity decision or prove offline that a different no-observation
teacher is representable by the current flat actor input.

## 2026-05-22 — Phase 4 actor information audit

**Status:** audit complete; implementation requires explicit user approval.
**Audit doc:**
`docs/plans/active/2026-05-22-phase4-actor-information-decision.md`

The observation audit reconciled the v1/v2/v3 rehearsal failures with the
direct teacher diagnostic. Phase 4 actor obs is still the 31-float Phase-1
layout, and in 3v3 `src/sim/src/actor_obs.cpp` fills the enemy block only via
`obs_utils::visible_enemy_1v1`. That helper maps each actor to its counterpart
enemy slot. Direct `cpp_basic` succeeds by choosing among all visible enemies;
the current actor input cannot faithfully represent those target switches.

Decision: do not launch another teacher/rehearsal variant without a direct
diagnostic that wins or scores in full `weak_basic_v2`. The recommended next
step is user approval for one bounded opt-in multi-enemy actor-observation
ablation, with strict hidden-enemy leak tests, shape tests, and import-boundary
checks. No reward, sim-rule, action, replay, phase-gate, or existing W&B schema
changes are authorized by this audit.

## 2026-05-22 — Phase 4 multi-enemy actor observation preflight

**Status:** implementation and preflight complete; W&B training not launched.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** none; local diagnostic only.

Implemented an opt-in Phase 4 actor-observation ablation via
`env.actor_obs: multi_enemy_entity_grid`. The default Phase 4 flat actor path is
unchanged. The new wrapper delegates sim, rewards, action semantics, critic obs,
and episode info to `Phase4MappoEnv`, and only transforms actor observations
into current visible enemy tokens. It uses native C++ line-of-sight visibility
and zeroes masked enemy token payloads; it does not use Phase 7 last-seen stale
enemy markers.

The direct widened-observation teacher diagnostic passed against
`weak_basic_v2`: `10W/0L/0D`, score `9.20/0.00`, Team A hit/fire `0.09`,
visible-fire rate `1.0`, and objective_on_point `0.875`. Artifact:
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/multi_enemy_visible_teacher_diagnostic.json`.

Verification passed: new ablation/diagnostic tests (`11 passed`), import
boundary PASS, Phase 5/6/7 observation suite (`12 passed`), Phase 4
env/pretrain/focus suite (`38 passed`), and C++ actor leak/actor obs/critic
obs/obs dims binaries (`5`, `12`, `8`, and `3` tests passed).

Decision: ready for one separate bounded W&B training assignment using the new
config. Do not treat the direct diagnostic as phase-gate evidence, and do not
retry no-observation teacher variants before this ablation is tested with a
trained policy.

## 2026-05-22 — Phase 4 multi-enemy actor observation training attempt

**Status:** `BLOCKED`; run crashed before usable metrics.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/bw9jsxte

The assigned worker passed preflight (`scripts.check_import_boundaries` and
the multi-enemy actor-observation/direct-diagnostic tests, `11 passed`) and
launched the W&B-enabled training command from `python/`. W&B authenticated and
created the run.

Training failed before eval or PPO metrics. The configured warm start
`runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt` is a flat-actor checkpoint,
while this ablation uses the `entity_attention_grid` actor topology. Strict
`load_state_dict` failed with missing `actor_entity_encoder`,
`actor_grid_encoder`, and `actor_fusion` keys and unexpected `actor_embed`
keys.

Artifacts verified:
`python/wandb/run-20260522_135640-bw9jsxte/files/wandb-metadata.json` and
`python/wandb/run-20260522_135640-bw9jsxte/files/output.log`. No checkpoint
manifest, gate artifact, matrix eval, replay artifact, or training metrics
were produced by this run.

**Decision:** do not retry the same config unchanged. Next work is a bounded
implementation/preflight fix for an explicit opt-in warm-start migration across
the actor observation topology change. This should preserve strict default
warm-start behavior and must not change rewards, sim rules, action semantics,
replay format, phase-gate thresholds, or existing W&B metric schema.

## 2026-05-22 — Phase 4 multi-enemy actor observation warm-start fix

**Status:** implementation and preflight complete; W&B training not launched.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`

Added explicit `run.warm_start_migration: compatible_exact` support. Default
warm-start remains strict. The opt-in migration loads only same-name,
same-shape tensors, reports missing model keys, unexpected checkpoint keys, and
same-name shape mismatches, and leaves incompatible flat actor encoder keys
unloaded.

The probe config now opts into this migration. A non-training smoke loaded the
flat Phase 4 checkpoint into the `entity_attention_grid` model without error:
`actor_obs=multi_enemy_entity_grid`, `obs_dim=3167`, `action_dim=6`,
`target_selection_dim=0`, `warm_start_migration=compatible_exact`, with 17
compatible tensors loaded and `actor_embed.*` skipped as unexpected.

Verification passed: import boundary PASS, multi-enemy actor-observation and
diagnostic tests `11 passed`, warm-start hook tests `8 passed`.

**Decision:** ready for one separate bounded W&B training assignment using the
same config. This implementation smoke is not phase-gate evidence.

## 2026-05-22 — Phase 4 multi-enemy actor observation training run

**Status:** `NOT_CLEARED`.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/ud4c09jw
**Output:** `python/runs/phase4_mappo_multi_enemy_actor_obs_v1/`

The bounded W&B run completed 100/100 updates with
`actor_obs=multi_enemy_entity_grid` and
`warm_start_migration=compatible_exact`. Preflight passed: import boundary
PASS, multi-enemy actor-observation and diagnostic tests `11 passed`, and
warm-start hook tests `8 passed`.

Best eval at update 50 and final eval at update 100 were both negative:
`0W/50L/0D`, score `0.00/37.00`, mean reward `-11.0`, and Team A hit/fire
`0.0`. Final objective focus fraction was `0.0`.

Matrix eval:

- vs `noop`: `0W/0L/50D`, score `0.00/0.00`
- vs `weak_basic_v2`: `0W/50L/0D`, score `0.00/37.00`
- vs `basic`: `0W/50L/0D`, score `0.00/37.00`

Transfer summary gate status was `evidence_insufficient`. Artifacts:
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/checkpoint_manifest.json`,
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/matrix_eval.json`,
`python/runs/phase4_mappo_multi_enemy_actor_obs_v1/mappo/transfer_summary.json`,
and replays
`data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_greedy.replay` and
`data/replays/phase4_multi_enemy_actor_obs_v1_ckpt_final_stochastic.replay`.

Replay analyzer found the greedy policy issued no Team A fire commands and did
no damage. The stochastic replay fired continuously but produced only `1000`
centi-HP Team A damage over five detected episodes, Team A hit/fire
`0.0002281022`, and Team B score `37.0`.

**Decision:** objective checks did not pass, so no human replay inspection is
required for clearance. Do not retry this same config unchanged. The direct
multi-enemy-visible scripted teacher succeeds, but the neural PPO run did not
learn objective pressure, firing, or scoring from the widened actor observation
alone.

## 2026-05-22 — Phase 4 multi-enemy training failure audit

**Status:** audit/design complete; implementation requires explicit user
approval.
**Audit doc:**
`docs/plans/active/2026-05-22-phase4-multi-enemy-training-failure-audit.md`

The audit reconciled the positive direct teacher diagnostic with the failed
PPO run. The widened actor observation is sufficient for scripted direct action
selection: `multi_enemy_visible` beat `weak_basic_v2` `10W/0L/0D`, score
`9.20/0.00`, Team A hit/fire `0.09`. The neural run did not inherit a useful
mapping because the compatible warm-start intentionally skipped the old flat
actor encoder; the new `actor_entity_encoder`, `actor_grid_encoder`, and
`actor_fusion` began effectively random. With sparse contested PPO at
`1.0e-6` for 100 updates, the policy stayed high-entropy and greedy Team A
never fired.

Decision: do not retry
`phase4_mappo_multi_enemy_actor_obs_v1.yaml` unchanged and do not run a longer
or coefficient-only PPO variant as the next step. The proposed next step is one
bounded opt-in supervised bridge using existing `multi_enemy_visible` teacher
labels before PPO, with a pre-PPO neural-policy gate that must show nonzero
Team A firing, useful hit/fire, objective pressure, and nonzero score before
any PPO updates. This is a new supervised training path and requires explicit
user approval before implementation.

## 2026-05-22 — Phase 4 multi-enemy supervised bridge

**Status:** `NOT_REACHED`; stop before PPO.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** none; this was a local implementation/preflight and direct gate
assignment.
**Output:** `python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/`

Implemented an opt-in `run.multi_enemy_supervised_bridge` path for
`actor_obs: multi_enemy_entity_grid`, using the existing
`multi_enemy_visible` teacher labels for movement, aim, and fire before PPO.
The bridge is off by default, requires the entity-grid actor observation, and
does not add action-space target fields. No reward, sim-rule, action,
replay-format, phase-gate-threshold, or existing W&B schema changes were made.

The no-W&B direct gate ran 2000 supervised steps. Labels converged to loss
`0.0005422` (`move=0.0000411`, `aim=0.0003568`, `fire=0.0001443`), but the
pre-PPO neural policy gate failed:

- Team A visible fire rate `1.0 >= 0.01`
- Team A hit/fire `0.0088888889 < 0.04`
- objective_on_point `0.0116666667 < 0.25`
- mean score A `0.0 < 1.0`
- losses `50 > 49`
- wins `0`, mean score B `31.9666666667`

Artifacts:
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/multi_enemy_supervised_bridge_summary.json`,
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/multi_enemy_supervised_bridge_gate.json`,
and
`python/runs/phase4_mappo_multi_enemy_supervised_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`.

Verification passed: import boundary PASS, bridge/full-env/pretrain focused
tests `29 passed`, and assignment-required focused suite `22 passed`.

**Decision:** do not launch PPO or W&B training from this result. Do not retry
the same supervised bridge config unchanged, do not increase bridge length as
the next move, and do not force PPO past the failed pre-PPO gate. The next
move should be an offline failure audit/design decision.

## 2026-05-22 — Phase 4 multi-enemy supervised bridge failure audit

**Status:** audit/design complete; next implementation requires explicit user
approval.
**Audit doc:**
`docs/plans/active/2026-05-22-phase4-multi-enemy-supervised-bridge-failure-audit.md`

The audit reconciled the positive direct `multi_enemy_visible` teacher, the
failed widened-observation PPO run, and the failed one-shot supervised bridge.
The actor-visible information surface is sufficient for scripted direct action
selection: the direct teacher beat `weak_basic_v2` `10W/0L/0D`, score
`9.20/0.00`, Team A hit/fire `0.09`, and objective_on_point `0.875`. The
plain PPO run failed because the new actor front end began effectively random
after compatible warm start. The supervised bridge fixed the "no firing"
failure mode on its training distribution, but the closed-loop gate still lost
`50/50`, scored `0.0`, produced Team A hit/fire `0.0088888889`, and produced
objective_on_point `0.0116666667`.

Decision: treat the one-shot bridge as expert-state behavior cloning that
failed under policy-state distribution shift. Do not launch PPO from
`phase4_mappo_multi_enemy_supervised_bridge_v1.yaml`, do not retry the same
bridge unchanged, do not increase bridge length as the next move, and do not
force PPO past the failed gate. The proposed next step is one bounded opt-in
closed-loop supervised bridge using policy-induced states and the existing
`multi_enemy_visible` teacher labels. It must report movement/aim/fire
agreement diagnostics on policy-induced states and pass the same pre-PPO
neural-policy gate before any PPO or W&B run. This is a new supervised training
path and requires explicit user approval before implementation.

## 2026-05-22 — Phase 4 multi-enemy closed-loop supervised bridge

**Status:** `NOT_REACHED`; stop before PPO.
**Config:**
`experiments/configs/phase4/probe/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1.yaml`
**Git commit:** `f776104eb95f64bea44975f0050af29f595f46af` plus dirty
working-tree Phase 4 changes.
**Seed:** `3519994490`
**W&B:** none; this was a local implementation/preflight and pre-PPO gate
assignment.
**Output:**
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/`

Implemented an opt-in closed-loop supervised bridge that rolls out the current
neural policy against `weak_basic_v2`, queries existing `multi_enemy_visible`
teacher labels on policy-induced states, and runs bounded supervised update
rounds. It writes policy-state movement/aim/fire agreement diagnostics. The
feature is opt-in/off-by-default. PPO and W&B were not launched. No reward,
sim-rule, tick-pipeline, action semantics, action-space field, replay format,
phase-gate threshold, or existing W&B metric/schema changes were made.

Final policy-state agreement: movement MSE `0.0154950926`, aim absolute error
`0.2026730925`, fire accuracy `1.0`, fire positive recall `1.0`, policy fire
rate `1.0`, and teacher fire rate `1.0`.

The pre-PPO gate improved over the one-shot bridge but still failed score:
Team A visible-fire rate `1.0`, Team A hit/fire `0.0427777778 >= 0.04`,
objective_on_point `0.29 >= 0.25`, losses `0 <= 49`, wins `0`, mean score B
`0.0`, but mean score A `0.0 < 1.0`. Gate artifact:
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_gate.json`.
Agreement artifact:
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/multi_enemy_closed_loop_supervised_bridge_agreement.json`.
Checkpoint:
`python/runs/phase4_mappo_multi_enemy_closed_loop_supervised_bridge_v1/mappo/ckpt_multi_enemy_closed_loop_supervised_bridge.pt`.

Verification passed: import boundary PASS, multi-enemy/pretrain focused suite
`20 passed`, and full-env/pretrain focused suite `22 passed`.

**Decision:** do not launch PPO or W&B training from this result. Do not retry
the same closed-loop config unchanged and do not force PPO past the failed
score check. Next work should be an offline zero-score draw audit before any
new implementation or W&B assignment.

## 2026-06-09 — Phase 4 objective-conversion reward terms + conversion_v1 setup

**Status:** implementation + smoke complete; the long run is queued for the
user's machine. Strategy doc:
`docs/reports/2026-06-09-phase4-breakthrough-analysis.md`. Runbook:
`docs/reports/2026-06-09-phase4-conversion-runbook.md`.

**Diagnosis behind this change:** a full review of the journal, the objective
state machine (`src/sim/src/internal/sim_objective.cpp`), and
`python/xushi2/reward.py` found that no reward term has ever pointed inside the
kill -> hold -> capture(240 uncontested ticks) -> own -> score chain that gates
all Phase 4 scoring. The 2026-05-22 closed-loop bridge checkpoint already wins
the fight against weak_basic_v2 (0/50 losses, 13.4/0 kills, 4.9s uncontested,
238 cap-progress gain ticks vs 197 loss) and fails only the final hold. Kills
pay instantly while holding pays nothing for ~80 decisions, so the gradient
actively teaches leaving the point. Related: the 2026-05-19 easy-timing
falsification used the old flat-obs spray warm start (0.0s uncontested); it
does not apply to the bridge checkpoint.

**Implementation (additive, config-gated, off by default):**
`RewardCalculator` gains `cap_progress_potential_coef` (potential-based shaping
with `Phi_A = owner_sign + cap_sign * cap_progress`, read from Team A's actor
obs via the existing `ObsAccessor`; PBRS, so policy-invariant and anneal-free)
and `capture_completed_bonus` (one-time team bonus on objective ownership
flip). Conversion metrics (`captures_a/b`, `conversion_phi_a`, per-step
contributions) flow into `info["reward_metrics"]`. No C++, sim-rule,
obs/action, replay, or existing-metric changes. Verification: 7 new unit tests
plus a real-sim integration test (noop opponent, scripted walk-on-point ->
capture completes and pays); `tests/test_reward.py` +
`tests/test_phase4_mappo_env.py` + selfplay/team-spirit suites all pass
(113 passed); ruff delta vs HEAD is zero.

**New configs:**
`experiments/configs/phase4/probe/phase4_mappo_conversion_v1.yaml` (300
updates: bridge-checkpoint warm start from the new tracked copy
`data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt`, timing
curriculum 5s/2s -> 15s/8s over 150 updates, `uncontested_on_point_coef 0.15`,
`cap_progress_potential_coef 1.0`, `capture_completed_bonus 2.0`,
`shaping_clip 30` — the default 3.0 cumulative clip has been silently
saturating every Phase 4 run — kill/death 0.5, LR 1e-5, per-action entropy
move 0.01 / aim 0.03 / binary 0.02) and
`experiments/configs/phase4/smoke/phase4_mappo_conversion_smoke.yaml`.

**Smoke result (4 updates, Windows):** the pipeline works end to end
(warm start loads, curriculum schedules, rewards finite, eval + checkpoints
written). More importantly, the update-2 eval under eased 10s/5s timing was
**4/4 WINS vs weak_basic_v2, score 3.10/2.93, kills 10.0/0.0** — the first
Team A wins against an objective-contesting bot in the project's history,
directly confirming that conversion opportunity, not combat, was the missing
piece. The update-4 eval regressed to 0/4 because the smoke anneals to
canonical 15s/8s in just 4 updates; the real config takes 150.

**Decision:** run `phase4_mappo_conversion_v1.yaml` next (from repo root).
Judge update 25-150 evals on conversion precursors (uncontested seconds,
cap-progress retention, captures/score under eased timing), and the gate on
canonical-eval score after the anneal completes. Escalation ladder and abort
criteria are in the runbook.

## 2026-06-10 — Phase 4 conversion PPO diagnostics + objective-conversion bridge

**Status:** `NOT_REACHED`; PPO diagnostics and conversion bridge preflight are
negative evidence, not blockers.

**Git commit:** `5e25a680bdb7c60cc48ed3f6e35c3b25d85e6bd0` plus the current
working-tree bridge/config changes.
**Seed:** `3519994490`.
**W&B:** disabled / unavailable; no run URL.

### conversion_v1 no-W&B diagnostic

Config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v1.yaml`.
Output:
`runs/phase4_mappo_conversion_v1/`.

The full `conversion_v1` run was launched with W&B disabled and intentionally
stopped at update 50 because it met the runbook's early-stop condition.

- update 25 eval: `0W/50L/0D`, score `0.00/15.63`, kills `12.0/0.0`,
  Team A hit/fire `0.0456`, Team A uncontested `0.50s`, cap gain `57`.
- update 50 eval: `0W/0L/50D`, score `0.00/0.00`, kills `1.0/0.0`,
  Team A hit/fire `0.0228`, Team A uncontested `0.50s`, cap gain `13`.
- Checkpoints: `runs/phase4_mappo_conversion_v1/mappo/ckpt_0025.pt` and
  `runs/phase4_mappo_conversion_v1/mappo/ckpt_0050.pt`.

**Decision:** the `1e-5` PPO run erased the bridge behavior before conversion
could stabilize. Do not continue this exact run to 300 updates.

### conversion_v1 LR 3e-6 diagnostic

New config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v1_lr3e6.yaml`.
Output:
`runs/phase4_mappo_conversion_v1_lr3e6/`.

This was a config-only diagnostic derived from `conversion_v1`: W&B disabled,
learning rate `1.0e-5 -> 3.0e-6`, eval/checkpoint cadence `25 -> 10`, separate
output directory, same seed, reward, timing curriculum, warm start, model, and
opponent. It was also stopped at update 50.

- best intermediate signal at update 30: `0W/0L/50D`, score `0.00/0.00`,
  kills `20.0/0.0`, Team A hit/fire `0.0700`, Team A majority `38.60s`,
  Team A uncontested `0.70s`, cap gain `97`.
- update 50 regressed: `0W/50L/0D`, score `0.00/25.00`, kills `5.0/0.0`,
  Team A hit/fire `0.0333`, Team A majority `8.50s`, Team A uncontested
  `0.50s`, cap gain `57`.
- Checkpoints: `runs/phase4_mappo_conversion_v1_lr3e6/mappo/ckpt_0010.pt`
  through `ckpt_0050.pt`.

**Decision:** lowering LR preserved combat/majority briefly but did not teach
the hold/conversion step. Do not spend more scalar PPO-tuning runs on this
exact bridge checkpoint.

### objective-conversion bridge preflight

Implemented an opt-in closed-loop objective-conversion bridge under the existing
`run.multi_enemy_supervised_bridge` path:

- New teacher mode `multi_enemy_conversion_hold` remains actor-observation-side
  and action-space-compatible.
- Closed-loop bridge samples policy-induced states against `weak_basic_v2`.
- Conversion-relevant actor-visible states (self on point, near objective, or
  cap progress visible in the objective token) receive supervised sample
  weighting.
- The pre-PPO gate now writes explicit conversion diagnostics from the existing
  evaluator: Team A/B uncontested seconds, majority seconds, cap-progress
  gain/loss, first alive-edge-to-score seconds, and majority-to-uncontested
  transition fraction.
- No sim-rule, tick-pipeline, reward, observation/action schema, replay-format,
  phase-gate threshold, or existing W&B schema changes were made.

Config:
`experiments/configs/phase4/probe/phase4_mappo_objective_conversion_bridge_v1.yaml`.
Output:
`runs/phase4_mappo_objective_conversion_bridge_v1/`.
Gate artifact:
`runs/phase4_mappo_objective_conversion_bridge_v1/mappo/objective_conversion_bridge_gate.json`.
Checkpoint:
`runs/phase4_mappo_objective_conversion_bridge_v1/mappo/ckpt_multi_enemy_supervised_bridge.pt`.

The bridge completed 20 closed-loop rounds with finite supervised losses and
conversion-state sample fractions around `0.69`. The pre-PPO gate failed and
correctly skipped PPO:

- status `NOT_REACHED`
- Team A visible-fire rate `1.0 >= 0.01`
- Team A hit/fire `0.0533333333 >= 0.04`
- objective_on_point `0.525 >= 0.25`
- mean score A/B `0.0/0.0333333333`
- losses `50 > 49`
- Team A/B majority seconds `31.5/13.3`
- Team A/B uncontested seconds `0.5/8.5`
- Team A cap-progress gain/loss `238/238`
- first Team A alive-edge-to-score seconds `-1.0`
- majority-to-uncontested-within-window fraction A `0.0`

Post-training matrix artifacts were written, but the gate status is
`evidence_insufficient`: vs `noop` the final checkpoint drew `50/50` with
score `0.00/0.00`; vs `weak_basic_v2` it lost `50/50`, score `0.00/0.03`,
kills `10.0/0.0`; vs `basic` it lost `50/50`, score `0.00/26.20`.

Verification:

- `cd python && .venv/bin/python -m pytest tests/test_full_env_rehearsal.py tests/test_phase4_multi_enemy_actor_obs.py -q`
  -> `22 passed`.
- `cd python && .venv/bin/python -m scripts.check_import_boundaries` -> PASS.
- `git diff --check` -> clean.

**Decision:** the sharper conversion bridge did not solve Phase 4. The learned
policy can fight and can accumulate almost exactly one full capture's worth of
gross progress, but it loses all of it (`cap_gain=238`, `cap_loss=238`) and
never produces a Team A uncontested window beyond `0.5s`. The missing behavior
is now narrower: not reach, fire, hit, or majority presence, but finish and
retain uncontested control after capture progress starts. The next design move
should focus directly on conversion retention, not more PPO scalar tuning or
generic movement/aim/fire cloning.

## 2026-07-09 — Phase 4 plateau review: respawn treadmill + warm-start stabilizers

**Status:** review + implementation + Step 0 diagnostic complete; the
`conversion_v2_respawn` long run is ready to launch on the user's machine.
Review doc: `docs/reports/2026-07-09-phase4-3v3-review-recommendations.md`.
Branch: `claude/3v3-rl-training-review-rpir5f`.

### Bug found: the multi-enemy wrapper silently dropped every runtime setter

`Phase4MultiEnemyMappoEnv` never delegated `set_team_spirit`,
`set_majority_on_point_alpha`, `set_uncontested_on_point_alpha`, or
`set_objective_timing_seconds` to its wrapped `Phase4MappoEnv`, and the sync
vector env discovers setters with `getattr(..., None)` and silently skips
envs that lack them. **Every multi-enemy run to date trained with the
objective-timing anneal, team_spirit ramp, and eval alpha/timing overrides
silently dropped.** Concretely: the 2026-06-10 `conversion_v1` and
`conversion_v1_lr3e6` runs never annealed timing (envs stayed at the yaml
base 5s/2s), their "canonical" evals actually ran at 5s/2s, and eval
`mean_reward` included the uncontested shaping that evaluation is supposed to
zero out. The eased-timing interpretation of those runs is therefore wrong —
they failed at *eased* timing, which strengthens the "PPO destroys the warm
start" diagnosis. Fixed by explicit delegation; a new regression test fails
if `Phase4MappoEnv` ever grows a `set_*` runtime setter the wrapper doesn't
forward.

### Step 0 diagnostic: the respawn treadmill hypothesis is CONFIRMED

`mechanics.respawn_ticks: 240` (8s, equal to the capture requirement) has
been identical in every config in the project's history. With three enemies
respawning every 8s and walking back, killing them one at a time never opens
an 8s uncontested window — the June bridge checkpoint's
`cap_gain=238/cap_loss=238, uncontested=0.5s` signature.

New script `python/scripts/respawn_ablation_eval.py` evaluates a checkpoint
across `respawn_ticks` values with no training. Result for the tracked bridge
checkpoint (`data/checkpoints/phase4_multi_enemy_closed_loop_bridge_v1.pt`)
vs `weak_basic_v2` at CANONICAL objective timing (15s unlock / 8s capture —
the checkpoint env cfg carries no timing overrides):

- respawn 240t (canonical): `0W/0L/50D, score 0.00/0.00, uncontested 3.1s`
  — the historical scoreless draw, reproduced.
- respawn 720t (24s): `0W/50L/0D, score 2.20/7.03, uncontested 14.5/16.2s,
  cap_gain 570` — the checkpoint SCORES for the first time at canonical
  timing with respawns on, but weak_basic_v2 out-converts it and wins. The
  mid-curriculum regime is a live contest, exactly the gradient PPO needs.
- respawn 2400t (no respawn inside a 60s round): **50/50 WINS, score
  2.70/0.00, uncontested 15.0s, kills 8/0** — a clean sweep from the same
  checkpoint that has never won a canonical eval episode.

Artifact: `runs/respawn_ablation/respawn_ablation.json` (50 episodes/cell,
seed 0x9E5F0A47). The progression 240→720→2400 is monotone in Team A score,
so the annealed curriculum has signal at every stage rather than a cliff.

The blocking factor was never combat, aim, observation, or imitation quality
— at canonical rules minus respawn pressure, the June checkpoint already
scores. The lever is a respawn curriculum, which is config-side mechanics in
the same class as the (already-allowed) objective-timing curriculum.

### Implemented (additive, config-gated, off by default)

1. **Respawn curriculum** — `env.respawn_curriculum: {enabled, initial_ticks,
   final_ticks, anneal_updates}`; linear anneal pushed per update via new
   `set_respawn_ticks` plumbing (env → wrapper → sync/async vector env →
   trainer → hooks). Reset-time-only application (no live-sim setter;
   respawn_tick is stamped at death). `info["respawn_ticks"]` reports the
   value the running episode was built with; W&B gets `train/respawn_ticks`.
   Canonical eval (every `eval_canonical_every`) runs at `final_ticks` +
   15s/8s timing, so the transfer signal is always visible.
2. **Critic warmup** — `ppo.critic_warmup_updates: N`: for updates 1..N only
   the value loss is optimized; actor/trunk receive zero gradient. Rationale:
   warm starts pair a competent policy with a critic that is random or fit to
   a different reward scheme, so early advantages are noise and PPO's first
   steps destroy the policy — the empirical smoke showed value_loss 21.9 →
   1.5 after a single warmup update, confirming how wrong the cold critic was.
3. **Anchor KL** — `ppo.anchor_kl_coef` + `ppo.anchor_kl_anneal_updates`:
   at PPO start (after warm start + any pretrain stage) the trainer freezes a
   copy of the policy and adds `coef * KL(pi_current || pi_anchor)` on rollout
   states, annealed linearly to zero. Analytic KL over the action heads
   (tanh-squashed Gaussian KL is exact via transform invariance; clamped
   Bernoulli; optional categorical). This is the missing trust region behind
   the 1e-6-freezes / 2e-6-collapses dichotomy documented across ~10 runs.
   Metrics: `train/anchor_kl`, `train/anchor_kl_coef` (emitted only when
   configured, so legacy W&B schemas are unchanged).

### The combined run (queued)

`experiments/configs/phase4/probe/phase4_mappo_conversion_v2_respawn.yaml`:
bridge warm start + multi-enemy obs + conversion_v1 rewards with
`kill_bonus: 0.1` (a kill must not locally out-pay holding; PBRS pays ~0.004
per progress tick, so 0.5 was ~6x the hold gradient for a 2s chase) +
respawn curriculum 2400→240 over 200 updates + timing curriculum 5s/2s→15s/8s
over 150 + `critic_warmup_updates: 25` + `anchor_kl_coef: 1.0` annealed over
250 + LR 1e-5, 500 updates. Launch from repo root:

    python/.venv/bin/xushi2-train --config experiments/configs/phase4/probe/phase4_mappo_conversion_v2_respawn.yaml

Judge on the funnel (kill edge → wipes → uncontested seconds → captures →
score), expect nothing from eval during updates 1-25 (warmup), and use the
canonical eval rows as the transfer signal. Falsification criteria are in the
config metadata.

### Verification

3-update pipeline smoke (`experiments/configs/phase4/smoke/
phase4_mappo_conversion_v2_respawn_smoke.yaml`, W&B disabled): warm start
loads, anchor freezes, update 1 shows `policy_loss=0.000`,
actor/trunk grad norms exactly 0 and critic grad only; updates 2-3 train the
actor with the anchor term; `obj_t` anneals 10/5 → 15/8 in the training envs
(the wrapper fix working); eval keeps the bridge kill edge (14/0 kills,
hit/fire 0.058). Tests: new `tests/test_phase4_respawn_curriculum.py` (13) +
`tests/test_mappo_critic_warmup_anchor.py` (13) plus focused suites
`test_mappo_team_spirit_ramp` / `test_phase4_mappo_env` /
`test_phase4_multi_enemy_actor_obs` / `test_mappo_public_api` /
`test_mappo_aux_aim` / `test_cap_duel_distill` / `test_reward` /
`test_full_env_rehearsal` — 172 passed total. Import boundary check PASS.
Ruff delta vs HEAD is zero on touched files.

**Behavior/schema changes to flag:** the wrapper setter fix changes
multi-enemy training behavior (curriculum anneals now actually apply — this
is the bug fix, not a new mechanic); additive W&B metrics
`train/respawn_ticks`, `train/anchor_kl`, `train/anchor_kl_coef`,
`train/critic_warmup_active` and additive env info key `respawn_ticks`; no
sim-rule, reward-formula, action, observation, or replay-format changes.

## 2026-07-29 — conversion_v2_respawn post-hoc: first canonical score, two eval bugs, noop blindspot

**Status:** analysis of the 2026-07-11 `conversion_v2_respawn` run (which was
never journaled). Full report:
`docs/reports/2026-07-29-conversion-v2-posthoc-eval.md`. Config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v2_respawn.yaml`,
seed_base 3519994490, artifacts `runs/phase4_mappo_conversion_v2_respawn/`.

### Headline

`ckpt_0100` (= `ckpt_best_eval` by state-dict hash) is the **first checkpoint
ever to beat `weak_basic_v2` with nonzero objective score at canonical
settings** (15s/8s/240t): 50/0/0, score 1.30, ~10s uncontested, one full
capture per round. The June bridge baseline at the same settings: 0-0 draw,
cap 284 gained / 246 lost, score 0.00. The July 9 respawn-treadmill +
critic-warmup + KL-anchor recipe is validated — but only checkpoints 100
(1.30), 275 (0.97) and 350 (0.10) score at canonical; by the anneal tail the
policy reverts to kill-chasing (16–21 kills, conversion dead). The 200-update
respawn anneal was too fast to consolidate defend-and-hold.

### Eval bug 1: checkpoints bake eased sim settings; all post-training evals inherited them

`ckpt["config"]["env"]["sim"]` holds curriculum *initial* values (unlock 5s,
capture 2s, respawn 2400t). Post-training matrix eval, eval_mappo_matrix.py,
and dump_replay.py rebuild envs from checkpoint config → the run's
transfer_summary (50/0/0, 3.70 vs weak) was measured with respawns off and 2s
captures. Same class of failure as the 07-09 setter drop: no explicit
canonical-eval contract. Decision needed: trainer writes curriculum-final
values into the checkpoint, or matrix eval applies canonical overrides.

### Eval bug 2 (fixed): matrix rows dropped every funnel stat

`mappo_matrix_row()` emitted a hand-picked subset; the transfer summary read
majority/uncontested/cap-gain via `.get(..., 0.0)` → all-zero funnel in every
transfer summary ever written (self-evidently impossible: team B logged 45.07
score with "0.0s majority"). Fixed: rows now spread the full
`eval_stats_dict`. `evidence_insufficient` on 07-11 was the blind gate, not
the policy.

### The noop result is a real behavioral hole, not an eval artifact

At every checkpoint and both difficulty settings, vs `noop` the squad walks
past the objective, parks at the enemy spawn, fires zero shots, captures
nothing (behavior probe with flat-obs capture; replays in
`runs/phase4_mappo_conversion_v2_respawn/replays/`). All point play is cued
by enemy contact. A contact-refusing opponent zeroes our offense — this will
bite anchor-mixed self-play. Candidate fix: small curriculum mix vs
noop/passive bots so unconditional point-standing exists.

### Eval hygiene: greedy + fixed map = n=1

Reset seeds produce bit-identical episodes (fixed spawns, deterministic bots,
greedy policy). Every 50-episode cell is one sample repeated; all W/L columns
are 50/0 / 0/50 / 0/0/50, and adjacent-checkpoint gate flips are single
trajectories bifurcating. Before the next run: stochastic eval or spawn
randomization, and report spread over distinct episodes.

### Also verified today

Critic warmup provably froze the actor (bridge == ckpt_0025 behavior
bit-for-bit); in-training best-eval selection (upd 100, score 8.07 at
mid-anneal ~1320t respawn) ran at a third difficulty level, incomparable to
both sweeps, but still picked the canonical-best checkpoint.

### Addendum (same day) — eval contract + opponent mix implemented

Three follow-ups from the post-hoc report, all additive:

1. **Canonical eval is now the matrix default.** `MatrixEvalConfig` gained
   `canonical: true` (default) and `stochastic: false`; matrix rows and
   `scripts/eval_mappo_matrix.py` force unlock 15s / capture 8s / respawn 240t
   through the existing `evaluate_mappo` overrides unless `--as-trained` /
   `matrix_eval.canonical: false`. Rows now also record `respawn_ticks` and
   `std_score_a/b`, so an eased eval can no longer pass as canonical silently.
2. **Stochastic eval** (`evaluate_mappo(stochastic=True)`, seeded torch
   Generator through `sample_action`). First use immediately paid for itself:
   at canonical settings, `ckpt_best_eval` under *sampled* actions goes
   **0W/2L/10D vs weak_basic_v2 with score 0.00** (12 eps) — the canonical
   50/0/1.30 exists only in the greedy mode of the policy. The conversion
   behavior is a knife-edge trajectory, not a robust policy property. Gate
   criteria for the next run should demand stochastic score > 0, not greedy.
3. **Opponent-mix curriculum** (`env.opponent_bot_mix: {weak_basic_v2: 0.9,
   noop: 0.1}`): deterministic largest-deficit per-env assignment
   (`train/opponent_mix.py`), new reset-time `set_opponent_bot` on
   `Phase4MappoEnv` (delegated by the multi-enemy wrapper, propagated by both
   vector backends per-env), applied once by the training hooks. Off by
   default. Targets the noop blindspot: some episodes must contain zero enemy
   contact for "walk on point and stand" to have a gradient at all.

Tests: 453 passed (12 new in `tests/test_opponent_mix.py`); trainer smoke
with the mix enabled runs end-to-end.

## 2026-07-29 — conversion_v3_slowanneal: record canonical peak, the ~550t wall, and the entropy diagnosis

**Status:** run complete (500 updates, ~2h), analyzed same day. Config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v3_slowanneal.yaml`,
seed 3519994491, W&B `mn0boi3v`, artifacts
`runs/phase4_mappo_conversion_v3_slowanneal/`. Warm start: v2 ckpt_0100;
respawn 1200->240 over 400 updates (4.5x slower than v2), timing 12/6 -> 15/8
by update 150, 10% noop mix, anchor 300, critic warmup 10.

### Canonical greedy curve (in-training canonical_eval + post-hoc sweep agree)

Zero through update 200, then **2.87 / 50-0 at update 225** (project record;
uncontested 11.4s) and 1.13 at 250 — then zero from 275 to the end while
kills climbed 13 -> 24. Training respawn at those updates: 660t (225), 600t
(250), 540t (275). The timing ramp finished at 150 and the collapse happened
under frozen timing, so v3b's timing hypothesis is dead. The wall is now
precisely located: **canonical conversion survives while the training env
respawn is >= ~600t and is destroyed once training drops below ~550t —
independent of anneal speed.** The 125-dip/175-recovery around the timing
ramp's end was real but transient; the terminal failure is the respawn
destination, not the approach. Endgame fingerprint at 500: majority 44.9s,
kills 20-0, uncontested 0.9s, cap_gain 51 — the kill treadmill, again.

Corollary: update 225 proves 600t training transfers to 240t greedy eval.
Nobody has ever needed to *train* at 240t to score at 240t.

### Stochastic sweep: the greedy/sampled gap is total

All 20 checkpoints at canonical, 24 sampled episodes each: no checkpoint
scores (best 0.09; ckpt_0225 itself: 0W/4L/20D, 0.00, out-scored by the
bot). Sampled uncontested never exceeds 1.7s vs the 8s requirement, at every
checkpoint, all run. Entropy sat at ~3.10 for 500 updates (fixed
entropy_coef 0.02, log_std_init -1.0, never annealed).

### Step-0 temperature ablation: noise is the blocker, monotonically

ckpt_0225, canonical, stochastic, 24 eps, policy std scaled at eval:

| std scale | W/L/D | score A/B | uncontested s |
|---:|---:|---:|---:|
| 1.0 | 0/1/23 | 0.00/0.13 | 0.9 |
| 0.5 | 2/1/21 | 0.03/0.07 | 3.8 |
| 0.25 | 3/0/21 | 0.16/0.00 | 4.4 |
| 0.1 | 5/0/19 | 0.22/0.00 | 4.6 |

Monotone: wins appear and the bot is shut out as noise drops. The policy's
mean already converts; its sampling distribution cannot hold still for 8s.
(Residual gap at 0.1x = binary-action sampling noise + a mean tuned under
high-noise rollouts.)

### Noop mix: no measurable effect at 10%

vs noop, every v3 checkpoint brushes the point for ~1.9s and never captures —
and re-measuring v2's checkpoints with fixed funnel stats shows the identical
~1.5s brush, so the mix changed nothing (the earlier "v2 = 0.00 contact"
readings were the blind-stats bug). Likely suppressed by the KL anchor
(pinning the walk-past warm start for 250 updates) and 0.02 on-point shaping
being noise-level. Verdict: 10% mix + anchor is insufficient; revisit with
anchor-free noop episodes or a stronger stand-on-point signal.

### Leg 2 design (evidence-driven, one knob)

Warm-start + anchor ckpt_0225. Respawn FIXED at 600t (the proven zone — no
descent). Timing fixed canonical 15/8. The one new knob: **anneal the
entropy bonus to ~0** (and let log_std shrink) over the leg, so the sampled
policy sharpens toward its converting mean. Gate on the stochastic canonical
matrix (score > 0 vs weak_basic_v2), which is now the honest bar. Requires a
small additive trainer feature (entropy-coef anneal schedule, off by
default) — not yet implemented.

Replays: `runs/phase4_mappo_conversion_v3_slowanneal/replays/`
(ckpt0225 vs ckpt0450 at canonical — the conversion and the treadmill).

## 2026-07-30 — conversion_v4_entropy: the anneal that never bit, and the sampling-starvation diagnosis

**Status:** run complete (400 updates at fixed 600t respawn, warm start v3
ckpt_0225, entropy bonus scaled 1.0->0.1 over 300). Config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v4_entropy.yaml`,
W&B `pztpotw7`, artifacts `runs/phase4_mappo_conversion_v4_entropy/`.
(Operational: the Mac slept overnight — caffeinate does not survive a closed
lid on battery; plug in / lid open for unattended runs.)

### Result: null on the stochastic gate, and the intervention never happened

Stochastic canonical matrix (ckpt_final = best_eval = upd 175): 0W/9L/41D
vs weak_basic_v2, score 0.00/1.04. Canonical greedy: the warm start's 2.87
was **gone by update 25** (with the anchor at full strength) and stayed ~0
apart from flickers (375: 0.63/50-0). At *training* difficulty (600t) the
policy stayed strong throughout (best 2.93/50-0 at 175) — so 600->240
transfer is itself fragile and erodes under any continued training; v3's
ckpt_0225 transfer was a property of the anneal trajectory, not of 600t
training per se.

**Root cause the run exposed: `log_std` never moved.** std = 0.451 at every
checkpoint, bit-identical to the warm start. Scaling the entropy *bonus*
only removes upward pressure on entropy; PPO's surrogate provides almost no
gradient to a global log_std parameter, so the policy never sharpened. The
sharpening experiment the temperature ablation motivated was never
physically run. entropy_anneal_updates as implemented is a no-op knob for
its intended purpose (kept for bonus-shaping uses; do not expect it to
reduce std).

### Reward equilibrium is fine; sampling is starved

From the run's own canonical evals: the converting eval (upd 375) accrued
mean_reward +12.39 vs +2.2/+2.9 for adjacent treadmill evals — conversion
out-pays the treadmill ~5x *when it happens*. The problem is that under
std 0.451 an 8s uncontested hold is essentially never sampled, so captures
never appear in rollouts, so there is no gradient toward them: the
converting mode is invisible to PPO, not unrewarded. (Consistent with the
07-29 temperature ablation: eval-time std 0.1x -> sampled wins appear.)

### Next (v5): anneal log_std directly

Implement a scheduled offset applied to `log_std` inside `policy_outputs`
(affects sampling, logprob, and entropy coherently), annealed 0 ->
~ln(0.25) over the run, additive and off by default. Train at 600t, warm
start ckpt_0225 again, anchor short. Gate unchanged: stochastic canonical
score > 0. Secondary question for v5's canonical evals: does 240t transfer
survive when the sampled policy actually experiences captures.

## 2026-07-30 — conversion_v5_logstd: sharpening works, moves the funnel, and finds the real ceiling

**Status:** run complete (400 updates, 600t fixed, warm start v3 ckpt_0225,
log_std pinned 0.451 -> 0.113 over 300 updates). Config:
`experiments/configs/phase4/probe/phase4_mappo_conversion_v5_logstd.yaml`,
artifacts `runs/phase4_mappo_conversion_v5_logstd/`.

### The mechanism worked; the funnel moved; the gate still didn't clear

Checkpoint std tracks the schedule exactly. Under sampling at canonical,
the sharpened policy holds the point like no previous run: majority 13-21s
(v3/v4: 5-9s), uncontested up to 3.6s (previously <=1.7s), cap_gain
100-205. Sampling starvation was real — reducing noise directly improved
every funnel stage. But uncontested time **plateaus at ~3s against the 8s
requirement at every std from 0.28 down to 0.11**, wins stay at 0-1/24, and
scores stay ~0. Lower noise alone cannot buy an 8s window.

### And sharpening at 600t destroys 240t transfer

In-training (600t) eval hit the best score of any run (8.8/50-0 at upd 125)
while the same checkpoints at canonical lost outright (bot scoring 7.10 at
125, 12.37 at 325 — the first time the bot outscores us at canonical). The
sharpened policy specialized hard to the training difficulty.

### Synthesis of the 24h campaign (v3 + v4 + v5)

The July 9 review's root cause 1 stands as the final boss: **at canonical
240t respawn, staggered kills never open an 8s window; conversion requires
near-simultaneous team wipes or lane zoning.** Everything since is
consistent: curricula visit converting behavior transiently on the way down
(v3), continued training at any difficulty erodes it (v4), the reward
prefers it ~5x but PPO's sampled experience caps at ~3s windows even when
sharp (v5). Conversion at 240t is not an attractor of PPO + this reward +
these opponents; single-knob curriculum probes are now exhausted with
instrumented falsifications for each.

### Structural options (next campaign, pick one)

1. **Shape the required behavior directly**: team-wipe simultaneity bonus
   (all three enemies dead within a ~2s window) and/or an approach-lane
   zoning signal — reward the mechanism, not just the outcome.
2. **Move the gate to 600t and ladder via opponents** (weak -> basic ->
   snapshots -> self-play) instead of via respawn; revisit 240t only with a
   stronger policy class.
3. **Coordination capacity**: the wipe requirement is fundamentally a
   3-agent synchronization problem; consider comms/centralized-execution
   features before more reward surgery.

Eval infra note: all three post-mortems ran on the fixed canonical/
stochastic matrix stack built 07-29; every claim above is reproducible from
`runs/*/matrix_eval.json` and the sweep scripts.

## 2026-07-30 — Ladder campaign: gate moves to 600t, Rung A cleared, Rung B1 designed

**Decision (option 2 of the synthesis):** the Phase 4 gate is redefined as
stochastic conversion at 600t respawn / 15s / 8s, laddered by opponent
strength toward snapshots and self-play. 240t is deferred until the policy
class can express coordinated wipes.

### Step 0 — the stochastic 600t baseline (as-trained matrix, 24 eps)

v5's sharpened checkpoints CLEAR rung A — the first stochastic gate ever
passed in this project: vs weak_basic_v2 at 600t, ckpt_0250 18/1/5
(score 2.53), **ckpt_0300 17/1/6 (2.71, bot 0.15)**, ckpt_0400 17/5/2
(3.57), uncontested 10-12.6s. v5's "overfit" is the ladder's foundation.

### Opponent roster probed (ckpt_0300, stochastic, 600t)

walk_to_objective: 18/24 wins (2.25) — below current level.
weak_basic_v2: cleared (above).
**hold_and_shoot: 0-0 stalemate x24** — a turret: never contests (majB
0.0s) but kills us 15.4-0.3/round; our hit_fire collapses to 0.009. The
approach-discipline / coordinated-assault lesson, isolated.
weak_basic: 0/24, -23.5. basic: 0/24, -33. Late rungs.

Ladder: [A: weak_basic_v2 ✓] -> [B1: hold_and_shoot] -> [B2: weak_basic]
-> [B3: basic] -> [C: snapshots/anchor-mixed self-play].

### Rung B1 run design

`phase4_ladder_b1_hold.yaml`: warm start v5 ckpt_0300 (std 0.113 baked),
600t/15s/8s fixed, opponent_bot_mix {weak_basic_v2: 0.5, hold_and_shoot:
0.5} (the 07-29 mix machinery's first real use), anchor on warm start
annealed 150, no std/entropy anneals, matrix_eval canonical: false +
stochastic (the gate is as-trained 600t). Gate: retain weak_basic_v2
stochastic score >= 2 while hold_and_shoot moves off 0-0 (any stochastic
wins / score > 0). Falsified if weak retention breaks (mix too aggressive)
or hold_and_shoot kills-against stay ~15/round by update 300 (mix cannot
teach approach discipline; consider a dedicated hold-breaking reward or
aim-noise-softened variant).

## 2026-08-01 — ladder_b1_hold: retention machinery works; the turret lesson doesn't take

**Status:** run complete (400 updates, 600t, warm start v5 ckpt_0300, mix
{weak_basic_v2: 0.5, hold_and_shoot: 0.5}). Artifacts
`runs/phase4_ladder_b1_hold/`. Post-run stochastic matrix at 600t (50 eps):

- **weak_basic_v2: 36/3/11, score 2.49/0.70** — rung A RETAINED through 400
  updates of mixed training (gate was >= 2). The mix + anchor machinery
  does its job; training on a second opponent no longer destroys the
  cleared rung. Best eval 5.23/50-0 at upd 75.
- **hold_and_shoot: 0/0/50, 15.9 deaths/round against, hit_fire 0.008** —
  bit-identical to the pre-run baseline. 400 updates of 50% exposure taught
  nothing. The B1 falsification criterion fired as written.
- basic: 0/50 (expected, untargeted).

### The pattern, now twice

Turret-breaking fails for the same structural reason canonical conversion
failed: the rewarded behavior (a coordinated, well-executed assault) is
never sampled — every naive approach dies before any positive signal, so
"stay away and stalemate" is the sampled-data optimum. Mixing more turret
episodes cannot fix a gradient that never sees a success.

### Next levers (from the B1 config's falsification clause)

1. **Softened-turret curriculum**: hold_and_shoot with opponent aim noise /
   damage reduction annealed toward full strength (the July 9 review lists
   opponent aim noise among existing sim levers — verify the env knob).
   Same shape as the respawn curriculum, applied to the opponent.
2. **Dedicated shaping**: reward damage dealt to the holder / first-contact
   survival, so partial progress on the assault pays before a kill does.
3. Skip the turret: rung order is a choice — weak_basic (mobile, -23) may
   be more learnable than a one-shot turret, though it likely shares the
   failure mode.

## 2026-08-01 — ladder_b1c_handicap: avoidance is absorbing; the turret rung is skipped with cause

**Status:** run complete (400 updates, handicap anneal 1.5rad/60t -> 0/1 on
hold_and_shoot, mix 40/60, warm start B1 best). Artifacts
`runs/phase4_ladder_b1c_handicap/`. Double failure: full-strength turret
still 0/0/50, and weak_basic_v2 retention BROKE (13/17/20, 1.96/4.13 —
gate was >= 2; B1 held 36/3/11).

### Root cause: the warm start never engaged, so the anneal never bit

Diagnostic: B1c ckpt_0050 vs a turret softened BELOW its training-time
handicap (1.25rad/50t): 0/0/24, **0.1 kills, hit_fire 0.014** — no
engagement at any softness. The B1 warm start carried 400 updates of
learned turret-avoidance, and avoidance is an absorbing equilibrium: the
policy never samples the turret zone, so opponent softness is unobservable
and provides no gradient. Third instance of the campaign's master pattern
(canonical conversion, assault, now re-engagement): **a behavior that is
never sampled cannot be trained, no matter how it is rewarded or how easy
it is made.** The handicap mechanism itself is unexercised, not refuted
(cf. v4's entropy knob).

Eval-infra note: transfer_summary silently filters to
`matrix_eval.transfer_bots` (default noop/weak_basic_v2/basic) — the
turret row existed only in matrix_eval.json. Set transfer_bots explicitly
in ladder configs.

### Decision: skip the turret (journal 08-01 option 3, now evidence-backed)

Two runs, two structural failure modes (no-signal cliff; avoidance
absorption). hold_and_shoot is a stationary siege specialist HARDER than
basic in pure combat, and nothing on the ladder path (weak_basic, basic,
snapshots, self-play) is stationary. It is a side quest, not a rung.

### Rung B2: weak_basic via handicap, with perfect continuity

`weak_basic` at handicap (1.5rad, 60t) is BIT-IDENTICAL to weak_basic_v2
(same walk_to_objective + hold_and_shoot composition; those parameters ARE
the v2 nerf). B2 therefore starts as an opponent the policy already beats
(guaranteed engagement from update 1) and anneals parametrically to native
weak_basic (0.5rad, 1t) over 300 updates — no behavior cliff at any point.
Warm start: v5 ckpt_0300 (clean, no avoidance baggage). Mix
{weak_basic_v2: 0.3, weak_basic: 0.7}, anchor 150. Gate: stochastic
weak_basic wins/score > 0 at full strength, weak_basic_v2 retention >= 2,
transfer_bots set explicitly.

## 2026-08-01 — ladder_b2_weakbasic: capacity is the wall; the redefined gate is CLEARED

**Status:** run complete (400 updates, handicap continuity weak_basic
1.5rad/60t -> native 0.5rad/1t, mix 30/70, warm start v5 ckpt_0300).
Artifacts `runs/phase4_ladder_b2_weakbasic/`.

### Result

- **weak_basic_v2 retention: 34/2/14, score 2.35/0.37** — strongest
  stochastic result vs v2 yet; clean-warm-start + 30% mix + anchor holds.
- **weak_basic: -20.00 reward, 0 wins at EVERY checkpoint** including
  300-400 (trained at full native strength). The B2 falsification fired as
  written: engagement was guaranteed, the anneal was cliff-free
  (weak_basic at initial handicap is bit-identical to v2), and the skill
  still did not track. The binding constraint is combat capacity — our
  policy fights at hit_fire 0.03-0.07 / aim error ~0.5rad against a bot
  with 0.5rad noise at full cadence. Curriculum shape is exonerated by
  construction; the policy class cannot express the required aim/dodge.
- Selection artifact worth fixing for any future rung run: best-eval uses
  the env-cfg opponent (v2), so ckpt_final aliased to update 50 and the
  auto-matrix judged a barely-trained checkpoint. Rung configs should set
  the eval opponent to the target bot (retention via the matrix instead).

### Gate accounting (the honest ledger)

The redefined Phase 4 gate (journal 2026-07-30) — **stochastic conversion
at 600t** — is **CLEARED**, three times over: v5 step-0 (17/1/6), B1
(36/3/11), B2 (34/2/14), all vs weak_basic_v2 at full strength, all
reproducible from matrix artifacts. What remains open is the ladder ABOVE
v2, and the B1/B1c/B2 falsification chain shows every scripted rung beyond
it is capacity-blocked, not curriculum-blocked.

### The fork (next campaign, one of):

1. **Self-play now** (the ladder's stated destination): snapshot-league /
   anchor-mixed machinery exists (phase11 env, snapshot retention).
   Self-play opponents share the policy's capacity limits — the game is
   fair by construction, which is exactly what the scripted rungs are not.
   Bootstrap from the B2 final population (v2-retention-strong).
2. **Combat capacity**: dedicated aim/dodge pretraining in the existing
   mini-envs (phase4_aim_only, combat_1v1, cap_duel + distill plumbing),
   or obs/action/network upgrades (target-lead features, aim assist,
   bigger heads) — then re-run B2 unchanged as the capacity metric.

## 2026-08-01 — selfplay_l1: the flywheel spins backwards against frozen greedy experts

**Status:** run complete (400 updates, static pool {v5-0300, B1-075,
B2-400} + 30% weak_basic_v2 anchor, warm start v5-0300). Artifacts
`runs/phase4_selfplay_l1/`. Machinery all worked: snapshot factory wiring,
snapshot:<path> mix entries, obs-routing fix — snapshot rows scored in the
matrix on the first try.

### Result (stochastic matrix, ckpt_final = best_eval = upd 325)

- weak_basic_v2: 35/6/9, 2.69 — retention gate holds again.
- **B2-400 (weakest ancestor): 41/0/9, score 9.22** — crushed.
- **v5-0300 (its own warm start): 0/46/4, outscored 0.00/5.00,
  majority ceded 7.2s/26.6s.** B1-075: 2/45/3. At the pre-run smoke the
  warm start drew its frozen self 2/2; training made it WORSE against the
  strong ancestors.

### Master pattern, fourth appearance

SnapshotPolicy plays GREEDY, and a greedy v5-class converter camped on the
objective is functionally a turret. Strong-ancestor episodes punish
approach -> the learner learns avoidance -> avoidance absorbs (cedes the
point and farms the beatable 50% of its mix instead). Static frozen
experts are not fair-by-construction self-play; they are scripted bots
with extra steps. Canonical conversion (v3-v5), the turret (B1/B1c),
weak_basic (B2), and now frozen ancestors: every failure in this campaign
is the same theorem — behavior that is never successfully sampled cannot
be trained, and against punishing opponents the sampled optimum is
avoidance.

### Leg 2 design — make the fight fair two ways

1. **Sampled snapshot opponents** (`opponent_snapshot_stochastic`): frozen
   selves play their stochastic policy — the distribution the learner
   actually was — not a sharpened greedy caricature of it.
2. **Recent-self pool refresh**: snapshot slots re-point at the learner's
   OWN recent checkpoints (lags of 25/50/100 updates) at every checkpoint
   event, using the existing mix-assignment + runtime setters. The skill
   gap stays small by construction — true iterated self-play — instead of
   frozen experts from other lineages. Warm start v5-0300 (leg-1's final
   carries avoidance). Anchor 30% weak_basic_v2 unchanged; matrix gate:
   positive edge vs the leg-1 seed ancestors under sampled opponents +
   retention >= 2.

## 2026-08-01 — selfplay_l2 + the control that flipped both verdicts

**Status:** leg 2 complete (sampled opponents, recent-self pool lags
25/50/100, warm v5-0300). Artifacts `runs/phase4_selfplay_l2/`. Raw matrix:
vs sampled v5-0300 11/38/1 (2.52/5.97), vs sampled B1-075 14/35/1, vs
B2-400 43/3/4; weak_basic_v2 retention SLIPPED to 1.62 (16/11/23), below
the >= 2 gate. Best-eval 9.10 at update 400 (still climbing at the end).

### The control: evaluate leg-1's final against a SAMPLED ancestor

Leg 1's matrix used greedy ancestors; leg 2's used sampled ones — the two
legs were never measured on the same axis. Control (24 eps, seed 4242):
**leg-1 final vs sampled v5-0300: 13/8/3, score 5.40/2.09.**

Both verdicts flip:
- Leg 1's "0/46 catastrophe" was a MEASUREMENT artifact: its policy fights
  sampled opponents fine and holds a positive edge over its ancestor. The
  avoidance it learned was specifically anti-greedy-turtle, and greedy
  frozen play is not a deployment condition any policy will ever face.
- Leg 2's "0/46 -> 11/38 progress" was the same artifact in reverse
  (opponents got easier). Measured fairly, leg 2 is WORSE than leg 1
  against the ancestor AND lost weak retention. The recent-self treadmill
  (chasing a moving copy of yourself) produced treadmill skill, not
  transferable strength.

### Corrected ledger

Under fair sampled measurement, **leg 1 already passed the self-play
gate**: positive W/L and score edge vs its seed ancestor (13/8, +5.40),
retention 35/6/9 (2.69 >= 2), weakest ancestor crushed 41/0. Champion
checkpoint: `runs/phase4_selfplay_l1/mappo/ckpt_final.pt` (upd 325).

### Doctrine addition (extends 07-29 eval hygiene)

Greedy vs sampled applies to OPPONENTS, not just learners. Frozen-greedy
evaluation turtle-ifies converters and measures a fight nobody deploys
into. All future matrix snapshot rows: sampled opponents
(opponent_snapshot_stochastic: true in eval env cfg), and never compare
rows across opponent-sampling regimes.

### Open next steps (not yet run)

Leg 3 candidate: leg-1's static-pool recipe but with SAMPLED training
opponents and anchor >= 0.35 — test whether removing the greedy-turtle
training environments improves on leg 1's already-positive edge without
the leg-2 treadmill. Secondary: PFSP-style weighting before pool growth.

## 2026-08-02 — selfplay_l3 (overnight): Champion 2 crowned at update 600

**Status:** run complete (800 updates, champion warm start, sampled static
pool incl. frozen champion copy, anchor 0.35). Artifacts
`runs/phase4_selfplay_l3/`. The auto-matrix judged ckpt_final = best_eval =
**update 25** (the best-eval selection artifact, third bite: in-training
eval measures the anchor game the warm start already aces). The real
result was in the late checkpoints, recovered by sweep:

| ckpt | vs sampled champion | vs weak_basic_v2 |
|---:|---|---|
| 100 | 4/., -8.4 | 15/., +4.1 |
| 400 | 1/., -12.1 (transient trough) | 2/., -0.9 |
| **600** | **13/11/0, score 7.40/1.80** | **13/3/8, score 3.02 (gate ok)** |
| 800 | 15/8/1, 5.34/0.44 | 0/10/14, 0.00 (retention DEAD) |

**ckpt_0600 passes both gates simultaneously: it beats the leg-1 champion
under fair sampled play with a dominant score margin AND retains the
weak_basic_v2 gate.** Crowned Champion 2:
`data/checkpoints/phase4_selfplay_l3_champion2.pt`. The 600->800 tail is
the retention see-saw: continued specialization vs converter-shaped
opponents ate the scripted-walker game — anchor 0.35 held to ~600, not
800.

### Lessons carried to leg 4

1. Fix the selection artifact structurally: with snapshot opponents now
   config-native, set the in-training eval opponent to the SAMPLED current
   champion (env opponent_bot: snapshot + stochastic flag), so best-eval
   tracks the true objective and ckpt_final stops aliasing to warm-up
   checkpoints.
2. Generation cadence: ~600 updates per generation at anchor 0.35, then
   re-crown and restart, rather than longer runs that decay past the
   sweet spot. This is the league loop: crown -> freeze -> train against ->
   crown.
