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