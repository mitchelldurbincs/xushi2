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