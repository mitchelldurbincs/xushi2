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
