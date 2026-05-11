# Phase 4 Cap-Training Escalation — Design

Status: active
Owner: TBD
Last-updated: 2026-05-11

**Goal:** Get the Phase 4 MAPPO policy to actually train against the `basic` opponent. Apply a 3-rung escalation, climbing only when the prior rung empirically fails.

## Why this is needed

Two 250-update runs of `phase4_mappo_basic.yaml` produced `mappo_final = -11.000` with 0/10 wins, 0/7 score. Viewer of the update-25 checkpoint shows agents spinning *away* from the cap; the basic bot fires continuously while the (random-init) policy tries to approach. Without enough cap-aligned signal, the policy never discovers "stay on cap is good." Our previous post-implementation-fix run (3× shaping) made it *worse* (agents avoided combat entirely) — which falsifies "shaping is too weak in magnitude" as the root cause and points at "shaping shape is wrong" + "curriculum is too hard."

Counter-example: `phase4_mappo_objective_probe.yaml` uses BC pretrain + on-point shaping against `noop` and trains a winning policy in one update (verified by `test_phase4_mappo_bc_eval_can_be_best_result`). So the training pipeline works; the issue is the specific config.

## Strategy: 3-rung escalation

Each rung adds one mechanism. Climb only after the prior rung empirically fails. This isolates which intervention is *necessary* (vs. nice-to-have) and minimizes total training time.

**Rung 1 — Shaping fix.** Distance + on-point shaping coefs from `objective_probe.yaml` (`distance_shaping_coef: 0.05, on_point_shaping_coef: 0.02`). Same opponent (`basic`), same ramp, no other changes.

**Rung 2 — Add BC pretrain.** Same as rung 1 plus `bc_pretrain_steps: 200, bc_batch_size: 192, bc_learning_rate: 1.0e-3`. BC imitates a "walk to objective" expert before PPO starts, giving the policy a non-random initialization that's already cap-pointed.

**Rung 3 — Two-stage warm-start.** (a) Train against `noop` (no enemy fire) with rung 1's shaping. (b) Warm-start that checkpoint into a run against `basic`. The agent learns to hold the cap first, then learns to fight while holding it.

## Stopping criterion (when to declare a rung "failed")

After eval at update 100 (40% of 250-update run), declare success and finish the run if **any** of:
- `mean_reward > -8` (better than always-lose-by-shaping-cap baseline of -11)
- `onpt > 0` in the trainer's onpt metric (touched the point at least once)
- our team `kills > 0` in eval (engaged in combat at all)

If all three are still at frozen-agent values at update 100, kill the run and proceed to the next rung.

This is asymmetric on purpose: easy to trigger "keep going," hard to trigger "this is broken." Avoids false-positive failures from variance.

## Out of scope

- Tuning entropy_coef, lr, value_coef, etc. — those are a separate axis if shaping+BC+curriculum still fail.
- Modifying the per-agent reward path or `team_spirit` ramp — keep both as-is so we can compare against the basic.yaml runs directly.
- Phase 5+ envs — Phase 4 only.
- Defining a numerical "Phase 4 acceptance gate" — separate plan.

## Success conditions

Plan succeeds when *one* of the rungs produces an eval with `wins > 0` and `mean_reward > 0` (i.e. some terminal wins, not just shaping). The rung that succeeds becomes the "phase 4 baseline" config. The implementation plan documents which rung was needed.
