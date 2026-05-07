# Phase 3 — 1v1 Ranger result

**Date:** 2026-05-07
**Status:** Gate cleared. Phase 3 sign-off.

## Final numbers

From `python -m train.train --config experiments/configs/phase3_ranger_objective_curriculum_warmstart_v3.yaml`:

```
[phase3] recurrent_final=+9.688
```

The v3 run reached a stable scoring plateau and was stopped at update 600 with 12 consecutive 50/50 wins.

| Eval window (updates 325–600) | Mean reward | Win rate | Score component (A) |
|---|---|---|---|
| Min | +9.567 | 50/50 | 0.07 |
| Max | +9.718 | 50/50 | 1.20 |
| Final (eval@600) | **+9.688** | 50/50 | 0.63 |

Win rate held at 50/50 for **12 consecutive evals** (eval@325 through eval@600). The score component oscillated within the plateau (A0.07–1.20) but did not break the win rate, indicating the policy reliably wins by terminating the round on score even when the per-tick scoring rate is low.

Artifacts on disk (`runs/phase3_ranger_objective_curriculum_warmstart_v3/recurrent/`):
- `ckpt_0600.pt` — final saved checkpoint, the recommended Phase-4 warm-start source
- `ckpt_{0050..0600}.pt` — checkpoints every 50 updates, 12 total

## What was hard

The Phase-3 task is harder than Phase 2 in a way the original config didn't appreciate: the agent must transition from a stable-but-zero-score equilibrium (deny the cap) to a higher-variance scoring policy. With the default reward shape, the deny equilibrium is locally attractive and PPO never escapes it.

Three runs traced the failure modes:

| Run | Reward shape | LR | Outcome |
|---|---|---|---|
| `phase3_ranger_objective_curriculum` (baseline) | `score_per_second=0.05`, no time penalty | 3e-4 cosine | Perfect cap-denial. Best eval **−0.074** at update 1425. 19+ stable 50/50 draws across updates 875–1475. Never scored. |
| `phase3_ranger_objective_curriculum_warmstart` (v1, warm-start from baseline) | `score_per_second=0.15`, no time penalty | 1e-4 cosine | Single scoring cluster at evals 350–400 (peak **+10.098**, score A1.20). Collapsed at eval@425 and never recovered. |
| `phase3_ranger_objective_curriculum_warmstart_v2` (warm-start from v1's score peak) | `score_per_second=0.10`, no time penalty | 5e-5 cosine, clip 0.15 | Multiple scoring clusters but bouncing back to deny basin every ~50–100 updates. Best peak **+10.226** (score A2.00) at eval@300. Oscillated indefinitely; killed at update ~500. |

The pattern across v1 and v2 was unmistakable: brief scoring breakthroughs followed by reliable collapse to 0/0 draws. Tightening LR, clip ratio, and warm-start source delayed but did not eliminate the collapse — proof the failure was structural, not a hyperparameter problem.

## The hurdle: deny-stalemate basin

The reward landscape under `walk_to_objective` opponent (which doesn't fire) had two locally stable behaviors:

1. **Deny:** approach the cap to contest. Both teams' Rangers stand on the point. Neither side scores. Episode truncates at 30s with terminal reward 0. Per-tick `distance_shaping_coef=0.01` rewards being near the cap, so the agent collects a small positive shaped reward each tick. **Net per-episode return: ~0 to slightly positive.**
2. **Score:** push the opponent off the cap (requires firing/positioning) and hold solo control. Earns `score_per_second` per tick of solo cap. Wins terminate the round early at `terminal_win=+10`. **Net per-episode return: ~+10 when it works, ~0 when it fails.**

Strategy 2 has a strictly higher expected return *if* the policy can execute it reliably. But during the policy-update steps that happen between successful executions, PPO's gradient estimate is dominated by the lower-variance strategy 1 — and as long as strategy 1's expected return is non-negative, there is no gradient signal pulling the policy *away* from it. The result is a stable basin: every time PPO discovers scoring, the next few rollout batches contain a mix of scoring and not-yet-scoring trajectories, and the noisy gradient update tends to revert toward the safe basin.

## The fix that landed this result

`time_penalty_per_second=0.05` was added to `RewardCalculator`. It applies a small per-tick negative reward to **both teams**, intentionally breaking the zero-sum symmetrization of the rest of the shaping (the symmetric block stays zero-sum; the time penalty is added on top to both teams equally).

This eliminates the deny basin: a 30-second 0/0 draw now nets approximately `−0.05 × 30 = −1.5` for both teams (capped by `shaping_clip=3.0`). The deny basin's expected return goes from ~0 to clearly negative, while the scoring policy's expected return is unchanged (winning still terminates the episode early, *avoiding* the time penalty). PPO's gradient now points unambiguously at scoring.

The behavior change after introducing the time penalty (run v3, otherwise identical to v2):

| Eval | v2 (no tps) | v3 (tps=0.05) |
|---|---|---|
| 25 | +10.129 (won — warm-started) | −0.648 (deny — basin tax visible) |
| 100 | +10.197 | +9.412 (broke out of deny) |
| 125 | −0.497 (collapsed) | −0.497 (collapsed — same pattern still present) |
| 175 | −0.110 (still in deny) | +9.614 (recovered in 2 evals) |
| 250 | +10.090 (cycle repeats) | −0.394 |
| 325 | −0.077 (cycle repeats) | +9.691 — stable from here |
| 400 | +0.071 (deny+kills, no score) | +9.596 — 5 wins in a row |
| 500 | … | +9.587 — 8 wins in a row |
| 600 | … | +9.688 — 12 wins in a row |

The oscillation pattern doesn't disappear at the 25-update granularity in v3 either, but the basin escape becomes much faster (1–2 evals instead of v2's 75–100), and after update 325 the deny excursions stop entirely. v3 found a stable scoring policy where v2 could not.

## Code changes that landed this result

All in commit `c482235 Phase 3 gate: warm-start + reward time-penalty`:

1. **Warm-start hook** in `python/train/ppo_recurrent/orchestration.py`. New `_load_init_checkpoint(model, ckpt_path, expected_model_cfg)` helper plus a `run.init_from_checkpoint:` config key. Loads weights only — optimizer state, LR schedule state, and rollout buffer are intentionally fresh. Architecture mismatch (different `obs_dim`, etc.) raises a clear ValueError. 4 new tests in `python/tests/test_warm_start.py`.
2. **`time_penalty_per_second` parameter** in `python/xushi2/reward.py`. Default 0.0 (backwards compatible). When non-zero, subtracts `tps / TICK_HZ` from both teams' raw shaped reward each tick before clipping. 3 new tests in `python/tests/test_reward.py`.
3. **CI extension build.** `.github/workflows/ci.yml` now builds the C++ Python extension for the test interpreter so `pytest` can import `xushi2_cpp` on CI runners.
4. **Three new configs** under `experiments/configs/`: `phase3_ranger_objective_curriculum_warmstart{,_v2,_v3}.yaml`. v3 is the one that cleared the gate.

Full test suite: 114 passing.

## What we'd do differently next time

- **Bake the time penalty into the default reward.** With `tps=0.0` as the default, the deny basin is the default behavior. For Phase-4 onward, recommend `time_penalty_per_second=0.05` (or higher; clip caps it anyway) as the default config, with `tps=0.0` reserved for explicit "no draw pressure" probes.
- **Don't iterate hyperparameters when the failure is structural.** v1 → v2 was 8h of training to discover that lr/clip didn't fix the basin. The pattern (peak → collapse → never recover for v1; peak → collapse → recover → collapse for v2) was clear after ~3h of v1; the structural diagnosis should have come faster. Lesson: when two consecutive runs show the same qualitative failure mode, stop tuning and look at the reward landscape.
- **Save best-checkpoint explicitly.** v1's `ckpt_final.pt` was the best-eval snapshot (working as designed). But during v1, the best snapshot was at eval@375 = +10.098, between two save boundaries (ckpt_0300 and ckpt_0400 at the time, since checkpoint cadence was 100). v2 dropped checkpoint cadence to 50 to mitigate. Better: a separate `ckpt_best.pt` written at any improvement, in addition to the periodic snapshots. (Not required for Phase 4; flagged for follow-up.)

## Next step

`docs/plans/2026-04-24-phase4-prep.md` is the kickoff plan. The recommended order there starts with widening `src/sim/src/critic_obs.cpp` to 3v3 (item 2 in that doc, now item 1 since the viewer gate was reordered to be done in parallel rather than as a hard prerequisite). The Phase-4 warm-start source is `runs/phase3_ranger_objective_curriculum_warmstart_v3/recurrent/ckpt_0600.pt` once `n_agents` extension lands and an architecture migration path is decided.
