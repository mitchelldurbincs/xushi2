# Phase 3 — Noop-probe result

**Date:** 2026-04-24
**Status:** End-to-end plumbing confirmed. This is a *diagnostic probe*, not the Phase 3 sign-off — the real gate is the `basic`-opponent run (`phase3_ranger_recurrent.yaml`). The probe proves the C++-sim → recurrent-PPO pipeline produces a policy that reliably beats a scripted opponent on the real objective state machine.

## Final numbers

From `py -3.13 -m train.train --config ../experiments/configs/phase3_ranger_noop_probe.yaml` (interrupted at update 275/500 — converged well before the stop point):

```
[phase3/recurrent] eval@ 25=+1.650  win=0/20  score=A0.00/B0.00  draw=20/20
[phase3/recurrent] eval@ 50=+1.697  win=0/20  score=A0.00/B0.00  draw=20/20
[phase3/recurrent] eval@ 75=+11.926 win=20/20 score=A0.30/B0.00  term=20/20  ← first win
[phase3/recurrent] eval@100=+1.832  win=0/20  score=A0.00/B0.00  draw=20/20   ← one regression
[phase3/recurrent] eval@125=+12.190 win=20/20 score=A2.77/B0.00
[phase3/recurrent] eval@150=+12.082 win=20/20 score=A1.83/B0.00
[phase3/recurrent] eval@175=+11.913 win=20/20 score=A0.77/B0.00
[phase3/recurrent] eval@200=+11.770 win=20/20 score=A0.37/B0.00
[phase3/recurrent] eval@225=+11.790 win=20/20 score=A0.23/B0.00
[phase3/recurrent] eval@250=+11.758 win=20/20 score=A0.47/B0.00
[phase3/recurrent] eval@275=+11.800 win=20/20 score=A0.13/B0.00
```

| Signal | Gate | Actual | Status |
|---|---|---|---|
| Wins at eval (≥ update 125) | 20/20 | 20/20 at every eval | ✅ |
| Eval reward | ≥ +10 (terminal win floor) | +11.76 → +12.19 | ✅ |
| `critic_gn` | < 10 (was 686 pre-fix) | 0.025–0.2 | ✅ |
| `value_loss` (update 275) | stable < 0.1 | 0.005 | ✅ |
| `term_adv_std` | real variance, then collapse as critic fits | 3.97 → 0.00 at 275 | ✅ |
| `policy_loss` | non-zero (was ±0.000 pre-fix) | -0.001 to -0.005 | ✅ |

First capture at **update 75** — 3× faster than the "100–200 update" prediction I made at update 50.

## What this probe confirmed

End-to-end plumbing works for Phase 3:

1. **C++ sim → pybind11 → Python env**: actions decode correctly (movement, aim, fire all exercised).
2. **Objective state machine**: 15-second lock window, 240-tick capture, ownership transfer, score accrual all fire in order (verified independently in `python/scripts/diag_phase3_plumbing.py`).
3. **Env → reward calculator**: shaped + terminal rewards symmetrize and clip correctly; new optional `distance_shaping_coef` term flows through.
4. **Rollout buffer → recurrent PPO**: GAE propagates, value targets are learnable, advantage variance drives policy updates.
5. **Policy → sim**: emitted actions produce the intended behavior (verified at update 75 when policy first walked to cap, completed the 240-tick capture, held past tick 690 to score, won by differential).

## Root-cause fixes that landed this result

Before these fixes, evals were stuck at `draw=20/20, score=A0.00/B0.00, eval_reward=+0.000` for all 100 updates observed. Three pathologies were compounding:

1. **Value-normalization divide-by-zero-std** — with all episodes drawing at tick 900 and no shaping events firing, per-rollout return std ≈ 0 made value-norm z-scoring blow up. `critic_gn` hit ~2500. **Fix:** give the rewards real variance (see #3).
2. **Kill-bonus distractor vs. noop** — `kill_bonus: 0.25` against a motionless opponent creates a locally-attractive strategy (chase enemy, kill, repeat) that is *orthogonal* to the cap-sitting strategy needed to score. **Fix:** `reward.kill_bonus: 0.0` in the probe config.
3. **Zero-shaping-density around the objective** — `rl_design.md` §5 shaping is event-triggered (score ticks, kills, deaths). Agent at random init never triggers any event, so there's no gradient toward the cap. **Fix:** added optional `distance_shaping_coef` (default 0.0, probe uses 0.01) in `python/xushi2/reward.py`. Per-decision symmetrized `-coef × (dist_A − dist_B)` gives a dense gradient toward the cap without breaking the zero-sum invariant.

Files changed:
- `experiments/configs/phase3_ranger_noop_probe.yaml` — probe YAML with `kill_bonus: 0.0`, `distance_shaping_coef: 0.01`, `score_per_second: 0.1`.
- `python/xushi2/reward.py` — new `distance_shaping_coef` kwarg on `RewardCalculator`, preallocated obs buffers behind the feature flag, added to `raw_a` before `raw_b = -raw_a` symmetrization.
- `python/tests/test_reward.py` — three new tests (validation, default-zero behavior, end-to-end smoke test through `XushiEnv`).
- `python/scripts/diag_phase3_plumbing.py` — plumbing diagnostic used to prove the issue was *not* a wiring bug before touching reward.

## What was NOT the issue (retracted)

During debugging, the following were suspected and ruled out via the diag script:

- **Round too short to win.** Earlier claim that `kWinTicks = 3000` (100 s) couldn't be reached in a 30 s round and therefore made the game "mathematically unwinnable." False — the sim declares the team with more score_ticks the winner at timeout, so *any* positive score differential wins. The `sit_on_cap` diag scenario confirmed +10.700 reward in a 30 s round before any code changes.
- **Movement or aim plumbing bugs.** The `homing` scenario killed the noop twice in a single episode with hand-written actions, confirming action decoding, aim delta accumulation, and hitscan combat all work.
- **Cap geometry or `on_pt` detection.** The `sit_on_cap` scenario hit the `on_pt 0→1` flag at tick 123 (~4 s from spawn), exactly matching the expected travel time at max speed. All state-machine transitions fired at the expected ticks (450 unlock, 510/570/630/690 cap quartile crossings, 690 ownership, 693 first score).

Real root cause was the exploration–reward mismatch described above.

## Subtle observations worth noting

**Score trend is non-monotone (peaks at update 125 with A=2.77, drifts down to A=0.13 by update 275).** Wins stay 20/20 the whole time. This is the reward structure saturating: once the agent reliably earns distance shaping (~+2.4) + any-score-to-win + terminal +10, total reward plateaus near +12. Additional hold time past "first score tick" yields no gradient because cumulative shaping has already hit the ±3 clip. The policy drifts toward the cheapest winning behavior (just graze the cap long enough to score once), which is optimal under the current reward landscape but produces a less-robust policy than intuition suggests.

**`mean_log_std` widened slightly over training** (-0.971 at update 25 → -0.900 at update 275; σ ≈ 0.38 → 0.41) rather than narrowing. Entropy bonus (`entropy_coef: 0.01`) is stronger than exploitation pressure in steady state. Not a problem here — wins are unconditional — but the final policy is not particularly deterministic. Would matter more against an adversarial opponent.

**Update-100 regression** (+11.926 at 75 → +1.832 at 100 → +12.190 at 125). One eval dropped back to draw=20/20. Possibly a brief excursion during the initial strategy lock-in; critic was still fitting (`value_loss 0.056 → 0.080` at that transition). Recovered cleanly. Worth watching for in subsequent runs but not a blocker here.

## Artifacts on disk

`runs/phase3_ranger_noop_probe/`:
- `ckpt_100.pt`, `ckpt_200.pt` — periodic snapshots (checkpoint_every=100).
- `ckpt_final.pt` — best-eval snapshot (expected to be from around update 125 based on peak eval=+12.190).
- TB event files for the full trajectory to update 275.

## Next step

The real Phase 3 gate is `experiments/configs/phase3_ranger_recurrent.yaml` — the `basic`-opponent run. Considerations for that run:

1. **Re-enable `kill_bonus`** (default 0.25). Against `basic`, combat is genuine, not a distractor; a learner that can shoot back is a prerequisite for contesting the cap.
2. **Keep `distance_shaping_coef: 0.01` or reduce to 0.005.** Basic walks to cap on its own, so contesting is denser; the shaping may no longer need to be the dominant signal. But reducing to zero likely reintroduces the exploration problem.
3. **Drop `score_per_second` back to the default 0.01** unless explicit probe tuning is wanted — the probe's 10× default was to compensate for the otherwise-sparse reward.
4. **Consider `entropy_coef: 0.005`** (half the current) if `mean_log_std` continues widening during the basic run — against an adversarial opponent, a mushy policy will lose ties.

The fixes in `reward.py` and the new probe YAML are additive and don't require any change to the pending Phase 3 basic-opponent run.

## What was also fixed in the session

Not in this run's commit scope, but produced alongside the debug work:
- `python/scripts/diag_phase3_plumbing.py` — reusable plumbing diagnostic with `--round-length-seconds` override. Four scenarios (`sit_on_cap`, `homing`, `forward`, `still`). Use when PPO stops converging — if `sit_on_cap` still wins in 30 s, the sim is fine.
- Surfaced in the earlier spec-drift audit (`docs/plans/2026-04-24-spec-drift-audit.md`) as follow-ups: the misleading `[xushi2] phase=3 episodes=4 bots=basic vs basic` header in `python/train/train.py:86-87` should pull `opponent_bot` from `env_cfg` for Phase 3 so logs accurately reflect the running scenario.
