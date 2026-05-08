# Phase 2 — Memory-toy result

**Date:** 2026-04-23
**Status:** Gate cleared. Phase 2 sign-off.

## Final numbers

From `python -m train.train --config experiments/configs/phase2_memory_toy.yaml`:

```
[phase2] recurrent_final=-0.107 feedforward_final=-0.995 gap=0.889
```

| Variant | Best eval | Update | Analytic bound | Gate |
|---|---|---|---|---|
| Recurrent | **−0.107** | 660 | 0.0 | ≥ −0.15 ✅ |
| Feedforward | **−0.995** | 420 | ≈ −1.00 | matches bound ✅ |
| Gap (recurrent − feedforward) | **0.889** | — | ≈ 1.0 | > 0.5 ✅ |

The 0.89 reward gap is the memory proof: a ~1.0 gap is only achievable if the recurrent policy is genuinely using its hidden state to recall the cue through the 60-tick silent period.

Artifacts on disk (`runs/phase2_memory_toy/`):
- `recurrent/ckpt_final.pt` — best-eval snapshot (eval@660 = −0.107)
- `feedforward/ckpt_final.pt` — best-eval snapshot (eval@420 = −0.995)
- `{recurrent,feedforward}/ckpt_{100..700}.pt` — periodic checkpoints

## Config change that landed this result

Relative to the config at the start of the session, the only changes in `experiments/configs/phase2_memory_toy.yaml` are:

```yaml
ppo:
  lr_schedule: cosine      # was: constant
  lr_final_ratio: 0.1      # was: 1.0
```

Cosine annealing from `3e-4 → 3e-5` over 700 updates tightened late-training oscillation substantially. Comparison against the earlier constant-LR diagnostic run (value-normalization + minibatch 16→4, but `lr_schedule: constant`):

| | constant LR run (diagnostic) | cosine LR run (this doc) |
|---|---|---|
| First eval ≤ −0.15 | ~update 430 | ~update 360 |
| Late-training eval band | oscillating −0.08 to −0.17 | tight: −0.107 to −0.135 |
| Best eval | −0.080 (update 560) | −0.107 (update 660) |
| Run completion | interrupted at update 660 | clean finish, 700/700 both variants |

Note that the diagnostic run reached a slightly lower single-eval low (−0.080), but oscillated widely and was never run to completion on the feedforward baseline. The cosine run trades a small amount of peak for much lower variance and a clean, committable artifact.

## What was also fixed in the session

The session produced several changes beyond the LR schedule. All are in commit `af4003c Big trainer refactor`:

1. **Package split.** `python/train/ppo_recurrent.py` (769 lines) → `python/train/ppo_recurrent/` package (7 files, behavior-identical).
2. **Value normalization.** Per-rollout return mean/std; value loss now computed in normalized space. Config flag `value_normalization: true` (default on).
3. **Smaller minibatches.** `minibatch_size: 16 → 4` to increase gradient update count per rollout.
4. **Per-head gradient instrumentation.** `actor_grad_norm`, `critic_grad_norm`, `trunk_grad_norm`, `terminal_adv_std`, `mean_log_std` now logged per update. These were the diagnostic signal that identified the value-scale + full-batch-update bottleneck in the pre-fix run.

The combination of (2) + (3) moved the run from "plateaus above the gate at best −0.835" to "clears the gate around update 360 and holds."

## Next step

Phase 2 is done. The natural next work is **Phase 3: wire the C++ sim into the recurrent PPO trainer** (`docs/rl_design.md` §6). The trainer code at `python/train/ppo_recurrent/` is designed to be env-agnostic and should transplant with an env-swap.
