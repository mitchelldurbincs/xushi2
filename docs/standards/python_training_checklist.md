# Python Training Checklist (Tier 2)

Operational checklist extracted from `docs/coding_philosophy.md` for training/eval code review.

## Scope

- `python/train/`
- `python/eval/`
- `python/xushi2/`

## Must

### Reproducibility and experiment traceability

- **Must** log seeds for training/eval runs and persist them with run artifacts.
- **Must** keep checkpoint loading/saving reproducible (model, optimizer, scheduler, RNG state when applicable).
- **Must** export replay artifacts for every eval game so behavior is reconstructable.

### Tensor and numeric safety

- **Must** validate tensor shape/dtype/device at key function boundaries (policy/value forward, loss assembly, env batch IO).
- **Must** check for NaN/Inf in observations, actions, advantages/returns, losses, and gradients (or explicit equivalent safeguards).
- **Must** fail loudly on invalid numerics; no silent clipping that hides corruption without logging.

### Information-boundary correctness

- **Must** preserve strict actor/critic observation separation.
- **Must** prevent hidden-enemy leakage in Python-side observation builders and wrappers.
- **Must** keep action canonicalization/packing aligned with `action_spec.md` expectations.

## Should

- **Should** centralize validation utilities so checks are consistent across train/eval entry points.
- **Should** emit concise per-step/per-epoch diagnostics that make divergence root-causeable.
- **Should** include smoke tests for observation schema and action boundary behavior when touching wrappers/builders.

## Used in PR review

For PRs touching Tier 2 Python paths:

- Reviewer **must** confirm every “Must” item remains true or is explicitly deferred with justification.
- Author **should** include a “Training checklist notes” section in PR description summarizing:
  - seed/reproducibility handling,
  - NaN/Inf and boundary checks,
  - actor/critic separation impact,
  - replay export behavior and tests run.
- If a “Must” cannot be met in the same PR, reviewer **must** require a follow-up issue/PR link before approval.
