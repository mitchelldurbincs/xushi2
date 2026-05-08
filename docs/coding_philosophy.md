# Xushi2 — Coding Philosophy

> **Xushi2 code should be boring, bounded, explicit, deterministic, and easy to falsify.**
> **The simulation is critical code. The viewer, tools, and trainer are clients.**

This document is the **intent/rationale** layer. Operational enforcement lives in checklist docs:

- C++ determinism and sim-boundary review checklist: `docs/standards/cpp_determinism_checklist.md`
- Python training/eval review checklist: `docs/standards/python_training_checklist.md`

Related specs remain authoritative for behavior and interfaces:

- `docs/determinism_rules.md`
- `docs/action_spec.md`
- `docs/observation_spec.md`
- `docs/replay_format.md`
- `docs/rl_design.md`

## Purpose

Xushi2 is a deterministic 3v3 control-point sim used as a multi-agent RL environment. At all times, we optimize for three invariants:

1. Same seed + same canonical action stream => bit-identical trajectory (same machine, same binary).
2. Actor observations contain no hidden enemy state.
3. Replay files reconstruct matches exactly, with golden replay CI guarding drift.

## Tiered strictness

Rules apply by codebase tier, not uniformly:

- **Tier 0 (sim core/boundaries):** strict determinism discipline, explicit validation, replay safety.
- **Tier 1 (viewer/tools/bots):** practical ergonomics, but never at the cost of authoritative sim correctness.
- **Tier 2 (Python train/eval):** practical ML workflows with strict reproducibility, numeric hygiene, and no information leakage.

Use the checklist docs above during PR review to apply concrete must/should gates per tier.
