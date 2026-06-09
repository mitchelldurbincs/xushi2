# Documentation Index

This index groups project documentation by purpose.

## How to use this index

- **Stable specs/design docs**: Read these first to understand interfaces, invariants, and architectural intent.
- **Active execution plans**: Use these during implementation to track current Phase 4 work and near-neighbor phase transitions.
- **Archived results**: Use these to review outcomes, regressions, and prior phase decisions.

## Specs

- [`observation_spec.md`](./observation_spec.md): Actor/critic observation contracts and feature layout.
- [`action_spec.md`](./action_spec.md): Action space definition and action semantics.
- [`replay_format.md`](./replay_format.md): Replay file structure and serialization details.
- [`determinism_rules.md`](./determinism_rules.md): Determinism invariants and guardrails.

## Design

- [`game_design.md`](./game_design.md): Gameplay goals and simulator design constraints.
- [`rl_design.md`](./rl_design.md): RL stack architecture and phase progression design.
- [`coding_philosophy.md`](./coding_philosophy.md): Coding principles used across the repo.

## Architecture

- [`architecture/python_layers.md`](./architecture/python_layers.md): Python package layering contract (`xushi2` → `envs` → `train`) and import-direction rules; enforced by `python/scripts/check_import_boundaries.py`.

## Standards

- [`standards/cpp_determinism_checklist.md`](./standards/cpp_determinism_checklist.md): C++ determinism checklist for simulator/runtime changes.
- [`standards/python_training_checklist.md`](./standards/python_training_checklist.md): Python training checklist for RL pipeline and experiment changes.
- [`standards/cpp_benchmarking_guide.md`](./standards/cpp_benchmarking_guide.md): Benchmark build/run standards and determinism guardrails for C++ performance work.

## Plans

- [`plans/README.md`](./plans/README.md): Plan workflow plus links to active and archived plans.

### Active (Phase 4 execution and adjacent phases)

`plans/README.md` lists the small set of canonical drivers; the entries below are the full set of active plan files for discoverability.

- [`plans/active/2026-04-21-memory-toy-plan.md`](./plans/active/2026-04-21-memory-toy-plan.md): Memory toy environment plan.
- [`plans/active/2026-04-22-ppo-recurrent-split.md`](./plans/active/2026-04-22-ppo-recurrent-split.md): Recurrent PPO module split plan.
- [`plans/active/2026-04-22-sim-cpp-modularization.md`](./plans/active/2026-04-22-sim-cpp-modularization.md): C++ sim modularization plan.
- [`plans/active/2026-04-24-phase3-to-phase4-cleanup.md`](./plans/active/2026-04-24-phase3-to-phase4-cleanup.md): Phase boundary cleanup before Phase 4 work.
- [`plans/active/2026-04-24-phase4-prep.md`](./plans/active/2026-04-24-phase4-prep.md): Phase 4 preparation checklist and sequencing.
- [`plans/active/2026-04-24-spec-drift-audit.md`](./plans/active/2026-04-24-spec-drift-audit.md): Spec-vs-implementation drift audit.
- [`plans/active/2026-05-07-phase4-mappo-env-design.md`](./plans/active/2026-05-07-phase4-mappo-env-design.md): Phase 4 MAPPO environment design plan.
- [`plans/active/2026-05-07-phase4-mappo-env-implementation.md`](./plans/active/2026-05-07-phase4-mappo-env-implementation.md): Phase 4 MAPPO environment implementation plan.
- [`plans/active/2026-05-07-phase4-critic-obs-design.md`](./plans/active/2026-05-07-phase4-critic-obs-design.md): Phase 4 critic-observation design plan.
- [`plans/active/2026-05-07-phase4-critic-obs-implementation.md`](./plans/active/2026-05-07-phase4-critic-obs-implementation.md): Phase 4 critic-observation implementation plan.
- [`plans/active/2026-05-08-phase4-cap-training-escalation-design.md`](./plans/active/2026-05-08-phase4-cap-training-escalation-design.md): Phase 4 curriculum/cap escalation design.
- [`plans/active/2026-05-08-phase4-cap-training-escalation.md`](./plans/active/2026-05-08-phase4-cap-training-escalation.md): Phase 4 curriculum/cap escalation execution plan.
- [`plans/active/2026-05-08-team-spirit-per-agent-rewards.md`](./plans/active/2026-05-08-team-spirit-per-agent-rewards.md): Team-spirit / per-agent reward shaping plan.

## Results

- [`plans/archive/`](./plans/archive/): Archived execution/result notes by phase and probe.

## Journal

- [`journal/reinforcement_learning_journal.md`](./journal/reinforcement_learning_journal.md): Running RL experiment journal and iteration notes.
