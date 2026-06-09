# Runtime Specs And Experiment Phases

Runtime behavior is selected by explicit specs, not by experiment phase numbers.

Phases remain useful experiment metadata: config organization, W&B tags, journal
lineage, phase-gate decisions, and progress reporting. A phase label may be
logged or saved in a checkpoint, but it should not be the reason an environment,
learner, observation shape, action head, replay path, or eval mode is selected.

Current runtime selection flows through `train.runtime_specs.resolve_runtime_spec`:

- `learner.kind` selects the learner family (`mappo`,
  `scripted_determinism`). The single-agent `ppo_recurrent` family was
  removed in the 2026-06 cleanup.
- `env.kind` selects the task family (`mappo_match` is the only explicit
  kind; `ranger_duel` and `memory_toy` were removed with the single-agent
  pipeline).
- Env capabilities select behavior: `actor_obs`, `critic_obs`, `team_size`,
  `opponent`, `features.fog`, `features.map_randomization`,
  `features.target_slot`, snapshot/self-play config, and mini-game config.
- `train.checkpoint_runtime.checkpoint_runtime` normalizes checkpoint config
  shapes before replay and matrix eval reconstruct runtime behavior.

Legacy phase configs still adapt through `train.phases` for current continuity,
but that module is a compatibility adapter. New production code should not
import `train.phases` or branch on numeric phase comparisons. The boundary
checker enforces that:

```bash
cd python
python -m scripts.check_import_boundaries
```

Explicit runtime YAMLs live under `experiments/configs/runtime/`. They may keep
`experiment.phase` metadata when a run belongs to a phase gate, but they should
not need a top-level `phase` field for dispatch.

Known temporary residual: `train.phases` still contains the legacy
`PHASE_REGISTRY` and env-bundle construction for existing phase configs. It is
kept only as a current-config compatibility adapter. New runtime behavior should
be added to explicit runtime specs and `envs.runtime_factory`, not to
`PHASE_REGISTRY`.

Slow smoke tests are intentionally separated at verification time. Fast runtime
unit coverage lives in `tests/test_runtime_specs.py`, `tests/test_train_dispatch.py`,
and `tests/test_mappo_public_api.py`. End-to-end training smoke coverage lives
in `tests/test_mappo_phase4_smoke.py`, while the heavier cross-phase BC probe
sweep lives in `tests/test_mappo_bc_probe_smoke.py` and is marked `bc_probe`.
Run those explicitly when proving end-to-end training behavior.
