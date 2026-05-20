# Runtime Specs And Experiment Phases

Runtime behavior is selected by explicit specs, not by experiment phase numbers.

Phases remain useful experiment metadata: config organization, W&B tags, journal
lineage, phase-gate decisions, and progress reporting. A phase label may be
logged or saved in a checkpoint, but it should not be the reason an environment,
learner, observation shape, action head, replay path, or eval mode is selected.

Current runtime selection flows through `train.runtime_specs.resolve_runtime_spec`:

- `learner.kind` selects the learner family (`mappo`, `ppo_recurrent`,
  `scripted_determinism`).
- `env.kind` selects the task family (`mappo_match`, `ranger_duel`,
  `memory_toy`).
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
and `tests/test_mappo_public_api.py`. Long one-update and BC smoke paths remain
in legacy compatibility tests until they are split further; run them explicitly
when proving end-to-end training behavior.
