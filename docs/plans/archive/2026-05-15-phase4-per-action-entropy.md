# Phase 4 Per-Action Entropy Plan

## Context

Phase 4 remains in the Escape Protocol draw basin after the isolated auxiliary
aim-head probe. That probe learned its supervised aim target but did not change
PPO behavior: update 500 stayed at 0/50 wins, 50/50 draws, score 0/0, kills
1.0/6.0.

Do not create another Phase 4 hyperparameter or opponent variant from this
plan. This plan implements Escape Protocol Section 5.2 as one isolated
architecture/training-loss change.

## Goal

Add opt-in per-action-type entropy coefficients for MAPPO so movement, aim, and
binary action exploration can be weighted independently.

## Scope

- Preserve existing `ppo.entropy_coef` behavior when no per-action coefficients
  are configured.
- Add optional `ppo.entropy_coef_move`, `ppo.entropy_coef_aim`, and
  `ppo.entropy_coef_binary` parsing for MAPPO configs.
- Split entropy logging into movement, aim, and binary components.
- Use per-action coefficients only in the entropy bonus term; do not change
  observations, actions, rewards, sim rules, replay format, or existing config
  defaults.
- Add focused tests for entropy splitting, config parsing, and update metrics.

## Non-Goals

- No new Phase 4 experiment config in this implementation task.
- No long PPO benchmark run.
- No combination with auxiliary aim, action masks, aim-only curriculum, or reward
  changes.

## Verification

- `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py tests/test_mappo_loss_mask.py -q`
- `cd python && .venv/bin/ruff check train/ppo_recurrent/losses.py train/mappo_model.py train/mappo_rollout_trainer.py tests/test_mappo_aux_aim.py`
- `cd python && .venv/bin/pytest tests/test_phase_registry.py::test_phase4_smoke_config_builds_mappo_config tests/test_mappo_warm_start.py::test_mappo_warm_starts_from_init_checkpoint -q`

## Implementation Result

Implemented the opt-in code path:

- `ppo.entropy_coef_move`, `ppo.entropy_coef_aim`, and
  `ppo.entropy_coef_binary` are parsed into `MappoConfig`.
- Existing behavior is preserved when those fields are omitted: the trainer uses
  `ppo.entropy_coef * total_entropy`.
- When any per-action coefficient is set, the entropy bonus is
  `move_coef * entropy_move + aim_coef * entropy_aim + binary_coef *
  entropy_binary`, with any target/other entropy still weighted by the base
  `entropy_coef`.
- PPO update metrics now include `entropy_move`, `entropy_aim`,
  `entropy_binary`, `entropy_other`, and `entropy_bonus`.
- Focused tests and ruff passed using the commands above.

## Follow-Up Gate

Only after this code path passes focused verification should a separate Phase 4
probe config be created. That config must include `metadata.hypothesis`,
`metadata.falsification_criteria`, `metadata.max_updates_if_no_signal`, and a
cheap diagnostic or unit-level evidence that the entropy bonus is actually
using the per-action coefficients.

## Probe Result

`phase4_mappo_per_action_entropy_v1` ran to the configured 500-update stop
point. The separate entropy path was active at update 500
(`entropy_move=1.2543`, `entropy_aim=0.6406`, `entropy_binary=0.0189`,
`entropy_bonus=0.1221`) and `action_binary_mean` stayed near `0.33`, but final
eval remained 0/50 wins, 50/50 draws, score 0/0, kills 6.0/5.0.

Stochastic replay was dumped to
`data/replays/phase4_per_action_entropy_v1_ckpt0500_stochastic.replay`.

Conclusion: the isolated Escape Protocol 5.2 per-action entropy intervention is
falsified as a Phase 4 fix. Do not continue with coefficient variants.
