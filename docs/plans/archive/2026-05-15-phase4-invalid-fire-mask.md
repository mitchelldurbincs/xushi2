# Phase 4 Invalid Fire Mask Plan

## Context

The Escape Protocol Section 5.1 auxiliary aim head and Section 5.2 per-action
entropy probes both failed to produce score. The policies fire almost
continuously and move while firing, but still remain in 50/50 draw outcomes.

Do not create another Phase 4 hyperparameter or opponent variant from this
plan. This plan implements Escape Protocol Section 5.3 as one isolated
training/action-distribution change.

## Goal

Add an opt-in primary-fire action mask that suppresses `primary_fire` when the
actor observation has no currently visible/alive enemy.

## Scope

- Add an opt-in MAPPO config flag for invalid-fire suppression.
- Apply the mask in stochastic sampling, greedy action selection, and PPO
  log-prob/entropy recomputation.
- Derive the mask only from actor-observation fields already available to the
  policy, preserving actor/critic leak boundaries.
- Log the fraction of valid fire timesteps for diagnostics.
- Add focused tests for config parsing, masked action sampling, greedy masking,
  and PPO update metrics.

## Non-Goals

- No sim-rule, reward, observation, action-schema, replay-format, or bot changes.
- No new Phase 4 experiment config in this implementation task.
- No combination with per-action entropy coefficient changes or auxiliary aim.

## Verification

- `cd python && .venv/bin/pytest tests/test_mappo_aux_aim.py tests/test_mappo_loss_mask.py -q`
- `cd python && .venv/bin/ruff check train/mappo_model.py train/mappo_rollout_trainer.py tests/test_mappo_aux_aim.py`
- `./build/tests/test_actor_leak && ./build/tests/test_actor_obs && ./build/tests/test_critic_obs && ./build/tests/test_obs_dims`
- `cd python && .venv/bin/pytest tests/test_obs_manifest.py -q`

## Implementation Result

Implemented the opt-in code path:

- `ppo.mask_fire_when_no_visible_enemy` parses into `MappoConfig`.
- Masking uses only flat actor observation fields already available to the
  actor: `enemy_alive` and nonzero `enemy_relative_position`.
- Stochastic sampling, greedy action selection, and PPO log-prob/entropy
  recomputation all apply the same primary-fire mask.
- Rollout metrics include `fire_valid_fraction` when the mask is enabled.
- Focused tests, ruff, and actor/critic leak coverage passed using the commands
  above.

No Phase 4 experiment config was created in this implementation task.

## Probe Result

`phase4_mappo_invalid_fire_mask_v1` ran to the configured 500-update stop
point. Final eval was `0/50` wins, `0/50` losses, `50/50` draws, score `0/0`,
and kills `5.0/5.0`.

The stochastic replay was dumped to
`data/replays/phase4_invalid_fire_mask_v1_ckpt0500_stochastic.replay`.
Replay action stats show Team A still fires almost continuously
(`primary_fire` rate `0.9987`) and moves while firing (`0.982`). The rollout
metric `fire_valid_fraction` was `0.9994`, so the mask was nearly always open
and did not concentrate fire gradients enough to improve score conversion.

Conclusion: Escape Protocol 5.3 is falsified as an isolated Phase 4 fix. Do
not queue mask variants.
