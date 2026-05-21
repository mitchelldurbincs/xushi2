# Phase 4 Cap-Duel Result - 2026-05-21

## Scope

This result records the cap-duel escalation from `GOAL_INSTRUCTIONS.md`: build a Phase 4-compatible `cap_duel` mini-game, train it with current-vs-current self-play warm-started from `phase4_mappo_basic_v6_5`, then probe transfer into full 3v3 against `weak_basic_v2`.

No commit was created.

## Stage 1 - Cap-Duel Env

Implemented `python/envs/phase4_cap_duel_mappo.py` and routing for the mini-game, including current-policy self-play support while preserving the Phase 4 actor/action tensor surface.

Verification:

- `py -3.13 -m pytest tests/test_phase4_cap_duel_mappo.py tests/test_phase4_combat_1v1_mappo.py tests/test_phase4_mappo_env.py tests/test_phase4_current_selfplay.py tests/test_mappo_matrix_eval.py -q` - passed, `56 passed`
- `py -3.13 -m scripts.check_import_boundaries` - passed

## Stage 2 - Cap-Duel Self-Play

Config: `../experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v1.yaml`

Run identity:

- Git commit: `c562fcf7b571b837b167e6195eecf2297fc8c0f9`
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/64zvtdgr
- Output: `runs/phase4_mappo_cap_duel_selfplay_v1/`
- Evidence: `runs/phase4_mappo_cap_duel_selfplay_v1/evidence.json`
- Gate: `runs/phase4_mappo_cap_duel_selfplay_v1/gate_decision.json`

Gate status: `HUMAN_INSPECTION_REQUIRED`. Objective checks passed:

- `cap_duel_score=9.02 >= 6.0`
- `cap_duel_kills=56.0 >= 5.0`
- `cap_duel_wins=32 >= 25`

Replay artifacts:

- `../data/replays/phase4_cap_duel_selfplay_v1_ckpt_final_greedy.replay`
- `../data/replays/phase4_cap_duel_selfplay_v1_ckpt_final_stochastic.replay`

Remaining subjective question: does the greedy replay show kill/displace-then-hold behavior, rather than just trading fire?

## Stage 3 - 3v3 Transfer Probe

Config: `../experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml`

Run identity:

- Git commit: `c562fcf7b571b837b167e6195eecf2297fc8c0f9`
- Seed: `3519994490`
- W&B: https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/s02dhwwd
- Output: `runs/phase4_mappo_cap_duel_transfer_v1/`

Gate status: `NOT_REACHED`. The run was intentionally stopped after the update-50 eval because the Stage 3 decision rule fired: transfer eval still scored 0 by update 50.

Observed transfer eval:

- Update 25: `0W/0L/50D`, score `0.00/0.00`, kills `0.0/2.0`, hit_fire `0.0161/0.2111`, majority seconds `0.00/56.00`
- Update 50: `0W/0L/50D`, score `0.00/0.00`, kills `0.0/2.0`, hit_fire `0.0145/0.2111`, majority seconds `0.00/56.00`, damage from fire `14.5/211.1`

No Stage 3 evidence, gate artifact, matrix eval, or replay was produced because the decision rule stopped the run before final checkpointing.

## Decision

The cap-duel rung is a Stage 2 objective success but not a Phase 4 transfer success. The learned duel behavior did not survive the full 3v3 reward gradient against `weak_basic_v2`.

Recommended next escalation: Strategy 3 focus-fire target conditioning, or a composition rehearsal using the new cap-duel checkpoint as the available teacher. Both are outside this loop and require explicit approval before further code-level or training-plan changes.
