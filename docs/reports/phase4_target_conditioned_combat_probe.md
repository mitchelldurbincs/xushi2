# Phase 4 Target-Conditioned Combat Probe

Date: 2026-05-15

## Question

Can an actor-side target-selection head improve Phase 4 weak_basic combat
composition without changing game rules, rewards, opponent, damage, round
length, action space, or observation space?

## Implementation

The probe adds an opt-in internal target-selection path for flat Phase 4 MAPPO
actors:

- `actor_target_selection_head`: three-way logits over enemy slots.
- Target-conditioned aim/fire heads: the model reconstructs the three enemy
  candidate positions from the ordered three-agent actor-observation batch,
  computes a soft selected relative position, and conditions the continuous and
  binary heads on selected-target visibility/confidence.
- Target-selection auxiliary loss: supervised toward the nearest visible enemy
  during BC and PPO.
- Eval combat instrumentation: per-checkpoint Team A/B hit/fire, visible-fire
  rate, nearest-visible aim error, target concentration entropy, and damage per
  fire command.

The run config is
`experiments/configs/phase4/probe/phase4_mappo_target_cond_v1.yaml`. It keeps
the aim-only checkpoint warm start, walk-and-shoot BC, weak_basic opponent,
LR `1e-6`, entropy `0.02`, `1000` damage, and `30s` rounds. To fit the
available interactive runtime, the BC collection batch and aim rehearsal batch
were set to `256` while preserving `500` BC steps.

## BC Gate Result

Command:

```bash
PYTHONPATH=python WANDB_MODE=disabled python/.venv/bin/python -m train.train --config experiments/configs/phase4/probe/phase4_mappo_target_cond_v1.yaml
```

BC training completed all `500/500` steps. Target-selection auxiliary accuracy
rose from `0.188` at step 1 to `0.829` at step 500, confirming the new head was
receiving and fitting the nearest-visible supervision.

Post-BC weak_basic eval:

- Mean reward: `+0.534`
- Wins/losses/draws: `0/0/50`
- Score: `0.00/0.00`
- Team A/B hit/fire: `0.0398/0.0635`
- Team A/B nearest-visible aim error: `1.300/0.624` radians

The configured BC combat gate was conjunctive:

- Team A hit/fire must be `> 0.025`
- Team A aim error must be `< 1.3` radians

Team A hit/fire passed, but Team A aim error was just above the strict
threshold despite printing as `1.300`. The gate failed, so PPO did not start.
Only `runs/phase4_mappo_target_cond_v1/mappo/ckpt_final.pt` was written; no
update-50 checkpoint exists for this probe.

## Behavioral Autopsy

The structural head improved the headline Team A hit/fire metric relative to
the recent replay range from `aim_target_bc_v1` (`0.0096-0.0219`), but did not
solve the combat composition problem:

- The policy still remained in the old scoreless draw basin against weak_basic.
- Team B retained a large combat quality advantage: hit/fire `0.0635` and aim
  error `0.624` rad versus Team A hit/fire `0.0398` and aim error about `1.3`
  rad.
- The new target-selection head learned its BC label, but learned slot choice
  did not translate into sufficiently accurate full-env aiming.
- Because the BC gate failed, extending to PPO would violate the experiment
  design and risk spending updates on another known-bad basin.

## Conclusion

`phase4_mappo_target_cond_v1` is falsified at the BC gate. The target head is
trainable and improved hit/fire versus previous replay autopsies, but it did
not reduce Team A aim error below the required `1.3` rad threshold or produce
any weak_basic score/win signal. Do not run PPO for this config without a new
reason to believe the aim-error gate is too strict or the target-position
conditioning should use richer observations.
