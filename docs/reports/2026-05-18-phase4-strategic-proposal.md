# Phase 4 Strategic Proposal: Breaking The Composition Stalemate

Date: 2026-05-18

## Executive Summary

Phase 4 is no longer blocked by a missing primitive skill. The evidence now
shows two separate working skills:

- Objective skill: `v6.5` reaches and contests the cap reliably, but produces
  scoreless draws because it has no decisive combat.
- Combat skill: `combat_1v1_v2` clears the simplified duel gate at
  `13.16/12` kills, but direct transfer into full 3v3 loses objective
  competence immediately.

The failure is composition. Every attempt that trained or transferred one
skill at a time destroyed, ignored, or overwhelmed the other:

- Standard full-env `walk_and_shoot` BC restored cap movement but erased the
  synthetic aim mapping.
- Aim-freeze BC preserved aim but froze too much of the shared actor pathway,
  leaving the full policy unable to contest weak_basic.
- Aim-target BC preserved synthetic aim better, but still produced only
  scoreless 3v3 draws.
- Target-conditioned combat improved hit/fire above the previous replay range,
  but did not reduce aim error or produce score.
- Weakening the bot to `weak_basic_v2` manufactured a greedy kill edge
  (`8-9` learner kills vs `1-2` bot kills) but still produced `0/0` score and
  `50/50` draws.
- Direct `combat_1v1` to 3v3 transfer collapsed into `0W/50L`, score
  `0/37`, kills `0/0`.

The next move should therefore stop asking PPO to discover the composition in
the full 3v3 environment. The recommended next strategy is a
multi-teacher composition rehearsal stage: initialize from the objective
checkpoint, distill combat aim/fire from the solved 1v1 checkpoint on combat
observations, and simultaneously preserve objective movement from the cap
checkpoint on full-env observations. Only if that BC gate preserves both
skills should PPO start.

## Evidence Synthesis

The strongest negative result is `weak_basic_v2`: the learner could kill the
heavily weakened objective-contesting bot more often in greedy eval but still
never converted kills into score. That means opponent weakness and raw kill
margin are not sufficient. The bottleneck has moved to timing and binding:
agents must aim/fire in a way that creates local majority, stay on or return
to the point during the advantage window, and avoid having movement updates
erase the combat behavior.

The strongest positive result is `combat_1v1_v2`: the same Phase 4 tensor
interface and MAPPO model can learn a real shoot/kill loop when the objective
and multi-target chaos are removed. The direct transfer failed because the
duel policy never learned cap approach, cap timing, or 3-agent objective
pressure. It fired in full 3v3, but produced almost no damage and gave up the
point.

The BC diagnostics explain why naive warm-starting fails. Full-env BC has been
acting like a destructive single-task optimizer: it is good at restoring the
old scoreless cap-and-spray basin, but it overwrites the isolated aim/combat
mapping. Freezing preserved aim, but because the shared trunk is load-bearing
for both movement and aim, it prevented the movement side from adapting back
to the full 3v3 setup.

The target-conditioned probe is also instructive. It used a nearest-visible
target label and improved hit/fire, but target concentration remained too
diffuse and aim error stayed around the gate. A target-selection head is not
useless, but nearest-visible per-agent supervision is not the same as team
focus fire or objective conversion.

## Strategy 1: Multi-Teacher Composition Rehearsal

### Mechanism

Create a BC/distillation stage whose job is explicitly to hold both known
skills in one policy before PPO begins:

- Student starts from the objective checkpoint:
  `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`.
- Objective teacher is the same v6.5 checkpoint. On full Phase 4
  `weak_basic_v2` or `walk_to_objective` observations, preserve movement and
  cap approach behavior.
- Combat teacher is
  `runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt`. On `combat_1v1`
  observations, preserve aim/fire behavior.
- Optional aim-only teacher remains available for a cheap aim-retention
  diagnostic, but it should not be the primary combat teacher because the
  current failure includes fire timing and repeated kill loops, not just
  visible-target aim.

The BC objective should be head-scoped:

- Movement loss on full-env objective samples: continuous rows `move_x`,
  `move_y`.
- Combat loss on `combat_1v1` samples: `aim_delta` and `primary_fire`.
- Rehearsal batches for both domains in every BC step.
- No standard `walk_and_shoot` aim label unless it is explicitly gated behind
  the combat teacher; that heuristic is already implicated in aim erasure.

This is not a reward-function change, game-rule change, action/obs-space
change, or MAPPO-core change. It is an opt-in pretraining/composition stage.

### Why Previous Approaches Failed Under This Lens

Direct v6.5 to `basic` or `weak_basic` asked a movement-only policy to learn
combat inside the hardest environment. It died, drew, or sprayed.

Aim-only transfer started from a combat/aim checkpoint but then standard
full-env BC erased the aim mapping. It optimized movement and heuristic fire
labels as a single task, so it restored the old draw basin.

Aim-freeze BC preserved aim but froze shared actor features that movement
needed. It proved preservation is possible, but not composition.

Direct `combat_1v1` transfer skipped the objective teacher entirely. It
preserved neither cap pathing nor point pressure, so full 3v3 became
uncontested objective loss.

Multi-teacher rehearsal directly targets that missing middle: preserve
movement and combat in the same model before any PPO gradient is allowed.

### Exact Code/Config Changes

Add an opt-in composition pretrain path, preferably in
`python/train/mappo_bc_pretrain.py` and orchestrated from
`python/train/mappo_eval_checkpoint.py`.

Concrete config fields:

```yaml
run:
  composition_pretrain: true
  composition_pretrain_steps: 1000
  composition_objective_teacher_checkpoint: runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt
  composition_combat_teacher_checkpoint: runs/phase4_mappo_combat_1v1_v2/mappo/ckpt_final.pt
  composition_objective_batch_size: 256
  composition_combat_batch_size: 256
  composition_objective_env:
    opponent_bot: weak_basic_v2
  composition_combat_env:
    mini_game: combat_1v1
    mini_game_config:
      episode_decisions: 64
```

New config:

```text
experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v1.yaml
```

Implementation details:

- Reuse `load_bc_aim_target_model` style checkpoint loading for frozen
  teachers.
- Reuse `Phase4Combat1v1MappoEnv` through the existing `env.mini_game:
  combat_1v1` registry.
- Add a `composition_rehearsal_pretrain(...)` helper that collects one
  objective batch and one combat batch per step, runs both teachers under
  `torch.no_grad()`, and applies masked behavior losses to the student.
- For continuous action distillation, use teacher mean action MSE or
  distribution KL for selected rows. Keep this simple first: MSE on normalized
  action means is enough for a BC gate.
- For `primary_fire`, use binary cross-entropy or Bernoulli KL against the
  combat teacher's probability.
- Log BC diagnostics: objective movement MSE, combat aim MSE, combat fire BCE,
  aim-only hits, combat_1v1 kills, full weak_basic_v2 wins/score/kills,
  hit/fire, visible-fire, aim error, and target concentration.
- Add focused tests in `python/tests/test_mappo_bc_freeze.py` or a new
  `python/tests/test_mappo_composition_rehearsal.py` that verify selected
  action rows are trained and unrelated rows are not accidentally included.

Do not run a long PPO job until the BC gate passes.

### Feasibility

Implementation in under 4 hours: likely yes, if kept to BC/distillation,
config, and focused tests. It reuses existing model loading, mini-game routing,
and eval diagnostics. The main risk is discovering that the existing model API
does not expose clean teacher action means for row-scoped losses; that is still
small-scope trainer work, not sim work.

### Probability Of Success

Estimated probability of producing a useful next signal: 60%.

Estimated probability of actually breaking Phase 4 into nonzero score/wins on
the first PPO probe: 25-35%.

This is higher than another opponent nerf because it directly attacks the
observed composition failure. It is still not high: the two teachers may encode
incompatible hidden-state/trunk features, and preserving both skill metrics may
still fail to produce objective timing.

### Falsification Criteria

Give up on this strategy if any of these occur:

- After composition BC, `combat_1v1` eval falls below `12` mean kills per
  64-decision episode.
- After composition BC, full weak_basic_v2 eval loses objective competence:
  `onpt < 0.25`, score against `0/7+`, or `50/50` losses.
- After composition BC, replay/eval combat diagnostics remain near the direct
  transfer failure: Team A hit/fire `<0.02` or aim error `>1.55` rad.
- If BC gates pass but update-50 PPO still has `0` Team A score, `0` wins, no
  hit/fire improvement over weak_basic_v2, and objective contact collapses,
  stop rather than extending to 500 updates.

## Strategy 2: Objective-Coupled Combat Micro-Curriculum

### Mechanism

Create a new Phase 4-compatible mini-game that includes both combat and
objective timing but removes 3v3 multi-target chaos. The current `combat_1v1`
mini-game proves aim/fire can be learned, but it contains no point-control
state. The missing rung is a synthetic or wrapped micro-fight:

- One active learner slot and one active enemy.
- Both spawn near or path toward the objective.
- Score ticks require the learner to kill, displace, or out-position the enemy
  and then remain on point briefly.
- The other two learner slots remain present in tensor shape but are inactive
  or scripted no-ops.
- Progression can then add a second active learner/enemy before returning to
  full 3v3.

This isolates the exact transition weak_basic_v2 exposed: kills must convert
into score.

### Why Previous Approaches Failed Under This Lens

`combat_1v1` solved pure combat but did not teach cap conversion.

`v6.5` solved cap contact but did not teach killing to create majority.

`weak_basic_v2` was still full 3v3, so even when the learner had a greedy kill
edge, timing and multi-target target selection prevented score. The learning
signal remained too entangled.

A micro-cap duel gives immediate feedback for the composition event: kill or
pressure the enemy, step onto point, score. It asks for one composition before
three simultaneous compositions.

### Exact Code/Config Changes

Add a new mini-game env:

```text
python/envs/phase4_cap_duel_mappo.py
```

Wire it into:

```text
python/envs/__init__.py
python/train/phases.py
python/tests/test_phase4_cap_duel_mappo.py
```

Config surface:

```yaml
env:
  mini_game: cap_duel
  mini_game_config:
    episode_decisions: 96
    enemy_hp: 3
    point_radius: 0.18
    score_ticks_to_clear: 12
    enemy_recontest_delay: 12
    hit_tolerance: 0.12
```

New configs:

```text
experiments/configs/phase4/probe/phase4_mappo_cap_duel_v1.yaml
experiments/configs/phase4/probe/phase4_mappo_cap_duel_transfer_v1.yaml
```

The synthetic env should preserve Phase 4 actor/critic/action tensor shapes
like `Phase4Combat1v1MappoEnv`. It should not modify simulator rules or the
canonical full-env reward. It is a curriculum diagnostic only.

### Feasibility

Implementation in under 4 hours: possible but tighter than Strategy 1. A
minimal synthetic env is straightforward because `combat_1v1` is already a
template, but getting the cap conversion signal shaped correctly and tested
will take care.

### Probability Of Success

Estimated probability of producing a useful next signal: 55%.

Estimated probability of first transfer producing full 3v3 score/wins:
20-30%.

This has a plausible mechanism, but it risks another synthetic-transfer gap:
the policy may solve the toy cap duel and still fail once three enemies are
present.

### Falsification Criteria

Give up or redesign if:

- The cap-duel env is not solved by 300 updates.
- A solved cap-duel checkpoint transferred to weak_basic_v2 produces update-50
  full 3v3 `0` score, `0` wins, no hit/fire improvement, and `onpt < 0.25`.
- The learned policy scores in cap-duel by exploiting synthetic dynamics that
  cannot appear in the real sim.

## Strategy 3: Team Focus-Fire Target Conditioning

### Mechanism

The previous target-conditioned head supervised each agent toward the nearest
visible target. That is not focus fire. With three agents, nearest-visible
labels naturally split attention across enemy slots and preserve the observed
target concentration entropy around `1.09`.

The next target-conditioning variant should supervise a shared team target
priority from actor-visible information:

- Prefer visible low-HP enemies.
- Break ties by center/objective proximity or deterministic slot order.
- Apply the same target label to all learner agents that can see that target.
- Condition aim/fire on that target.
- If the shared target is not visible to an agent, fall back to nearest visible
  but log the fallback rate.

This is still internal to the actor. It does not add a `target_slot` action to
Phase 4, change the observation, or change the simulator.

### Why Previous Approaches Failed Under This Lens

Per-action entropy increased exploration but did not create coordinated target
choice.

Invalid-fire masking did not matter because visible targets were almost always
present.

The target-conditioned probe learned its nearest-visible label but nearest
visible is the wrong team objective. It can improve individual hit/fire without
creating kill concentration or cap conversion.

Weak_basic_v2 showed that even an aggregate kill edge is not enough if kills do
not happen in a coordinated objective window. Focus-fire supervision tries to
compress damage onto one enemy quickly enough for score to start.

### Exact Code/Config Changes

Extend the existing opt-in target-conditioned path in:

```text
python/train/mappo_model.py
python/train/mappo_update.py
python/train/mappo_bc_pretrain.py
python/train/mappo_evaluate.py
```

Config fields:

```yaml
ppo:
  target_conditioned_combat: true
  target_selection_aux_coef: 0.5
  target_selection_label: team_focus_low_hp
  target_selection_visibility_fallback: nearest_visible
```

New config:

```text
experiments/configs/phase4/probe/phase4_mappo_focus_fire_v1.yaml
```

Instrumentation requirements:

- Log target concentration entropy per eval.
- Log same-target fraction across Team A slots when at least two agents have
  visible enemies.
- Log hit/fire, aim error, kills, score, and on-point contact.

### Feasibility

Implementation in under 4 hours: likely, because target-conditioned combat
already exists. The main work is replacing the label generator and adding the
team-level concentration metrics.

### Probability Of Success

Estimated probability of producing a useful next signal: 45%.

Estimated probability of breaking into full 3v3 score/wins first try: 15-25%.

This is a good follow-up if composition rehearsal preserves both skills but
still fails to concentrate damage. As a first next step it is weaker because it
does not directly solve the movement/combat checkpoint composition problem.

### Falsification Criteria

Give up on this variant if:

- BC target-selection accuracy rises but Team A hit/fire remains below
  `0.04` against `weak_basic_v2`.
- Target concentration entropy remains near `1.09` or same-target fraction
  does not improve over current diagnostics.
- Hit/fire and focus improve but score stays `0/0` through update 100; at that
  point the bottleneck is objective timing rather than target selection.

## Recommendation

Run Strategy 1 next: multi-teacher composition rehearsal.

Reasoning:

1. It is the most direct answer to the actual evidence. The project now has a
   cap teacher and a combat teacher. The failed approaches all trained from
   one side and hoped the other side would survive transfer. It did not.
2. It has a cheap pre-PPO falsification gate. We can test objective retention,
   combat retention, aim-only retention, and full `weak_basic_v2` first-eval
   behavior before spending a long training run.
3. It keeps changes out of load-bearing simulator, reward, observation, action,
   replay, and MAPPO core paths. The new work is an opt-in pretraining
   composition stage plus one probe config.
4. It produces clearer information than another full-env opponent or
   hyperparameter variant. If it cannot keep both teachers alive in one model,
   the next architecture decision is justified. If it can keep both teachers
   alive but still cannot score, Strategy 2 or a focus-fire variant becomes the
   next narrow target.

The recommended task should implement only the BC/distillation and diagnostics
first. Do not start a training run as part of that implementation card unless
the card is explicitly expanded later.
