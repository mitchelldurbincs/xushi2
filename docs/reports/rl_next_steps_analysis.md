# RL Next Steps Analysis: Phase 3 to Phase 4 MAPPO

## Executive Summary

Phase 3 worked because it separated objective learning from combat learning. The successful recipe was:

1. Learn cap-holding against `walk_to_objective` with `kill_bonus: 0.0`, `score_per_second: 0.05`, `time_penalty_per_second: 0.05`, `entropy_coef: 0.005`, and about 1500 updates.
2. Warm-start into a `basic` opponent stage, where the policy already has a cap behavior worth defending and survival pressure can teach shooting.

Phase 4 copied the first half well enough to recover objective movement, but the second half does not transfer cleanly from 1v1 to 3v3. In 3v3, contesting the cap is not enough. A team must clear enough enemies to create a majority before score appears. That makes the bridge from "walk to cap" to "kill then score" much wider than Phase 3.

The repeated failures are not random hyperparameter noise. They point to a specific gap:

```text
Phase 3 success: cap behavior + 1 enemy to clear + adversarial pressure can teach combat
Phase 4 failure: cap behavior + 3 enemies to clear + PPO cannot discover/retain coordinated combat before collapse or timeout
```

The strongest current evidence is from the v7 line:

- v6.5 reproduction: stable objective contesting (`onpt=0.52`, `dist=0.13`) but `50/50` draws, score `0/0`, no meaningful kills.
- Direct `basic` transition at 2500 damage: `50/50` losses, score `0.00/7.00`, Team A kills `0.0`, bot kills `27.0`.
- BC-with-firing at 1000 damage: BC policy alone drew `50/50`, but PPO at `5e-5` collapsed to `50/50` losses by update 50.
- Conservative PPO at `1e-6`, entropy `0.02`, 1000 damage preserved the draw basin for 450 updates, but produced zero wins and zero score.

If `phase4_mappo_basic_v7_basic_reduced_bc_v7.yaml` also fails, the next changes should stop trying to tune one scalar at a time and instead either create a decisive but easier combat teacher or change the learning decomposition so combat is learned before it is required for 3v3 objective control.

## Evidence Base

Primary source: `docs/journal/reinforcement_learning_journal.md`.

Supporting analysis:

- `docs/reports/v7_holdshoot_failure_analysis.md`
- `docs/reports/v7_500dmg_stalemate_analysis.md`
- `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v2.yaml` through `phase4_mappo_basic_v7_basic_reduced_bc_v7.yaml`

The relevant journal entries are:

- 2026-05-08 Phase 4 summary
- 2026-05-14 v6_5 reproduction
- 2026-05-14 v7_holdshoot
- 2026-05-14 v7_basic_reduced
- 2026-05-14 v7_basic_reduced_bc through v7_basic_reduced_bc_v6

## Phase 3 Success vs Phase 4 Failure

### What Phase 3 Proved

The journal says Phase 3 did not learn shooting from a random start against a firing opponent. It succeeded because the curriculum staged the problem:

- First stage: `walk_to_objective`, no combat reward pressure, `kill_bonus: 0.0`, objective-first shaping.
- Second stage: warm-start into `basic`, where survival pressure under fire could teach aim/fire timing.

The key mechanism was not "PPO can discover shooting from scratch." It was "PPO can refine combat once the policy already has a useful objective habit."

### What Phase 4 Copied Correctly

Phase 4 v6/v6.5 correctly recreated the cap-learning stage:

- `phase4_mappo_basic_v6.yaml`: `opponent_bot: walk_to_objective`, `kill_bonus: 0.0`, `score_per_second: 0.05`, `entropy_coef: 0.005`.
- `phase4_mappo_basic_v6_5.yaml`: added `time_penalty_per_second: 0.05`.

The journal records that mirroring `phase3_ranger_objective_curriculum_warmstart_v3.yaml` produced the first non-loss Phase 4 behavior: `50/50` draws, `onpt ~= 0.30`, `dist ~= 0.20`, agents reached and contested the cap, sometimes killed enemy bots, and never died.

The clean reproduction on `origin/main` commit `fe15b3e`, config `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v6_5.yaml`, seed `0xD1CEDA7A`, W&B run `9s4era9p`, ended with:

- `50/50` draws
- `0` wins, `0` losses
- `mean_reward = -0.387`
- `onpt = 0.52`
- `dist = 0.13`
- score `0/0`
- no eval kills of significance

So locomotion and cap contesting are not the bottleneck anymore.

### The Specific New Gap In Phase 4

The Phase 4 gap is the kill-and-score transition under 3v3 majority control.

In 1v1, a single kill or a single positional win can open scoring. In 3v3, standing on the point while all three enemies are also present produces a deny-stalemate. To score, the learner must either:

- kill enough enemies to create a local majority,
- push enemies off the objective,
- coordinate focus fire well enough that deaths convert into cap time,
- or exploit a weakened scripted opponent.

Current policies can reach the point. They cannot convert point contact plus firing into a kill advantage and then into score. Every later v7 result is a variant of this failure:

- too lethal: agents die before learning combat,
- too soft: everyone contests forever,
- stationary shooter: agents avoid cap because no objective pressure exists,
- conservative PPO: preserves draws but cannot escape them,
- longer rounds: gives the stronger bot more time to win.

## Config Progression And Failure Patterns

### Early v2-v5: Basic Opponent Was Too Hard From Random Or Weak Warm-Starts

| Config | Intervention | What It Tested | Result Pattern |
| --- | --- | --- | --- |
| `phase4_mappo_basic_v2.yaml` | `basic` with objective-probe shaping: `distance=0.05`, `on_point=0.02` | Is stronger objective shaping enough? | No. The design doc reports earlier `phase4_mappo_basic.yaml` and 3x shaping still produced `mappo_final=-11.000`, `0/10` wins, score `0/7`; shaping magnitude alone was not root cause. |
| `phase4_mappo_basic_v3.yaml` | v2 + walk-to-objective BC pretrain | Does cap-pointed initialization solve `basic`? | No durable combat learning. Walk-only BC teaches movement, not shooting. |
| `phase4_mappo_basic_v4.yaml` | warm-start from `v3_noop` cap policy into `basic` | Does a no-fire cap holder survive the adversarial push? | No. Config comments record the intended mechanism; later journal evidence shows cap holders get punished before combat emerges. |
| `phase4_mappo_basic_v5.yaml` | warm-start from v4, add damage-dealt shaping | Can damage reward teach aim? | No. Journal conclusion: damage-dealt shaping cannot bootstrap aim from random because it pays only after hits already happen. |

The early lesson: more objective shaping, BC movement, and hit rewards are not enough when the policy does not already sample useful firing and aiming under survivable pressure.

### v6/v6.5: Objective Curriculum Worked, But Only To Draws

| Config | Intervention | Metrics / Result | Interpretation |
| --- | --- | --- | --- |
| `phase4_mappo_basic_v6.yaml` | Phase 3-style objective stage: `walk_to_objective`, `kill_bonus=0.0`, `score_per_second=0.05`, `entropy=0.005` | Config comment says v6 plateaued around update 150 with `onpt~0.025`, score `0/7`, `0/50` wins. | Objective stage still hit deny-stalemate. |
| `phase4_mappo_basic_v6_5.yaml` | v6 + `time_penalty_per_second=0.05` | Earlier journal: transient `50/50` wins at update 1325, `mean_reward=+9.388`, score `1.00/0.00`. Reproduction: `50/50` draws, `mean_reward=-0.387`, `onpt=0.52`, `dist=0.13`. | Time penalty mattered. It recovered contesting and maybe transient wins, but did not reliably produce a scoring checkpoint. |

The important lever here was `time_penalty_per_second`. It broke the "stay home vs contest cap both near zero" equilibrium better than raw distance/on-point shaping.

But the reproduction also falsified the assumption that v6.5 reliably produced a best winning checkpoint. The run saved `ckpt_final.pt`, no `ckpt_best.pt` was found, and the final policy was a draw policy.

### v7 Hold-And-Shoot: Opponent Design Was Wrong

| Config | Change | Result |
| --- | --- | --- |
| `phase4_mappo_basic_v7_holdshoot.yaml` | warm-start v6.5 into stationary `hold_and_shoot`, damage 7500, `distance=0.01`, `on_point=0.0` | `onpt` dropped `0.041 -> 0.000` by update 500, `dist` rose `0.30 -> 0.66`, Team A kills stayed `0.0`, all `50/50` draws, score `0/0`, `mean_reward` rose to `+1.000`. |
| `phase4_mappo_basic_v7_holdshoot_v2.yaml` | add stronger cap shaping: `distance=0.05`, `on_point=0.02` | Existing report: still `onpt -> 0.000`, `dist ~= 0.5`, score `0/0`, `mean_reward +1.000`, all draws. |
| `phase4_mappo_basic_v7_holdshoot_v3.yaml` | v2 + reduce damage 7500 to 2500 | Stopped update 100: `onpt=0.092`, `dist=0.312`, score `0/0`, `mean_reward +0.998`, draws `50/50`, kills `0.0/17.0`. |

Pattern: `hold_and_shoot` decouples combat from objective control. Because the opponent does not contest the cap, the learner can avoid cap danger, remain closer to objective than the spawn-camped opponent, collect differential shaping, and timeout. The `+1.000` mean reward was misleading because shaped reward was saturating near the per-agent equivalent of the team clip, not because the policy was winning.

What mattered:

- Opponent objective pressure mattered a lot.
- `on_point=0.02` and `distance=0.05` did not fix the basin.
- Lowering damage did not fix a reward/opponent design where avoidance was optimal.

### v7 Basic-Reduced Line: Damage, BC, LR, And Round Length

| Config | Change | Metrics / Result | Lesson |
| --- | --- | --- | --- |
| `phase4_mappo_basic_v7_basic_reduced.yaml` | `basic`, warm-start v6.5, damage 2500, LR `5e-5` | Updates 50-200: `0/50` wins, `50/50` losses, score `0.00/7.00`, kills `0.0/27.0`, `mean_reward=-11.000`; `onpt` only crept `0.002 -> 0.049`. | 2500 damage is still too lethal for no-combat cap policy. |
| `phase4_mappo_basic_v7_basic_reduced_bc.yaml` | add 500-step walk-only BC | BC loss `0.2325 -> 0.0004`; still updates 50-200 `50/50` losses, score `0/7.00`, kills `0.0/27.0`, `bin=0.000-0.001`. | Walk-only BC re-anchors locomotion but leaves fire dead. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v2.yaml` | damage 500, walk-only BC | `50/50` draws, score `0/0`, `onpt=0.58-0.67`, `dist=0.14-0.18`, `bin=0.000`, bot kills `0-6`. | Low damage preserves cap behavior but removes combat pressure. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v3.yaml` | BC variant `walk_and_shoot`, damage 500 | `bin=0.323-0.333` through 425 updates; all draws; score `0/0`; kills mostly `0-3` per 50 episodes. | Firing BC worked, but 500 damage made combat non-decisive. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v4.yaml` | damage 1000, BC firing, LR `5e-5` | BC eval: `50/50` draws, score `0/0`, `mean_reward=-0.686`; PPO update 50+: `50/50` losses, bot score `2.30-6.03`, bot kills `12-15`, Team A kills `0.0`, `bin=0.333`. | 1000 damage is viable for BC, but PPO step size collapsed the policy. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v5.yaml` | LR `1e-6`, entropy `0.02`, damage 1000 | Updates 50-450: zero losses, zero wins, all draws, score `0/0`, kills trade `2-8`, `bin~0.333`, `onpt=0.23-0.46`. | Conservative PPO preserved the draw policy but could not escape it. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v6.yaml` | 60s rounds, LR `2e-6`, damage 1000 | BC eval already lost: score `0/7.53`, `mean_reward=-11.000`; PPO bot score climbed `7.50 -> 25.33`, Team A kills `4.0 -> 0.0`. | Longer rounds favor the better scripted combatant. |
| `phase4_mappo_basic_v7_basic_reduced_bc_v7.yaml` | queued/current: 30s, bot fire cooldown 30, LR `2e-6`, entropy `0.02`, damage 1000 | Not yet recorded in journal. | Tests whether weakening bot DPS breaks the 30s draw equilibrium without giving the bot 60s to dominate. |

## What Actually Mattered

### High-Impact Levers

1. **Curriculum staging mattered most.** The only productive Phase 4 behavior came from Phase 3-style staging: objective first, combat later.

2. **Opponent objective pressure mattered.** `hold_and_shoot` failed because it shoots without contesting. `basic` is harsher, but at least it creates a real score race.

3. **Time penalty mattered during objective learning.** v6.5's `time_penalty_per_second=0.05` is the clearest lever that improved cap contesting and broke the no-progress plateau.

4. **BC firing mattered for the binary head.** Walk-only BC gave `bin=0.000`; `walk_and_shoot` BC gave stable `bin~0.33`.

5. **PPO learning rate mattered after BC firing.** At 1000 damage, LR `5e-5` destroyed a draw-capable BC policy within 50 updates. LR `1e-6` preserved it for 450 updates.

6. **Opponent combat strength mattered more than round length.** 60s rounds did not help the learner convert kills into score; they gave the better-aiming bot enough time to win decisively.

### Medium-Impact Or Context-Dependent Levers

1. **Damage mattered, but only as an interaction.** 2500 was too lethal, 500 too soft, 1000 produced a viable BC draw. Damage alone did not create wins.

2. **Entropy helped preservation/exploration tradeoff but did not solve the objective.** Entropy `0.02` with LR `1e-6` prevented collapse, but there was no winning gradient to amplify.

3. **Distance/on-point shaping helped locomotion but could be gamed or clipped.** `distance=0.05`, `on_point=0.02` were not enough under `hold_and_shoot` and not enough to survive `basic`.

### Mostly Noise Or Falsified Hypotheses

1. **More generic shaping magnitude is not the fix.** The design doc says 3x shaping made behavior worse by encouraging combat avoidance.

2. **Damage-dealt shaping cannot teach aim from zero.** It requires hits before it pays.

3. **Mean reward alone was misleading.** v7 holdshoot showed `mean_reward ~= +1.000` while score stayed `0/0`, all episodes drew, and `onpt` went to zero.

4. **Longer rounds were the wrong direction against `basic`.** `v7_basic_reduced_bc_v6` showed the bot's advantage compounds with time.

## Diagnosis If v7 Fire-Cooldown Also Fails

If `phase4_mappo_basic_v7_basic_reduced_bc_v7.yaml` fails, interpret by failure mode:

- **All draws, score `0/0`, Team A kills not above bot kills:** bot fire-rate nerf was not enough to make BC aim decisive. The problem is likely aim quality and/or focus fire, not survival.
- **Losses return by update 200:** even half-rate `basic` is still too strong, or PPO at LR `2e-6` is still degrading the BC policy.
- **Team A kills exceed bot kills but score remains `0/0`:** the combat policy may be getting isolated kills but not converting them into cap majority. That points to coordination/objective coupling, not raw combat.
- **Score appears briefly then disappears:** PPO oscillation/checkpointing is the issue; treat early-stop/best-checkpoint preservation as load-bearing before changing curriculum again.

## Ranked Next Approaches

### 1. Add A Dedicated Combat-Skill Pretraining/Evaluation Stage Before Objective Combat

Expected impact: highest.

Current BC firing is heuristic: walk to objective, aim at nearest visible enemy, set primary fire. It activates `bin`, but it does not prove useful aim, target selection, or focus fire. If v7 fails even with a 2:1 DPS advantage, the policy likely cannot hit or coordinate well enough for 3v3.

Concrete approach:

- Add a short supervised or scripted-imitation stage focused only on combat mechanics.
- Environment should remove cap scoring or make it secondary.
- Use visible enemies, stationary or slowly moving targets, and a target policy that aims and fires accurately.
- Measure hit rate, damage dealt, and kills before returning to objective control.
- Then warm-start the current `basic` reduced setup from this combat-capable checkpoint.

Success criteria before PPO objective stage:

- `bin` remains nonzero.
- Team A damage dealt is nonzero in eval.
- Team A kills are nonzero against a weak scripted shooter.
- Replay shows shots aimed at enemies, not random firing.

This is the cleanest response to the evidence that `bin~0.33` alone is insufficient.

### 2. Replace `basic` With A Moving Weak Shooter Curriculum Opponent

Expected impact: high.

The failed opponents bracket the useful teacher:

- `hold_and_shoot`: shoots but does not contest, so avoidance wins.
- `basic`: contests and shoots, but aim/fire strength is too high unless heavily weakened.

The missing rung is a `walk_and_shoot` or "weak_basic" opponent that:

- walks to the cap,
- contests the objective,
- fires with reduced accuracy, longer cooldown, or delayed engagement,
- is still capable of punishing passive cap-sitting.

Concrete approach:

- Use 30s rounds, 1000 damage, BC `walk_and_shoot`, entropy `0.02`.
- Start with bot cooldown 30 or 45 and/or aim noise.
- Gate on Team A score, Team A kills, and `onpt`, not mean reward.
- Once stable wins appear, anneal bot cooldown/accuracy toward normal `basic`.

This preserves the Phase 3 principle: make objective behavior useful first, then slowly increase adversarial combat.

### 3. Make The 3v3 Objective Easier Temporarily, Then Restore Majority Pressure

Expected impact: high, but it changes the game curriculum more directly.

The core Phase 4 gap is that 3v3 scoring requires clearing enough enemies. A temporary objective variant can reduce that cliff without changing the final gate.

Options:

- require fewer uncontested ticks before score starts,
- reduce respawn time asymmetry for the learner only during curriculum,
- temporarily score partial progress for equal contest plus recent enemy kill,
- train 2v2 as an intermediate if supported,
- or run 3v3 with only one or two active enemy shooters while all slots still exist.

The cleanest version is probably an intermediate 2v2 or 3v3-with-two-passive-enemies curriculum if the env supports it without load-bearing sim changes. The point is to require combat conversion, but not three simultaneous enemy clearances.

Use this only as a curriculum rung. Do not treat it as Phase 4 gate evidence.

### 4. Add Conservative PPO Guardrails Around The BC Policy

Expected impact: medium-high.

v4 and v5 together show that PPO update size is load-bearing:

- LR `5e-5` collapsed a viable 1000-damage BC draw into losses by update 50.
- LR `1e-6` preserved it but did not improve.

Concrete approach:

- Keep LR in the `1e-6` to `2e-6` range.
- Add early-stop on eval regression from draw to loss.
- Save true `ckpt_best.pt` and verify the trainer actually restores it.
- Add a KL cap or stricter PPO update gate relative to the post-BC policy.
- Consider freezing lower layers or movement/aim heads for the first N updates if the model supports clean parameter grouping.

This does not create a winning gradient by itself, but it prevents destroying rare good behavior once another curriculum rung produces it.

### 5. Consider Architecture/Action Decomposition Only After The Curriculum Teacher Is Fixed

Expected impact: medium, higher engineering cost.

The evidence supports rethinking the architecture if v7 fails with weaker bot fire:

- Movement and objective behavior can be learned.
- Binary fire can be initialized.
- Aim quality and combat conversion remain weak.

Possible changes:

- separate combat head or auxiliary combat loss for aim/fire,
- target-centric auxiliary prediction from actor observations,
- staged head freezing: keep movement stable while adapting combat,
- explicit action masking or valid-fire modeling if invalid fire dominates gradients,
- separate entropy coefficients for movement, aim, and binary actions.

Do not start here unless the next opponent-curriculum attempt also fails. The current failures can still be explained by curriculum and teacher mismatch, and architecture changes risk invalidating comparisons.

## Recommended Immediate Order After v7

If v7 fails, run the next work in this order:

1. **Create a moving weak shooter rung.** Prefer `walk_and_shoot`/weak `basic` over any stationary shooter. Keep 30s, 1000 damage, BC `walk_and_shoot`, LR `1e-6` or `2e-6`, entropy `0.02`.
2. **Add a combat-only or combat-heavy BC/eval stage.** Prove hit rate and kills before asking PPO to solve 3v3 objective control.
3. **Fix checkpoint preservation before long runs.** The v6.5 reproduction found no `ckpt_best.pt` despite earlier claims. Do not rely on transient breakthroughs unless the artifact exists and is used.
4. **Use an easier objective transition if combat kills still do not score.** Try 2v2/intermediate majority pressure or a temporary score rule, then anneal back.
5. **Only then consider model-head changes.** Use them to preserve movement while learning combat, not as a substitute for a better teacher.

## Metrics To Treat As Gate Evidence

Do not use self-play win rate or mean reward alone.

For every next run, record:

- config path,
- commit,
- seed,
- W&B URL,
- checkpoint path,
- replay path,
- wins/losses/draws,
- score A/B,
- Team A and bot kills,
- Team A damage dealt if available,
- `onpt`,
- `dist`,
- `bin`,
- whether Team A kills convert into score within the same episode.

Minimum success for a curriculum rung is not "positive mean reward." It is one of:

- nonzero Team A score in eval,
- nonzero Team A wins,
- Team A kills consistently above bot kills plus maintained `onpt`,
- replay-confirmed useful firing that leads to objective control.

If subjective behavior is required, block with `HUMAN_INSPECTION_REQUIRED` and include the replay command.

## Bottom Line

The Phase 4 failure is not that MAPPO cannot move agents to the point. It can. The failure is the 3v3 transition from objective contesting to decisive combat control. The journal evidence says the next breakthrough will probably come from a better intermediate teacher: one that contests the objective, is weak enough to beat, and still forces combat. If that does not work, the next layer is explicit combat pretraining or architecture that separates movement preservation from aim/fire learning.
