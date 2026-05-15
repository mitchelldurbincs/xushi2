# Phase 4 Stalemate Escalation

Date: 2026-05-15

## Summary

Phase 4 remains blocked after the full 3v3 scoreless draw basin and one
post-escalation transfer attempt. The latest Escape Protocol work ruled out
Sections 5.1 through 5.4 as implemented:

- 5.1 auxiliary aim head learned its supervised target but did not improve
  score.
- 5.2 per-action entropy preserved firing and a small kill edge but did not
  produce score.
- 5.3 invalid-fire masking was almost always open (`fire_valid_fraction =
  0.9994`) and did not change score conversion.
- 5.4 aim-only mini-game succeeded synthetically (`94.96/96` greedy hits), but
  the full 3v3 transfer failed. A follow-up diagnostic showed the standard
  500-step `walk_and_shoot` BC pass erases the synthetic aim mapping before
  PPO (`94.84/96` hits before BC, `0.02/96` after BC).
- Human-selected Option 1, freezing the actor trunk and aim row during
  post-load BC, preserved the synthetic aim skill (`94.58/96` hits after BC)
  but failed full 3v3 transfer. Both BC eval and the first PPO eval collapsed
  to `0/50` wins, `50/50` losses, score `0/7`, kills `0/3`.

The current evidence says the actor can learn visible-target aim in isolation,
and Option 1 can preserve that aim pathway through BC. However, freezing the
shared actor trunk prevents the policy from restoring enough movement/objective
competence to survive weak_basic. The unresolved problem is no longer just
"preserve aim through BC"; it is how to compose aim skill with the known
movement/objective behavior without collapsing either side.

## Recent Runs

| Config | W&B | Replay | Final outcome |
|---|---|---|---|
| `phase4_mappo_aux_aim_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/r41572eu` | `data/replays/phase4_aux_aim_v1_ckpt0500_stochastic.replay` | 0/50 wins, 50/50 draws, score 0/0, kills 1/6 |
| `phase4_mappo_per_action_entropy_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/n3gh2mea` | `data/replays/phase4_per_action_entropy_v1_ckpt0500_stochastic.replay` | 0/50 wins, 50/50 draws, score 0/0, kills 6/5 |
| `phase4_mappo_invalid_fire_mask_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/x4mketjt` | `data/replays/phase4_invalid_fire_mask_v1_ckpt0500_stochastic.replay` | 0/50 wins, 50/50 draws, score 0/0, kills 5/5 |
| `phase4_mappo_aim_only_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/d6qgug61` | synthetic env; no C++ replay | Positive diagnostic: 94.96/96 hits |
| `phase4_mappo_aim_transfer_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/9n07ntl5` | `data/replays/phase4_aim_transfer_v1_ckpt0500_stochastic.replay` | 0/50 wins, 50/50 draws, score 0/0, kills 0/3 |
| `phase4_mappo_aim_freeze_bc_v1.yaml` | `https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/88w0yivd` | `data/replays/phase4_aim_freeze_bc_v1_ckpt0050_stochastic.replay` | 0/50 wins, 50/50 losses, score 0/7, kills 0/3 |

## Behavioral Findings

- Agents do fire. Recent replay primary-fire rates are approximately
  `0.997-1.000`.
- Agents do move while firing. Recent moving-while-firing rates are
  approximately `0.98`.
- Agents do not convert fire into score. Full 3v3 evals remain `50/50` draws
  with `score 0/0`.
- Per-agent focus-fire attribution and body/headshot attribution are not
  available in the current replay format.
- The aim-only mini-game proves the actor and PPO loop can learn a direct
  visible-target aim mapping when reward is immediate.
- The standard full-env `walk_and_shoot` BC pretrain is implicated: it restores
  draw-basin behavior but erases the synthetic aim mapping.
- Freezing the shared actor trunk and aim row preserves the synthetic aim
  mapping, but the resulting full 3v3 policy no longer maintains
  movement/objective competence and loses every greedy eval episode.

## Falsified Hypotheses

- Opponent strength alone explains the basin: `weak_basic_v1` still drew.
- Supervised auxiliary aim alone fixes aim: aux RMSE reached target but score
  stayed zero.
- Aim entropy alone finds transfer: kills improved transiently but no score.
- Invalid fire wastes the useful gradient: fire-valid fraction was already
  almost always true.
- Aim-only warm-start plus normal BC transfers: BC erased aim-only performance.
- Aim-only warm-start plus frozen aim/trunk BC transfers: aim survived, but
  full weak_basic eval collapsed to 50/50 losses with zero learner score.

## Recommended Human Decision

Choose one structural direction before any more full-length Phase 4 run:

1. Compose the movement checkpoint with the aim checkpoint.
   Start from the known movement/objective checkpoint and distill or transplant
   only the aim mapping from `phase4_mappo_aim_only_v1`, then run the same
   aim-retention and weak_basic BC-only diagnostics before PPO.

2. Replace the `walk_and_shoot` aim target.
   The current BC target appears incompatible with the mini-game-trained aim
   mapping. Use the mini-game target function or a hit-validated target for
   full-env BC, then run the same retention diagnostic before PPO.

3. Add full-env aim instrumentation before more architecture work.
   Log per-shot visibility, target slot, aim error, hit/miss, and damage
   attribution during eval/replay so the next intervention can distinguish
   tracking error, cooldown timing, focus-fire, and target-selection failure.

4. Implement Section 5.5 combat-head separation.
   This is larger-scope engineering: freeze proven movement/objective behavior
   and train a separate combat head for aim/fire. Do this only after deciding
   how to preserve or supervise aim through BC.

Specific question for the human reviewer:

Should the next implementation compose the movement and aim checkpoints
(option 1), replace the BC aim target (option 2), add full-env shot/aim
instrumentation first (option 3), or proceed directly to a separate combat head
(option 4)?
