# Phase 4 v7 500-Damage Stalemate Analysis

## Executive Summary

- The 500-damage run converted the previous 2500-damage failure from a slaughter into a stable timeout draw, but it did not create a combat-learning curriculum. The learner held the cap area (`onpt=0.58-0.67`, `dist=0.14-0.18`) while emitting no fire actions (`bin=0.000`) and producing `0/0` score.
- The immediate failure is not just "our agents never fire." It is the combination of a zero-fire learner and an under-lethal `basic` opponent: at 500 damage, the bot can contest and shoot, but it cannot kill or clear point fast enough to create terminal or score pressure.
- The reward signal at 500 damage is mostly timeout penalty plus capped position shaping. With `50/50` draws, `0/0` score, and almost no kills, PPO receives no clear gradient saying "combat is required to avoid losing."
- A likely next damage probe is `1000-1500` centi-HP, but damage tuning alone is probably insufficient unless firing exploration is also addressed. The stronger recommendation is to combine moderate damage with an explicit fire-discovery aid or BC that includes firing.

## Run Context

- Config: `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v7_basic_reduced_bc_v2.yaml`
- Task: `t_30110cee`
- W&B run: [`35tuh4hh`](https://wandb.ai/mitchelldurbinuky-aspect/xushi2/runs/35tuh4hh)
- Local W&B metadata: `python/wandb/run-20260514_153658-35tuh4hh/files/wandb-metadata.json`
- Commit: `c94728a9a98dd641c611a0a80b8c0d3da719a48d`
- Opponent: `basic` bot, which walks to the cap and fires
- Warm-start: `runs/phase4_mappo_basic_v6_5/mappo/ckpt_final.pt`
- BC pretrain: 500 steps, loss `0.1697 -> 0.0005`
- Mechanics change: `revolver_damage_centi_hp: 500`, or 5 HP per shot, about 20 hits to kill

## What Happened

The v7 `basic_reduced` line now has three clear regimes:

| Run | Damage | BC | Result |
| --- | ---: | --- | --- |
| `v7_basic_reduced` | 2500 | no | `50/50` losses, score `0/7`, learner kills `0`, bot kills `27` |
| `v7_basic_reduced_bc` | 2500 | yes | Same qualitative failure: cap behavior improved, but the bot still slaughtered the learner |
| `v7_basic_reduced_bc_v2` | 500 | yes | `50/50` draws, score `0/0`, learner kills `0`, bot kills `0-6` per 50 eval episodes |

The 500-damage intervention succeeded at one narrow goal: it stopped immediate deaths. BC re-anchored objective movement, and PPO did not erase that behavior through update 200. The learner stayed close to the point:

| Eval | W/L/D | Score | Kills learner/bot | `onpt` | `dist` | `bin` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| BC eval | `0/0/50` | `0.00/0.00` | not reported | not reported | not reported | not reported |
| Update 50 | `0/0/50` | `0.00/0.00` | `0.0/6.0` | `0.580` | `0.157` | `0.000` |
| Update 100 | `0/0/50` | `0.00/0.00` | `0.0/6.0` | `0.674` | `0.140` | `0.000` |
| Update 150 | `0/0/50` | `0.00/0.00` | `0.0/0.0` | `0.604` | `0.163` | `0.000` |
| Update 200 | `0/0/50` | `0.00/0.00` | `0.0/3.0` | `0.622` | `0.160` | `0.000` |

But the intervention failed at the actual curriculum goal. The learner never discovered firing, the bot rarely killed, neither team scored, and every eval episode timed out at 30 seconds.

## Root Cause Analysis

### 500 Damage Removed The Only Useful Pressure

At 2500 damage, the setup was too punishing for a zero-combat learner. Four hits were enough to kill, so a policy that walked to the point and did not fire got cleared before it could score. That produced a bad learning basin, but it did at least create an external fact PPO could observe: "if we stand on cap and do nothing, we lose."

At 500 damage, that fact disappeared. The learner can stand near or on the point for long periods while not firing. The `basic` bot still contests, so the objective remains frozen, but its damage output is too low to reliably clear the learner during a 30-second eval. The result is a locally stable non-game:

```text
 learner walks to cap
+ basic bot walks to cap
+ both teams contest
+ learner never fires
+ bot fires but rarely kills
= no score, no terminal winner, timeout draw
```

This is why 500 damage created a stalemate instead of combat learning. It made survival possible, but it did not make fighting necessary.

### `bin=0.000` Is A Critical Failure, But Not The Whole Failure

The learner's `bin=0.000` means the binary action heads are effectively inactive in rollout. For Ranger-only Phase 4, that means no primary fire, no combat roll, and no meaningful combat behavior. This alone is enough to prevent learning combat from kill rewards, because there are no learner damage or kill events to reinforce.

However, the 500-damage result also shows the opponent is no longer a useful teacher. If the bot could reliably kill a passive cap-sitter at 500 damage, the learner would still receive loss/death pressure. Instead, bot kills dropped from `27/50` at 2500 damage to `0-6/50` at 500 damage, and bot score dropped to `0.00`. So the stalemate has two sides:

- The learner never samples the fire action.
- The opponent cannot punish that omission strongly enough at 500 damage.

The second point matters because a curriculum can sometimes tolerate a zero-fire learner if the environment makes "never fire" consistently lose. This run does not.

### The Reward Signal Is Mostly Non-Combat Shaping

With `50/50` draws, terminal reward is `0.0`. With score `0/0`, score reward is also absent. With learner kills at `0.0` and bot kills mostly near zero, kill/death reward is sparse to nonexistent.

That leaves the configured dense terms:

- `distance_shaping_coef: 0.05`
- `on_point_shaping_coef: 0.02`
- `time_penalty_per_second: 0.05`
- shaped reward clipping through the normal per-episode cap

The eval `mean_reward=-1.000` is consistent with a timeout policy paying time penalty while receiving some position shaping. It is not a combat gradient. Since both teams are present and contesting, score never starts; since the learner never fires, kills never start; since the bot rarely kills, losses do not start. PPO is optimizing inside a low-information draw basin.

The important difference from 2500 damage is that the 2500-damage run at least produced losses, deaths, and occasional bot score. That was too harsh, but it supplied pressure. At 500 damage, the pressure vanished.

## Why Neither Extreme Works

High damage makes the binary fire-discovery problem fatal. At 7500 damage, a Ranger dies in roughly 1.3 hits; at 2500 damage, about 4 hits. A warm-started policy that knows how to walk to cap but not how to fight is eliminated before random or entropy-driven binary-action exploration can connect "fire" to "survive."

Low damage makes passive contesting viable. At 500 damage, about 20 hits are required to kill. The bot's aim, line of sight, reload cadence, pathing, and the short 30-second round combine to make kills rare. Once neither side can clear the other, the objective state machine freezes in contested state and no team gets score.

The underlying issue is therefore not a scalar damage issue alone. Damage controls how quickly the curriculum punishes passivity, but the learner still needs a path to discover and retain firing.

## The Goldilocks Hypothesis

A plausible damage range is `1000-1500` centi-HP:

| Damage | Approx hits to kill | Expected curriculum effect |
| ---: | ---: | --- |
| 7500 | 1-2 | Too lethal; cap policy dies almost instantly |
| 2500 | ~4 | Still too lethal for a zero-combat warm start |
| 1500 | ~7 | Potentially dangerous but survivable; passive cap-sitting should eventually lose |
| 1000 | ~10 | More forgiving; may preserve cap behavior while restoring kill pressure |
| 500 | ~20 | Too forgiving; passive contested timeout is stable |

The target is not "agents survive forever." The target is "agents survive long enough to sample firing, but not long enough to ignore firing." That suggests damage should be high enough that a passive cap-sitter usually dies within the eval window, but low enough that one or two early aim/fire mistakes do not instantly erase the cap-reaching behavior.

`1000` and `1500` are both reasonable probes. If only one should run first, `1500` is the sharper test because it is more likely to restore "lose if you do nothing" pressure. If `1500` immediately returns to slaughter, try `1000`. If both still show `bin=0.000`, then damage tuning is not addressing the primary bottleneck.

## Alternative Approaches Beyond Damage Tuning

### Add A Tiny Fire-Use Exploration Bonus

A small, temporary reward for valid primary-fire attempts could solve the exact discovery problem shown by `bin=0.000`. This should be treated as a curriculum scaffold, not a gate reward. It should be logged separately and annealed away before claiming Phase 4 progress.

The risk is obvious: a raw fire bonus can teach ammo dumping or firing at nothing. To reduce that risk, make it tiny and preferably conditional:

- valid primary-fire action while alive and not reloading,
- optional enemy-visible condition,
- optional line-of-sight or aim-cone proximity condition,
- explicit anneal to zero after the binary head becomes active.

### Increase Entropy Temporarily

The run used `entropy_coef: 0.005` to preserve BC structure. That likely helped keep cap behavior but did not wake up the binary heads. A short higher-entropy phase, or separate entropy weighting for binary actions if supported, may help exploration without changing rewards.

The risk is that generic entropy may also disturb movement and aim, eroding the cap anchor BC just restored. This is less targeted than a firing scaffold.

### Add Firing To BC

The current BC pretrain successfully re-anchors movement to the point, but the post-BC eval already draws `50/50` with no score. If the demonstrations do not include firing, BC may be actively reinforcing a no-fire cap-walk policy.

A better BC stage would include "walk to cap and fire at visible/basic opponent" behavior. This is probably the cleanest fix if scripted demonstrations are easy to generate, because it initializes the binary head away from zero before PPO starts.

### Change The Opponent Curriculum

The `basic` bot is better than `hold_and_shoot` because it contests the objective, but at low damage it still creates a symmetric jam. A more useful scripted opponent for this rung may need to be intentionally imperfect but reliable:

- contest cap,
- fire often enough to punish passivity,
- aim well enough to eventually clear passive agents,
- not aim so well that zero-combat learners are instantly deleted.

This could be achieved through damage, aim noise, fire cooldown, or scripted behavior. Damage is only one axis.

### Make Timeout Draws Less Attractive

Increasing timeout or contest penalties could break the draw basin, but this should be handled carefully. If the penalty flows into the same shaped-reward machinery and clipping, it may not create the intended distinction. It can also produce perverse incentives to leave the point if contesting itself is punished.

The cleaner signal is still objective outcome: if a passive learner does not fight, the opponent should eventually kill it or score.

## Recommendations Ranked By Expected Impact

1. Add a short BC refresh that includes firing while moving to or holding the cap.

Expected impact: highest. It directly addresses `bin=0.000` without relying on PPO to discover a sparse binary action under pressure. Success criterion: post-BC or early-PPO `bin > 0`, nonzero learner shots, and no collapse in `onpt`.

2. Run a `1500` damage probe with the same BC setup, then a `1000` probe if `1500` is too lethal.

Expected impact: high. This tests the survivable-but-dangerous hypothesis. Success criterion: bot kills are high enough to punish passive play, but learner on-point behavior remains nonzero and score/kills start moving.

3. Add a tiny temporary valid-fire exploration reward, annealed to zero.

Expected impact: high if BC firing demos are unavailable. Keep it explicitly marked as curriculum shaping. Success criterion: binary fire rate rises above zero and remains useful after the bonus is removed.

4. Temporarily increase binary-action exploration pressure.

Expected impact: medium. This is less invasive than reward changes but less targeted than BC or a fire scaffold. It should be monitored for movement-policy damage.

5. Tune opponent lethality through more than damage.

Expected impact: medium. If `1000-1500` still fails, tune bot aim, fire cadence, or behavior so passive cap-sitting reliably loses while early combat attempts are survivable.

6. Deprioritize further pure low-damage runs at `500`.

Expected impact: low. The current evidence is already clear: 500 damage removed slaughter, but it also removed the learning pressure needed for combat.

## Phase-Gate Interpretation

This run is complete but does not clear a Phase 4 gate. It produced usable negative evidence:

- BC movement anchoring worked.
- Damage reduction from 2500 to 500 avoided immediate slaughter.
- The learner still did not discover firing.
- The opponent at 500 damage did not create enough pressure to force combat.
- The resulting environment has no meaningful win/loss/score gradient.

Decision: evidence insufficient for Phase 4 progress; use the result to adjust the next curriculum rung rather than treating it as blocked.
