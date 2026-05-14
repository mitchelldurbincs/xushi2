# Phase 4 v7 Hold-and-Shoot Failure Analysis

## Executive Summary

The three v7 hold-and-shoot runs failed for the same reason: the curriculum made timeout survival under a stationary shooter easier and more consistently rewarded than holding the cap. Because shaped reward is capped at `+3.0` per team episode and eval reports the per-agent mean, many bad strategies can appear as `mean_reward ~= +1.0` while producing `0/0` score, `50/50` draws, and complete cap abandonment.

The attempted fixes changed reward weights and incoming damage, but they did not change the underlying incentive basin. `hold_and_shoot` does not contest the objective, so it supplies no external pressure to occupy the cap; warm-started agents instead learn that the v6_5 behavior of reaching the cap now predicts death.

## Evidence Reviewed

- `docs/journal/reinforcement_learning_journal.md`
- `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v7_holdshoot.yaml`
- `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v7_holdshoot_v2.yaml`
- `experiments/configs/phase4/legacy/archive/phase4_mappo_basic_v7_holdshoot_v3.yaml`
- `python/xushi2/reward.py`
- `python/train/mappo_evaluate.py`

## Run Comparison

| Run | Config change | Damage | Shaping | Outcome |
| --- | --- | ---: | --- | --- |
| v1 `t_583de2a2` | First `hold_and_shoot` warm-start from v6_5 | 7500 centi-HP | `distance=0.01`, `on_point=0.0`, `time=0.05`, `team_spirit=1.0` | `onpt -> 0.000`, `dist -> 0.66`, score `0/0`, `mean_reward +1.000`, all draws |
| v2 `t_5645ff22` | Stronger cap shaping | 7500 centi-HP | `distance=0.05`, `on_point=0.02`, `time=0.05`, `team_spirit=1.0` | Stopped early; `onpt -> 0.000`, `dist ~= 0.5`, score `0/0`, `mean_reward +1.000`, all draws |
| v3 `t_bd4c6f26` | v2 plus reduced incoming damage | 2500 centi-HP | Same as v2 | Stopped at update 100; `onpt=0.092`, `dist=0.312`, score `0/0`, `mean_reward +0.998`, draws `50/50`, kills `0.0/17.0` |

## Reward Structure Problem

`RewardCalculator` applies a cumulative shaped-reward cap per episode through `_CumulativeClipper`. In the per-agent reward path, `_step_per_agent` builds a three-element reward vector for each team, then `scale_to_clipped_sum` clips the team sum to `[-shaping_clip, +shaping_clip]`. The default `shaping_clip` is `3.0`.

With `team_spirit=1.0`, the raw per-agent vector is replaced by the team mean before clipping:

```text
raw_a = raw_a.mean() for every Team A agent
team sum is clipped to max +3.0
each of 3 agents receives about +1.0 total shaped reward
eval mean_reward accumulates reward_np.mean(axis=1)
```

That explains the repeated `mean_reward ~= +1.0`: it is not evidence of cap success. It is the per-agent mean corresponding to a saturated `+3.0` team-shaped episode in a draw. Once the learner finds any behavior that reaches the positive shaped cap while avoiding terminal losses, additional reward distinctions disappear.

The stationary opponent makes this especially damaging. `_distance_shaping_delta` is differential:

```text
-distance_shaping_coef * (dist_a - dist_b)
```

Since `hold_and_shoot` stays near spawn and does not move to the objective, Team A can receive positive distance shaping without actually holding the cap. The agent only has to be closer to the cap than the stationary Team B reference. It can retreat to a safe midpoint or corner, avoid deaths, timeout the episode, and still saturate the shaped reward cap.

`_on_point_shaping_delta` is also differential:

```text
on_point_shaping_coef * (on_a - on_b)
```

Against a non-contesting opponent, `on_b` is effectively zero. This adds a small direct preference for cap contact, but it still flows into the same cumulative shaped clip as distance, score, kills, deaths, and time penalty. Once the episode reaches the positive cap through safer shaping, the extra reward for being on point no longer creates a useful gradient.

Finally, score reward does not rescue the setup. `score_per_second=0.10` only pays when scoring is already happening. The runs show score stayed `0/0`, so this term never became the dominant teacher. PPO was not comparing "score while holding cap" against "draw while retreating"; it mostly saw "cap leads to being shot" versus "retreat leads to clipped positive shaped reward and draw."

## Why The Interventions Failed

### v1: Stationary Shooter Plus Differential Distance Was Exploitable

v1 assumed the agent would keep the v6_5 cap-reaching behavior because the opponent does not contest the point. The actual incentive was different: the opponent created lethal pressure near the path/objective while contributing no objective pressure of its own.

With `on_point_shaping_coef=0.0`, there was no direct reward for cap contact. Differential distance shaping was enough to make a non-cap safe position positive because Team B stayed far away. The policy learned "avoid cap, avoid death, timeout" and still reached the clipped reward ceiling.

### v2: `on_point_shaping_coef=0.02` Was Too Small And Still Clipped

v2 raised `distance_shaping_coef` from `0.01` to `0.05` and added `on_point_shaping_coef=0.02`. This did not restore cap holding because the added on-point term was still part of the same clipped shaped-reward budget.

At `team_spirit=1.0`, the individual agent that steps onto the point does not get a preserved private credit signal; the team receives an averaged shared signal. If the team can already saturate the shaped cap without holding the cap, the marginal on-point reward is mostly invisible. Increasing differential distance may also have reinforced "be somewhat closer than spawn" rather than "stand on the objective."

### v3: Reducing Damage Did Not Change The Learned Basin

v3 reduced revolver damage from `7500` to `2500` centi-HP, but the observed opponent kills stayed about the same at `17/50`. That means the behavior did not materially enter a new "survive on cap long enough to score" regime. The agents still had `0/0` score, almost no on-point time, and no Team A kills.

Lower damage only helps if agents are already attempting the cap and the extra time alive lets score reward appear. Here, the policy had already found a high-reward draw by avoiding the cap. Because the positive shaped cap was still reachable through retreat/survival, weaker shots did not create a reason to relearn cap-holding.

## Role Of Team Spirit And Per-Agent Averaging

The journal notes that true per-agent rewards are about 3x weaker than the old broadcast path. In these runs, `team_spirit=1.0` then fully averages per-agent rewards back into a shared team signal. This is useful for cooperation when the team reward is well-shaped, but harmful when the team-level objective is ambiguous.

The averaging blurs assignment for cap contact, deaths, and positioning. One agent's risky cap behavior and another agent's safe retreat are mixed into the same team mean, then capped at the team sum. Once the team total reaches `+3.0`, the trainer cannot distinguish a policy that actually controls the point from one that merely times out safely while collecting differential shaping.

`mappo_evaluate.py` further explains the misleading metric: it accumulates `ep_rewards += reward_np.mean(axis=1)`. For a three-agent team whose clipped shaped sum is `+3.0`, the reported episode reward is approximately `+1.0`. That is exactly what v1 and v2 reported while failing every behavioral metric.

## Curriculum Diagnosis

`hold_and_shoot` appears flawed as the next Phase 4 rung. It removes the one thing that forced v6_5 to care about the objective: enemy presence on or near the cap. The opponent shoots, but it does not create a score race, does not deny by contesting, and does not force Team A to solve combat in the context of objective control.

Warm-starting from v6_5 may also be actively harmful in this exact setup. v6_5 agents know how to reach and contest the cap against a no-fire objective walker. In v7, that same behavior immediately exposes them to fire without giving enough successful scoring feedback. PPO can therefore learn the simple causal story "cap equals death" and erase the cap-holding policy.

## Recommendations For The Next Approach

1. Replace `hold_and_shoot` with a moving firing opponent, preferably `walk_and_shoot` if available.

Expected mechanism: the opponent should slowly approach or contest the cap while firing, so retreating gives up objective position and cannot remain a positively reinforced draw. Combat pressure then happens at the objective instead of being separable from it.

2. If `walk_and_shoot` is not available, try `basic` with reduced damage before more `hold_and_shoot` variants.

Expected mechanism: `basic` restores objective pressure. Reducing damage can soften the jump from v6_5 without removing the need to contest, kill, or score. This is more curriculum-aligned than a stationary shooter that never contests.

3. Stop relying on differential distance shaping against stationary opponents; add or test an absolute distance-to-cap penalty/reward.

Expected mechanism: an absolute cap-distance term rewards moving toward the objective regardless of where the opponent stands. This closes the exploit where Team A only needs to be closer than a spawn-camped Team B.

4. Revisit `shaping_clip` for this curriculum stage, or separate objective-control shaping from the global shaped cap.

Expected mechanism: with `shaping_clip=3.0`, bad draw policies can saturate the same budget as good cap policies. A larger clip, a lower distance contribution, or a separate unclipped/less-clipped objective occupancy term would preserve gradient differences between "nearer than spawn" and "actually holding cap."

5. Use BC or a short supervised refresh for cap-holding under fire before PPO, then resume PPO with stricter behavioral gates.

Expected mechanism: BC can re-anchor the warm-started policy to hold the objective despite incoming fire. PPO should then reinforce survival and combat around an existing cap-holding behavior instead of rediscovering that avoidance is safer.

Secondary knobs are worth testing only after the opponent/reward basin is fixed. A much larger `time_penalty_per_second` could make timeout less attractive, but if applied inside the same clipped shaped budget it may still be muted once the cap is reached. Raising kill bonus alone is unlikely to help because all three runs show Team A kills remained at `0.0`; damage or kill rewards cannot bootstrap combat when the policy never learns to aim and engage.

## Next Success Criteria

Do not judge the next run by `mean_reward` alone. Require at least:

- nonzero Team A score in eval,
- sustained `onpt` above the v3 early value, not a drift toward zero,
- decreasing `dist` without retreat-to-corner behavior,
- Team A kills above zero or clear replay evidence of useful firing,
- replay inspection before treating the phase as improved.

If subjective behavior judgment is required, block the phase decision with `HUMAN_INSPECTION_REQUIRED` and include the W&B URL, replay path, viewer command, and specific approval/rejection questions.
