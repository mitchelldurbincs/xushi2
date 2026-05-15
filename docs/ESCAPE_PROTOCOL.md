# Phase 4 Escape Protocol — Break-Glass for Draw Basin Stalemates

**Status:** Active. Phase 4 has produced 13+ variants across every hyperparameter axis (damage, round length, fire rate, BC pretraining, LR, entropy, opponent type) and all converge to the same draw basin: 50/50 draws, score 0/0, kills 4-6 per 50 episodes, mean_reward ~+0.96. **Hyperparameter whack-a-mole has been exhausted.**

**This document overrides the default "tweak one parameter and queue another run" behavior.** It is referenced by the main prompt alongside `GOAL_INSTRUCTIONS.md`.

---

## 1. The Circling Detector (Mandatory Gate)

Before creating ANY new config or task, query the kanban DB for the last **5 completed or failed Phase 4 tasks.**

```sql
SELECT t.id, t.title, e.payload
FROM tasks t
LEFT JOIN task_events e ON t.id = e.task_id AND e.kind = 'failed'
WHERE t.title LIKE '%Phase 4%'
  AND (t.status = 'done' OR t.status = 'blocked')
ORDER BY t.created_at DESC
LIMIT 5;
```

**If 3 or more of the last 5 tasks have eval outcomes matching this pattern:**
- Draws >= 45/50 episodes
- Score 0/0 or near-zero
- Kills per 50 episodes < 10 for both teams combined
- mean_reward between -1.0 and +1.5

**→ STOP. Do NOT create another hyperparameter-variant config. Proceed to Section 2 (Behavioral Autopsy).**

This rule is absolute. Violating it produces the 14th, 15th, 16th identical result.

---

## 2. Behavioral Autopsy (Mandatory Before Any New Config)

Every completed or failed training run MUST have replay artifacts dumped. If none exist, run:

```bash
cd /home/aspect/source/personal/xushi2/python
python scripts/dump_replay.py --config <config_path> --checkpoint <ckpt_path> --episodes 5 --stochastic
```

**Before declaring a run "failed" or "done", answer these questions in the task's journal entry:**

| Question | How to Answer | What It Tells Us |
|---|---|---|
| Do agents ever fire at visible enemies? | Replay viewer or `--stochastic` replay | If `bin=0.33` but shots miss, aim is the problem. If `bin=0.0`, fire action is dead. |
| Do agents focus fire on one enemy or spray? | Replay viewer | Spray = no coordination. Focus fire = aim/tracking issue, not action existence. |
| Do agents strafe/dodge while firing? | Replay viewer | Standing still = easy target. Strafing = survival skill exists, aim is the gap. |
| Do agents retreat from cap when shot at? | Replay viewer | Retreating = policy learned "cap = danger". This is behavioral collapse, not aim. |
| Are kills from our side concentrated in one agent? | Per-agent kill stats if available | Concentrated = one agent learned something. Distributed = nobody learned anything useful. |
| Is the bot killing us with body shots or headshots? | N/A (bot aim is perfect at current position) | Bot has perfect aim, we don't. Confirms aim gap, not survival gap. |

**If replays cannot be inspected (no viewer, no time), fall back to stochastic eval metrics:**
- `--stochastic` eval with `episodes=50` — log per-episode kills, score, onpt
- If stochastic eval differs materially from greedy eval (e.g., more kills, some wins), the policy HAS capability that greedy action masks. This is critical evidence.

**Without answering at least 3 of the 6 questions, the run is NOT done. It is `HUMAN_INSPECTION_REQUIRED` or `BLOCKED_FOR_DIAGNOSTIC`.**

---

## 3. Hypothesis-Driven Config Design (Mandatory)

Every new config file MUST include a `hypothesis:` field in the YAML `metadata:` block. The prompt template:

```yaml
metadata:
  hypothesis: "<specific behavioral hypothesis>"
  falsification_criteria: "<what eval metrics would prove this wrong>"
  max_updates_if_no_signal: <number>
```

**Examples of valid hypotheses:**
- "Agents have aim capability but greedy eval hides it; stochastic eval will show 10+ kills."
- "Agents retreat under fire because cap = negative value; adding `on_point_shaping_coef: 0.10` will override retreat."
- "Bot perfect aim is the bottleneck; `weak_basic` with ±0.5 rad noise will let agents survive long enough to score."
- "Agents can hit but not kill in 30s; 60s rounds will show score separation."

**Examples of INVALID hypotheses (already falsified):**
- "Higher entropy will discover better aim." ❌ Falsified by v5_high_entropy (340 updates, identical to v5).
- "Lower damage will let agents learn to fire." ❌ Falsified by v7_bc_v2 (500 damage, 0 kills, bin=0.0).
- "Longer rounds will help." ❌ Falsified by v7_bc_v6 (60s rounds, bot score climbed to 25.33).

**If you cannot articulate a falsifiable behavioral hypothesis that has NOT been tested, do NOT create a config. Proceed to Section 4 (Diagnostic Shortcuts) or Section 5 (Architecture Break-Glass).**

---

## 4. Diagnostic Shortcuts (Use Before 1000-Update Runs)

Before burning 90+ minutes on a full PPO run, answer the hypothesis with a cheaper experiment:

| Hypothesis Axis | Diagnostic | Cost | How to Interpret |
|---|---|---|---|
| "Can BC hit enemies at all?" | Run BC-only with `walk_and_shoot` variant for 1000 steps, eval 50 episodes | ~2 min | If BC eval shows 0 kills, the BC heuristic itself cannot hit the bot. Fix BC before PPO. |
| "Does the bot miss enough for us to survive?" | Run `basic` vs `basic` self-play eval with proposed damage/cooldown | ~1 min | If perfect-aim bots draw 50/50 at these params, no learner can win either. Params are too soft. |
| "Is the policy hiding capability behind greedy eval?" | Stochastic eval (50 episodes) on ANY existing checkpoint | ~5 min | If stochastic shows wins/good kills where greedy shows none, policy HAS capability. |
| "Would a weaker bot let us score?" | Script a 50-episode eval with `noop` or `walk_to_objective` opponent | ~2 min | If we still can't score against a non-shooting opponent, the problem is NOT combat. |
| "Is the fire action dead or just missing?" | Log `bin` action distribution per-agent in eval | ~0 min (already logged) | `bin=0.33` but 0 kills = aim problem. `bin=0.0` = fire action is dead. |

**Rule: If a hypothesis can be tested with a diagnostic costing <10 minutes, run the diagnostic FIRST. Only queue a 1000-update PPO run if the diagnostic gives a positive signal.**

---

## 5. Architecture Break-Glass (When Opponent + Hyperparameter Space Is Exhausted)

If the Circling Detector (Section 1) has fired, AND Behavioral Autopsy (Section 2) confirms agents fire but miss, AND Diagnostic Shortcuts (Section 4) confirm no hidden capability, the problem is likely **structural** — not opponent strength or hyperparameters.

**Permitted architecture changes (in order of expected impact):**

### 5.1 Auxiliary Aim Prediction Head (Highest Expected Impact)
Add an auxiliary supervised loss that predicts the angle to the nearest visible enemy. Trained on BC data from `walk_and_shoot` variant. This gives the network explicit aim gradients separate from PPO's sparse reward.

**Success criteria:** Auxiliary loss < 0.1 rad RMSE. PPO eval kills increase vs same config without auxiliary head.

### 5.2 Per-Action-Type Entropy (Medium-High Impact)
Separate entropy coefficients for `move`, `aim`, and `bin` actions. Current single `entropy_coef` may over-exploit movement while under-exploring aim. Try `entropy_coef_aim: 0.15`, `entropy_coef_move: 0.02`, `entropy_coef_bin: 0.05`.

**Success criteria:** `bin` stays >0.3, `move` doesn't collapse to 0.9+, aim distribution shows wider spread.

### 5.3 Action Masking / Invalid Fire Suppression (Medium Impact)
If `bin=0.33` but 0 kills, agents may be firing when no enemy is visible. Add a simple action mask: `primary_fire` is only valid when `visible_enemy_count > 0` in obs. This concentrates fire gradients on meaningful timesteps.

**Success criteria:** Kill-per-fire-action ratio increases (same `bin`, more kills).

### 5.4 Curriculum Stage: Aim-Only Mini-Game (Medium Impact)
Create a mini-environment or scripted episode where agents spawn facing a stationary target at fixed distance. Reward = damage dealt only. No movement, no objective, just aim-and-fire for 500 episodes. Warm-start into full 3v3 from this.

**Success criteria:** Mini-game produces >50% hit rate. Warm-start into 3v3 shows higher kills than random init.

### 5.5 Separate Combat Head with Frozen Movement (Low-Medium Impact, High Effort)
Freeze the movement/locomotion layers (proven to work from v6.5) and add a new combat head that only learns aim + fire. Train only combat head against `weak_basic` while movement is frozen.

**Success criteria:** `onpt` stays >0.3 (movement frozen), kills increase above baseline.

**Rule: Only ONE architecture change per experiment. Do not combine 5.1 + 5.2 + 5.3 in the same run. Isolate variables.**

---

## 6. Human Escalation Protocol

If the following conditions are ALL true:
1. Circling Detector fired (3+ identical draw outcomes)
2. Behavioral Autopsy completed with replays
3. No falsifiable hypothesis remains untested in opponent/hyperparameter space
4. Architecture changes from Section 5 require C++ sim modifications or multi-day engineering

**→ Mark current task `HUMAN_INSPECTION_REQUIRED`. Write a summary to `docs/reports/phase4_stalemate_escalation.md` with:**
- List of all configs tested and their identical outcomes
- Replay path and W&B URLs for the 3 most recent runs
- Behavioral autopsy findings (what agents actually do)
- Proposed architecture change with estimated effort
- Specific question for human: "Should we implement 5.1 (auxiliary aim head), 5.4 (aim-only mini-game), or something else?"

Then STOP. Do not queue another 1000-update run until human responds.

---

## 7. Success Criteria for "Escaping the Hole"

Phase 4 is NOT "done" when a single config wins. It is "done" when we can articulate WHY it won and reproduce it. The escape protocol succeeds when:

1. A config produces **>10 wins in 50 eval episodes** with **score > 0**
2. The win pattern is **reproduced on a second seed** (not just one lucky seed)
3. A replay shows agents **firing at enemies, moving while firing, and scoring**
4. The journal entry explains **which lever broke the draw basin** (opponent, architecture, or reward)

Until all 4 are true, we are still in the hole.

---

## Appendix: Already-Falsified Hypotheses (Do Not Retest)

| Hypothesis | Config(s) | Outcome | Falsification |
|---|---|---|---|
| Higher entropy discovers better aim | v5_high_entropy (entropy 0.08) | Identical to v5 through 340 updates | Draw basin invariant to entropy |
| Lower damage lets agents survive and learn | v7_bc_v2 (500 dmg) | 50/50 draws, 0 kills, bin=0.0 | Too soft, no combat pressure |
| Longer rounds help conversion | v7_bc_v6 (60s rounds) | Bot score 25.33, 0 wins | Longer rounds favor stronger bot |
| Damage-dealt shaping teaches aim | v5 (damage_dealt_coef: 0.01) | 0 kills, 0 score | Requires hits before it pays |
| hold_and_shoot opponent lets agents score while learning combat | v7_holdshoot, v7_holdshoot_v2, v7_holdshoot_v3 | onpt→0, dist→0.5, score 0/0 | No objective pressure = avoidance wins |
| LR 1e-6 preserves cap while slowly improving | v5, v5_high_entropy | 450+ updates, 0 wins, 0 score | Draw basin is stable but inescapable |
| BC walk_and_shoot gives enough aim to start | v7_bc_v3, v7_bc_v4, v7_bc_v5 | bin~0.33, 0-8 kills per 50 episodes | BC aim is not good enough against basic |
| Reducing bot fire rate (cooldown 30→60) tips balance | v7_bc_v7 (60-tick cooldown) | Bot still wins 7.50-15.67 score | 2:1 DPS advantage insufficient |

**Consult this appendix before writing any new `hypothesis:` field. If your hypothesis is a restatement of any row above, do not run it.**
