# Xushi2 — RL Design

**Project:** Xushi2 (虚实 2)
**Date:** 2026-04-20
**Status:** MVP design (Phase 1)

Companion to `2026-04-20-game-design.md`. This document covers the training algorithm, observation and action spaces, neural architecture, curriculum, self-play setup, evaluation, and implementation hazards.

**Coding philosophy:** Sim, replay, observation, and Python-boundary code follow `coding_philosophy.md` Tier 0 rules (bounded loops, no post-init allocation, `Result<T>` returns, strict assertions, actor/critic leak tests). Python trainer code follows Tier 2 (shape/dtype/NaN checks, seed logging, replay export).

## 1. Algorithm choice

**Recurrent MAPPO with self-play and a snapshot opponent pool.**

- PPO adapted to multi-agent via centralized-training / decentralized-execution
- Actor sees per-agent partial observation only
- Critic sees full simulator state during training
- Shared policy weights across all 6 hero slots, with a hero/role embedding input
- One RNN hidden state per agent
- Self-play with a snapshot pool of 10–20 historical policies + scripted-bot anchors

This is the natural lineage from OpenAI Five (recurrent PPO, shared policy with hero-id input, self-play with past snapshots) scaled down to one-machine feasibility, plus MAPPO paper findings that a centralized critic materially helps as agent count grows and credit assignment gets harder.

Caveats worth knowing: the MAPPO benchmark paper is mostly cooperative, discrete-action, homogeneous-agent environments. Xushi2 is competitive, hybrid-action, heterogeneous. Expect tuning work; do not assume paper hyperparameters transfer.

## 2. Actor architecture (decentralized)

Each agent has its own hidden state. Per decision step:

```
h_i[t+1], logits_i[t] = RNN(h_i[t], encoded_obs_i[t])
action_i[t] ~ π(action | logits_i[t])
```

### Observation encoder

Entity-based backbone with a local egocentric grid:

```
per-entity tokens ─► shared MLP ─► attention pool ─┐
                                                    ├─► concat ─► GRU(256) ─► heads
local 32×32 grid  ─► small CNN ─► flatten         ─┘
```

**Phase 1 entity tokens** (Phase 1 hero kits: Vanguard / Ranger / Mender; no team-reveal abilities, no zone healing):

- self
- allies × 2
- visible enemies × up to 3
- last-seen enemy markers × up to 3
- objective
- active allied Barrier
- active enemy Barrier (if visible)
- active Mender beam relationship (source/target entity IDs, beam active flag) — optional, see §3

**Future entity tokens** (introduced at their respective phases, not Phase 1):

- healing zone (no hero in MVP provides one)
- recon reveal zone (no hero in MVP provides one)
- deployables
- status zones

Variable-count tokens use explicit masking in attention.

**Grid channels (32×32 egocentric):** walls, objective area, allies (3 role channels), visible enemies (3 role channels), allied Barrier, enemy Barrier (if visible), last-seen enemy markers. Egocentric, agent-oriented. No `healing zone` channel in Phase 1 — Mender's healing is a beam relationship, not a spatial zone, and is represented on the entity side.

### RNN

- GRU first. LSTM only if memory tasks require gating.
- Hidden size 256 (scale to 512 if underfitting).
- Unroll length 64–96 policy steps — brackets a ~6–10 s teamfight at 10 Hz effective decisions.
- Hidden state zeroed on episode boundary.
- Stored at rollout time, replayed exactly during training — no stale-state reuse across PPO epochs.

### Shared policy with hero embedding

One actor network parameters-shared across all 6 hero slots. Hero identity enters as an embedding (role one-hot + specific hero one-hot). Same parameters, different embedding, different hidden state per agent. This is the OpenAI Five pattern at 1/1000th scale.

## 3. Critic architecture (centralized)

**Team critic with full simulator state.**

```
V_A(full_state) = expected Team A return
V_B(full_state) = expected Team B return
```

Critic input includes **everything**:

- True positions, velocities, aim directions of all 6 heroes
- All health and shield values
- All cooldowns, ability activation states
- Active Barriers, active Mender beam relationships (source/target)
- Objective ownership, progress, contested state, integer score/cap ticks
- Round timer
- Map seed / layout ID
- Team side (A or B)
- Future-phase: healing zones, recon reveal zones, deployables, status zones (none in MVP)

### Core discipline: strict actor/critic separation

The actor must never receive any hidden-enemy fields. Actor and critic observation builders have **separate top-level functions and separate manifests**. They may share low-level pure utilities that cannot access hidden entities (team-frame coordinate transform, angle normalization, scalar normalization, enum encoding). The hard rule is:

> No function that iterates over hidden enemies or full state may be called by `actor_obs_builder`.

This preserves leak prevention without forcing harmful duplication of transforms that would otherwise drift between actor and critic.

Explicit leak tests (§10) enforce this. A leak here silently invalidates the entire research contribution — the single most important piece of infrastructure in the project.

### Zero-sum and the V_B ≈ −V_A shortcut

Because shaped rewards are team-symmetrized (§5), the game remains approximately zero-sum at the tick level and `V_B ≈ −V_A` holds. In practice the critic still learns team-conditioned values independently for robustness — the symmetry is an alignment prior, not a hard constraint.

## 4. Action space

Per-agent, per decision step:

| Action | Type | Notes |
|---|---|---|
| `move_x` | continuous [-1, 1] | tanh-squashed Gaussian |
| `move_y` | continuous [-1, 1] | tanh-squashed Gaussian |
| `aim_delta` | continuous [-π/4, π/4] | tanh-squashed Gaussian, applied to current aim |
| `primary_fire` | Bernoulli | |
| `ability_1` | Bernoulli | |
| `ability_2` | Bernoulli | |
| `target_slot` | Categorical(K) | **Phase 1–9: omitted or always 0. Phase 10+: enabled** for explicit ally/enemy targeted abilities. |

Factorization:
```
π(a | obs) = π_move(move) · π_aim(aim) · π_buttons(buttons) [· π_target(target)]
```

### Notes

- **Aim is angular delta applied per *policy decision*, not per sim tick.** ±45° per decision cap. During the action-repeat window, aim is held constant — it only advances on the next decision. At action-repeat 3 (policy rate 10 Hz), max turn rate is 450°/sec. At action-repeat 2 (policy rate 15 Hz), it is 675°/sec. Absolute aim direction lives in simulator state and is observable to the agent.
- **tanh-squashed Gaussian** for all continuous actions. Avoids boundary mode collapse that raw-Gaussian-plus-clamp produces.
- **`target_slot` is omitted in Phase 1–9.** All Phase 1 heroes use direction-based or aim-cone targeting: Barrier points where Vanguard aims, Combat Roll dashes along movement direction, Mender's Beam auto-locks the nearest ally in the aim cone, Mender's Tether zips to the ally in the aim cone. At Phase 10, the action head is enabled via attention over the entity tokens already encoded for observation, with a valid-target mask (see `observation_spec.md`). The first shipped use is Ranger Mark Target: `ability_2` with an enemy token target (`target_slot` 1–3).
- **Invalid actions are masked to no-op** (ability on cooldown, firing while dead). Cooldown state is in the observation so the policy can learn to avoid wasted probability mass on unavailable abilities.

### Held vs impulse actions

The `primary_fire`, `ability_1`, and `ability_2` Bernoulli outputs have different semantics per hero per ability. The sim interprets each one as either **held** or **impulse**; the policy sees the raw binary output.

**Impulse semantics, not edge-triggered.** For abilities that fire once per activation (Guard Step, Combat Roll, Weapon Swap, Tether), the action value `1` means "attempt the ability once at the start of this policy decision" and `0` means "do not attempt." No previous-button state lives in the sim. This keeps the action interface Markovian: identical `(obs, action)` pairs produce identical outcomes. Held abilities (Warhammer, Barrier, Ranger Revolver, Mender current weapon) behave as a button held for the whole decision window.

| Hero | Action | Ability | Mode |
|---|---|---|---|
| Vanguard | primary_fire | Warhammer | **Held** (strikes while held, fire-rate gated; **suppressed while Barrier is active**) |
| Vanguard | ability_1 | Barrier | **Held** (active while held, subject to HP/lockout) |
| Vanguard | ability_2 | Guard Step | **Impulse** (one dash if 1 at decision start) |
| Ranger | primary_fire | Revolver | **Held** (fire-rate gated; no-op when magazine empty) |
| Ranger | ability_1 | Combat Roll | **Impulse** (one dash + instant reload if 1 at decision start) |
| Ranger | ability_2 | Mark Target | **Impulse** (`target_slot` 1–3; marks a visible enemy in range) |
| Mender | primary_fire | Beam *or* Sidearm | **Held** (whichever weapon is currently equipped) |
| Mender | ability_1 | Weapon Swap | **Impulse** (one swap STAFF ↔ SIDEARM if 1 at decision start) |
| Mender | ability_2 | Tether | **Impulse** (one zip to aimed ally if 1 at decision start) |

The viewer converts real human key/mouse rising edges into single one-decision impulses, so human play and RL play emit identical `Action` structs. See `action_spec.md` for the canonical specification.

**Vanguard's Barrier/Warhammer mutual exclusion** is enforced in the sim. The policy can output `primary_fire == 1` while `ability_1 == 1`, but the sim will suppress the Warhammer strike. This shows up in the `fire-while-shielding rate` metric (§11); the policy is expected to learn to not emit both simultaneously.

**Ranger's ammo state** (`current_magazine ∈ {0, ..., 6}`, `is_reloading ∈ {0, 1}`, `reload_progress ∈ [0, 1]`) is part of the actor observation. Out-of-ammo `primary_fire` is a no-op that should trend to zero in training (tracked as `out-of-ammo fire rate` in §11).

**Mender's weapon state** (`equipped ∈ {STAFF, SIDEARM}`, `beam_locked_target_id` if STAFF and beam active) is part of the actor observation.

## 5. Reward design

**Team-shared, symmetrized, terminal-dominant.** Scaled so terminal is numerically larger than any realistic accumulation of shaping across an episode.

### Terminal (dominant signal)

```
+10.0  win (reach 100 score, or higher score at timeout)
-10.0  loss
  0.0  draw (timeout with exactly tied score)
```

### Dense shaping (small, capped, symmetrized)

```
+0.01  per second own team controls the objective (score-tick delta / TICK_HZ)
-0.01  per second enemy team controls the objective
+0.25  enemy kill
-0.25  ally death
+small useful healing delivered (overheal excluded, capped) — Phase 6a+ (Mender)
+small damage blocked near ally (Vanguard shield, capped) — Phase 6a+ (Vanguard)

total non-terminal shaping clipped to [-3.0, +3.0] per episode per team
```

Under game-design §3's "+1 score point per second while controlling", the per-second and per-score-point framings are equivalent; the code literally implements the per-second form via `0.01 * (score_ticks_delta / TICK_HZ)`.

The ±3 clip is applied per team independently; once one team's running total saturates the cap, the other side's clipped step delta will no longer be the exact negation.

The healing and damage-blocked shaping terms are Phase 6a+ items (they activate once Mender and Vanguard enter the roster); Phases 1–5 are Ranger-only so they are unimplemented by design. Within Phase 6a they are turned on one at a time — healing first (tune the coef in isolation), then damage-blocked — to keep coef tuning a single-variable problem.

Every shaped event is applied as `team_reward = own_events − enemy_events`. This symmetrization preserves zero-sum at the reward level, which in turn preserves `V_B ≈ −V_A` and avoids per-role reward engineering complexity.

### Why these magnitudes

Under worst-case shaping accumulation — in Phases 1–5, full-round objective hold + many kills; in Phase 6a+, also useful healing and damage-blocked events — shaping sums to ~3.0. Terminal is ±10.0. The cap ensures terminal always dominates, so agents cannot learn to lose-but-farm-shaping.

Note: objective shaping is **0.01 per second while controlling the objective** (computed in code as `0.01 * score_ticks_delta / TICK_HZ`). Under game-design §3's "+1 score point per second while controlling" rule, this is equivalent to ~0.01 per score point as an approximate-equivalent framing. Max over a full-round hold ≈ 1.8, already comfortably under the shaping cap.

### Optional probe shaping (distance-to-objective)

`RewardCalculator` accepts an opt-in `distance_shaping_coef` kwarg (default `0.0` — off). When positive, each decision applies an additional per-team term:

```
+coef × (dist_enemy − dist_self)
```

where `dist_*` is the own-team-frame normalized distance from the hero's position to arena center (i.e., the objective location; same `own_position` field as the actor obs). The term is zero-sum symmetrized — team B sees the negation after teams swap — so the `V_A ≈ −V_B` invariant holds. It passes through the same `[-3.0, +3.0]` per-episode clip as other shaping.

This exists for probes where the canonical event-triggered shaping is too sparse for random-init exploration to discover the cap (notably `phase3_ranger_noop_probe.yaml`, where a motionless opponent never triggers kill/score events so the base reward is event-free until the agent independently discovers cap-sitting). Typical values: `0.005–0.01` while ramping a new scenario; `0.0` for baseline / gate-clear runs.

This is a curriculum lever, not part of the canonical reward. It should be zero or annealed to zero before any run treated as a gate-clear result.

### Anti-hack guardrails

- Shaped-reward magnitude clipped per-episode; terminal reward always dominates
- No reward for raw damage without context
- No reward for overhealing full-HP allies
- **Objective-score reward is always awarded when team score increases**, even if only one ally is alive. Score is part of winning; suppressing it would create a weirder exception than it fixes. No extra per-agent reward for merely standing on point. A diagnostic metric tracks solo objective time while allies are dead, so stagger-feeding behaviors can be detected in eval.
- No reward for shield-damage farming
- Periodic evaluation with shaping disabled — if win rate collapses, shaping is too strong

### Team spirit (credit-assignment lever)

Introduced at **Phase 4** (the first multi-agent phase) and carried forward through the remaining phases. `team_spirit ∈ [0.0, 1.0]` is a scalar hyperparameter that interpolates each agent's reward between purely individual and purely team-average:

```
r_agent_i = (1 - team_spirit) * r_individual_i + team_spirit * mean_j(r_individual_j)
```

Where `r_individual_i` is the existing per-agent reward after terminal + shaping + clipping. At `team_spirit = 0` every agent optimizes only its own slice of the shaped reward; at `team_spirit = 1` all teammates see the identical team-average signal.

This is the OpenAI Five credit-assignment lever. It is **not** a substitute for the centralized critic (§3) — the critic reduces variance on the value target, while `team_spirit` shapes what "success" means for each agent's policy gradient. Both are needed.

**Ramp schedule:** start Phase 4 at `team_spirit = 0.3` and ramp linearly to `1.0` over the first ~30% of training, then hold (OpenAI Five-style — once team coordination is the dominant signal, individual credit becomes vestigial and adds variance). Early training wants enough individual signal to discover basic kit usage; late training wants every teammate to optimize the same coordinated outcome. Log `team_spirit` in the tensorboard run; treat it as a first-class hyperparameter, not a constant.

**Applies to shaped rewards only.** The ±10 terminal reward is already a team-outcome signal (win/loss is team-defined), so interpolating it against its own mean is a no-op. Implementation applies the mix to the shaped component before the per-episode `[-3.0, +3.0]` clip.

**Per-agent attribution.** Each agent's pre-mix individual reward `r_individual_i` comes from per-slot kill/death attribution (sim exposes `kills_by_slot` / `deaths_by_slot`), with score-tick rewards split among teammates by their per-tick on-point share. Enemy-team mirror events (their kills, their score) are subtracted *uniformly across own slots* — the killer's bonus already credits one team, the victim's penalty already debits the other, so a per-slot enemy-mirror term would double-count. Because the default `kill_bonus == death_penalty`, summing per-agent rewards over a team exactly reproduces the team-scalar reward used pre-Phase 4, so the cumulative `[-3.0, +3.0]` clip operates on the team sum and preserves the existing per-episode magnitude cap. The clip binds first; team_spirit's mixin is invariant under team mean and therefore commutes with it.

## 6. Training curriculum

Phases are gates. **Each phase declares explicit exit criteria below — metrics and thresholds, not vibes.** Do not proceed until the prior phase clears every listed gate; "looks stable on the loss curve" is not sufficient and compounds badly across phases. Where a gate references a baseline from a prior phase (e.g., "≥ Phase 4 win rate"), record that baseline in the phase result log so subsequent regression checks are anchored to a real number.

**Principle: ladder up complexity.** The first learning run should not have CNN + attention + GRU + hybrid action space + MAPPO + self-play all at once. If it fails, the failure mode is unidentifiable. Start with the smallest learning setup that can fail, prove it works, then add one dimension of complexity at a time.

**Also: fixed map first, randomized map later.** Per-episode map randomization (game-design §5) is a capability of the sim from day one but should be **disabled** during early phases. Enable it only once the phase in question has solved the fixed map. Randomization is an anti-overfitting measure; adding it before the policy can solve anything just slows learning.

### Ladder

**Phase 0 — Sim determinism smoke test.** No learning. Scripted bot vs scripted bot. Run the same seed twice, assert identical trajectories bit-for-bit. Catches determinism bugs before they can poison training. Gates to exit:
- Same seed, same binary, two intra-process runs → `state_hash` matches every tick (dense golden mode).
- Two different seeds → hashes differ within the first 100 ticks (sanity: hashing is actually data-dependent).
- Golden-replay CI job is green and runs on every commit touching sim/replay/obs code.

**Phase 1 — Feedforward PPO, flat obs, 1v1 Ranger.** No RNN, no attention, no grid — a flat vector observation and a small MLP. Self-play (or scripted opponent) is fine here since the state space is small. Fixed map. Goal: validate the entire learning pipeline (env, PPO, logging) on the easiest possible version of the problem. Gates to exit:
- Win rate vs `walk_to_objective` scripted bot ≥ 90% (this opponent does not shoot; failing this means the pipeline, not the policy, is broken).
- No NaN / inf in gradient norms, value loss, or returns across the run.
- Ability-on-cooldown activation rate (§11) trends toward 0.
- `fire-while-dead rate` (§11) trends toward 0.
- Mean episode reward is stable (no decay) across the final 200 updates.
- Replay inspection (Phase 3.5 viewer if available, otherwise per-tick log dump): policy approaches and stands on the objective; no obvious reward hack.

**Phase 2 — Recurrent PPO on a 1v1 memory toy.** Not the real game — a stripped-down environment that *provably requires* recurrence (e.g., cue visible for 10 ticks, must act on it 30 ticks later). Purpose: validate GRU training (hidden state handling, BPTT, rollout/training consistency) in isolation from game complexity. This is the "memory sanity test" from §10 turned into a phase. Gates to exit:
- Recurrent policy reaches ≥ 90% of by-construction optimal return.
- **Feedforward control** on the same env tops out below the recurrent policy by a clear margin (≥ 30 percentage points of optimal). If the feedforward variant solves it, the env doesn't actually require memory and the test is invalid.
- Hidden-state-zeroed-on-episode-boundary leak test passes (§10).
- Rollout-vs-training hidden-state determinism test passes (§10): identical `(obs_seq, h0)` produces identical logits in both passes.

**Phase 3 — Recurrent PPO, 1v1 Ranger, flat obs.** Bring recurrence into the real game at minimal scale. Still flat obs, still fixed map. Gates to exit:
- Win rate vs `basic` firing scripted bot ≥ 60%.
- **RNN-ablation eval**: re-run inference with the GRU hidden state frozen at zero each step (effectively feedforward). Win rate vs `basic` should drop by ≥ 10 percentage points — proves memory is doing work, not decorative.
- Behavioral metrics: objective contest time > 0; kill rate per episode > 0; ally-deaths-while-isolated low (this is 1v1 so there is no ally, but the metric infrastructure should exist).
- Out-of-ammo fire rate, ability-on-cooldown rate, fire-while-dead rate all near 0.
- Final checkpoint preserved as a warm-start source for Phase 4.

**Phase 3.5 — Minimal replay/debug viewer gate.** Before Phase 4 scale-up,
build the smallest useful viewer/debugger: draw the arena, objective, agents,
HP, facing, shots, deaths/respawns, score, cap progress, and reward components
for an eval episode. Scrubbing a recorded replay is ideal; a deterministic
live/eval playback is acceptable as the first cut. This is a debugging gate,
not a polished human-play client. Gates to exit:
- Viewer renders a Phase 3 eval replay end-to-end without crashes or visual glitches.
- All listed overlay elements present and readable: arena, objective, agent positions + facing, HP bars, shots, deaths/respawns, score, cap progress, per-step reward components.
- The developer can sit down with the viewer and identify at least one specific behavior in a Phase 3 replay (reward hack candidate, weird movement pattern, ability misuse, etc.) — if you can't extract insight from it, it is not yet a debugging tool.
- Replay file format documented in `replay_format.md` and round-trips cleanly (load → render → identical to live playback).

**Phase 4 — Recurrent MAPPO, 3v3 Ranger, flat obs.** Introduce multi-agent
training and the centralized critic while holding hero diversity constant.
All six slots run Ranger so the phase isolates the multi-agent / CTDE delta.
Still flat obs, still fixed map. Gates to exit:
- Win rate vs `basic` 3v3 scripted bot ≥ 50% (per the journal, this was the first non-loss breakthrough; treat it as the floor, not a target).
- Per-agent reward attribution: summing `r_individual_i` over a team equals the team-scalar reward used in pre-Phase 4 logic to within a documented tolerance (§5; regression test required).
- Gradient-mask test green: for current-vs-snapshot rollouts, the loss batch contains only `is_learning == True` steps (§7).
- Hidden-enemy leak tests (§10) green.
- Behavioral metrics: ally-deaths-while-isolated lower than naïve 3v Phase-3-clone baseline (proves teammates coordinate at least loosely); objective contest time > 0 with all three agents present on point in at least some episodes.
- `team_spirit` ramp logged in the run; final value documented for downstream phases.
- Best-eval checkpoint preserved (per journal lesson — PPO oscillates around breakthroughs, so don't rely on `latest`).

**Phase 5 — Add entity attention.** Swap flat obs for entity-tokens + attention pooling. 2v2 or 3v3. Fixed map. No grid yet. Gates to exit:
- Win rate vs `basic` 3v3 ≥ Phase 4 baseline (regression check — the encoder swap should not hurt).
- **Attention-ablation eval**: replace attention pooling with mean-pooling over the same per-entity MLP outputs. Win rate should drop materially; if it doesn't, attention is decorative and the encoder needs debugging before Phase 6.
- Variable-entity-count edge cases pass (§10): all enemies dead, all teammates dead, 0 visible enemies, maximum visible enemies — no crashes, no NaN logits.
- Attention masking is sound: dead-agent entity tokens contribute zero post-mask (unit test).
- Hidden-enemy leak tests still green after the obs-builder rewrite (§10).

**Phase 6 — Heterogeneous roster, then grid, then teamfight-emergence evaluation.** Phase 6 is split into three sub-phases to keep each delta debuggable. The original "add grid + add new heroes + add new shaping + claim emergent teamfights" formulation bundled four deltas at once, which violates the one-delta-at-a-time rule. Run them serially:

- **Phase 6a — Heterogeneous roster on the Phase 5 encoder.** Introduce Vanguard + Mender alongside Ranger. 3v3 Vanguard / Ranger / Mender, fixed map, full vision. Keep the Phase 5 encoder stack (entity tokens + attention pooling, **no grid yet**). The entity-token schema exercises non-trivially for the first time — `active allied Barrier`, `active enemy Barrier`, and `Mender beam relationship` tokens become non-zero — and the hero embedding now spans three roles. New per-role shaping terms (§5) turn on **one at a time**: healing-delivered first, tuned in isolation; then damage-blocked once healing has settled. Gates to exit 6a:
  - Win rate vs `basic` scripted opponent ≥ Phase 5 baseline.
  - `fire-while-shielding rate` (§11) trends toward 0 — proves Barrier/Warhammer mutual exclusion is being learned, not just sim-enforced.
  - Healing efficiency metric (§11) climbs and is not overheal-spam (overheal exclusion in the shaping is working as intended).
  - Mender beam target distribution favors damaged allies over random.
  - Ability-on-cooldown rates near 0 across all three kits.
  - Hidden-enemy leak tests (§10) green — the schema changed, re-run them.

- **Phase 6b — Add the egocentric grid + CNN encoder.** Concat a small CNN feature with the entity features. Roster, shaping, map, and opponent ladder all held constant from 6a; the only delta is the added grid input and CNN encoder branch. Gates to exit 6b:
  - Win rate vs `basic` ≥ 6a baseline (regression check — adding the grid should not hurt).
  - **Grid-ablation eval**: zero out the grid input at eval time and re-measure win rate. The drop should be material; if the policy plays equivalently without the grid, the CNN is decorative and the encoder needs debugging before moving on. This is the cheapest catch for "attention already captured everything the grid would have."
  - Positioning metrics improve over 6a baseline: cover usage near walls, line-of-sight-maintenance time, teammate-position variance during contested-cap moments.

- **Phase 6c — Teamfight-emergence evaluation (analysis, not a training phase). *OA5-analog milestone.*** This is the testable research claim: given the full encoder stack + heterogeneous roster + centralized critic + symmetrized shaping + `team_spirit` ramp (§5), do coordinated teamfights emerge from self-play without explicit communication? The information structure (team-shared/full vision, no pings, no learned comms, recurrent policy per agent, centralized critic) mirrors OpenAI Five's setup. Evaluate the 6b checkpoint against:
  - Spatial clustering during contested-cap moments (teammates within radius X for >Y% of the time, vs a random-policy baseline).
  - Role-conditional positioning: Vanguard front, Mender back, Ranger flexible.
  - Replay-level causal chains: count Barrier-up → Mender-beam-on-tank → Ranger-DPS-from-off-angle sequences across N sampled eval replays.
  - First-pick conversion rate > 50% (proxy for focus-fire being learned).
  - Comeback rate > 0 (not pure snowballing).

  If 6c fails, the fix is almost never "add fog of war." It is revisiting shaping coefs, opponent curriculum, or `team_spirit` ramp on the 6b checkpoint. Do not advance to Phase 7 until 6c passes — fog will not rescue a non-coordinated policy.

**Phase 7 — Partial observation.** Split into two sub-phases so the fog delta is incremental on a working Phase-6 policy, not a cold start:

- **Phase 7a — Team-shared fog of war.** Walls block line-of-sight, but visible-enemy sets are unioned across teammates before building each agent's observation (Dota / OA5 fog model). Goal: show teamfights survive partial observation of the *map* without yet requiring agents to infer what teammates see. This is the strict OA5 parity point. Gates to exit:
  - Win rate vs `basic` ≥ 80% of the Phase 6b baseline win rate (some fog cost expected; > 20% degradation means the fog rollout has a bug or needs more curriculum).
  - Phase 6c coordination metrics (spatial clustering during contests, role-conditional positioning) retained within a documented tolerance — coordination must survive the fog.
  - Last-seen enemy markers used by the policy: ablate the marker feature at eval time → win rate should drop measurably. If it doesn't, the policy is ignoring memory.
  - Leak tests green specifically for fog: critic still sees hidden enemies, actor does not, no obs delta when a fully-hidden enemy moves/aims/fires (§10).

- **Phase 7b — Per-agent fog of war.** Drop the team-shared union. Each agent sees only what *it* directly has line-of-sight to; the only cross-agent information channel is allies-through-walls (position/HP/alive-state) plus each agent's own last-seen enemy markers. This is the research-novel claim — teamfights survive genuine per-agent partial observation. *Note: Phase-1 heroes (Vanguard / Ranger / Mender) have no team-level reveal abilities (game-design §4), so coordination here must emerge from positioning alone.* Gates to exit:
  - Win rate vs `basic` ≥ 70% of the Phase 7a baseline.
  - Coordination metrics from 6c retained within tolerance; spatial-clustering metric in particular must not collapse (the claim under test is that positioning-only coordination survives per-agent fog).
  - Per-agent vs team-shared comparison documented: same eval suite run on both 7a and 7b checkpoints, with delta-table recorded as the headline result for the research-novel claim.
  - Leak tests green: each agent's obs depends only on *its own* visibility, not the union (regression test).

**Phase 8 — Map randomization.** Turn on per-episode wall randomization. Overfitting mitigation. Anneal in: start with a small pre-validated set of map variants and expand, rather than enabling full random generation cold. Gates to exit:
  - **Train/test generalization gap** ≤ 10 percentage points: win rate on a held-out set of map variants the policy was never trained on should be within 10pp of training-map win rate.
  - Win rate vs `basic` on the training-map distribution ≥ 80% of Phase 7b baseline (some randomization cost expected; large gaps mean curriculum needs more annealing).
  - No reward-hack regression in eval replays from sampled novel maps (the viewer is a first-class tool here; sample N replays and inspect).
  - Determinism preserved across seeded map generation (same seed → same map; cross-process golden replay test still passes).

**Phase 9 — Snapshot self-play league.** Switch opponent sampling to the snapshot pool (§7). Gates to exit:
  - **Win-rate matrix** computed across the snapshot pool: current policy wins ≥ 55% vs the oldest snapshot, ≥ 50% vs the most recent. A current-vs-oldest win rate below 50% means catastrophic forgetting.
  - Win rate vs the scripted-bot anchor suite (`walk_to_objective`, `basic`, `hold_and_shoot`, and a defensive-hold variant if available) ≥ Phase 8 baseline. The pool must not let the policy drift away from basic skills.
  - Gradient-mask invariant from Phase 4 still holds: snapshot-controlled trajectories are excluded from the PPO loss (§7 regression test passes against pool opponents).
  - No cyclic strategy detected: if the win-rate matrix shows A>B, B>C, C>A patterns dominating across recent snapshots, escalate to investigation before continuing.

**Phase 10 — Target slots, second heroes per role, and missing abilities.** Enable the `target_slot` action head and wire the first targeted diagnostic, Ranger Mark Target. Continue introducing alternative heroes (e.g., Warden alt-tank, Specter flanker DPS, a dedicated utility support) and fill in the remaining deferred Phase-1 abilities (a Mender alternate with Damage Boost; Vanguard's Charge / Fire Strike). Composition-specific strategy learning. Split per-hero introductions into sub-phases (10a Mark Target + target_slot head, 10b first alt hero, etc.) for the same one-delta-at-a-time reasons as Phase 6. Gates to exit each 10.x:
  - `target_slot` action distribution is non-degenerate: not always 0, not uniform — selection conditions on observed enemy state.
  - For Mark Target specifically: the marked enemy is more likely to be focused by teammates in the following N decisions than a random visible enemy (proves the targeted info is being used by the team).
  - Win rate vs `basic` ≥ Phase 9 baseline (no regression from adding heroes/abilities).
  - New hero kit ability-use rates non-trivial (>0) and ability-on-cooldown rates near 0 — proves the hero is being played, not ignored by the shared policy.
  - Leak tests green after each obs-schema change.

### Why this ordering matters

The phases are ordered so that when a phase fails to converge, the delta from the previous phase is a single component: RNN (Phase 2→3), multi-agent (3→4), attention (4→5), heterogeneous roster + per-role shaping (5→6a), grid (6a→6b), team-shared fog (6c→7a), per-agent fog (7a→7b), randomization (7b→8), snapshot pool (8→9). Debugging is a binary search on the delta. If you combine deltas, debugging is combinatorial and you'll burn weeks on a training run you can't diagnose.

**Each phase needs its own opponent curriculum, not just a single warm-start.** The Phase 4 journal (`docs/journal/reinforcement_learning_journal.md`) shows even a "small" 1v1→3v3 delta required a three-stage opponent ladder (`walk_to_objective` → `hold_and_shoot` → `basic`) with explicit shaping-coef schedules at each rung. Treat the per-phase plan as required output: opponent rungs, reward-coef values per rung, and update budgets per rung, written before the run starts. A single warm-start across phase boundaries is the exception, not the default.

### Team-relative coordinate normalization

All spatial features in both actor and critic observations are expressed in a **team-relative coordinate frame**. Team A's policy sees the map as if A is always on the "bottom" side; Team B's policy sees the mirrored view so B is also on the "bottom." Implementation:

```
if team == A:
    obs_pos = world_pos
else:  # team == B
    obs_pos = mirror(world_pos)         # flip across map center
    obs_aim = mirror_angle(world_aim)
```

Rationale: without this, shared policy weights must learn two entirely different observation distributions (one per team side), which doubles sample complexity for zero learning benefit. The sim itself runs in world coordinates; the mirroring is an obs-builder concern only.

## 7. Self-play and the snapshot pool

### Opponent sampling (starting mix)

```
70%  current policy vs current policy
20%  current vs random snapshot from pool
10%  current vs scripted bot (anchor)
```

### Gradient masking by opponent type

The trainer must explicitly mask gradients based on which agents are controlled by the learning policy vs a frozen policy. If a snapshot or scripted agent's trajectories are accidentally included in the PPO loss, the snapshot policy effectively gets trained as if it were the current one — this silently corrupts learning.

| Match type | Trajectories included in PPO loss |
|---|---|
| Current vs current (mirror) | **Both teams** — all 6 agents contribute gradients |
| Current vs snapshot | **Only current-policy agents** — snapshot agents' trajectories are discarded from the loss (but still used by the env to step opponents) |
| Current vs scripted bot | **Only the learning agents** — scripted agents never contribute to loss |

Implementation: tag each trajectory at rollout time with `is_learning` per agent per timestep. Mask the PPO loss to only include `is_learning == True` steps. Value-function loss follows the same mask. This is a common subtle bug — worth an explicit test that asserts the expected token counts in the loss batch.

### Snapshot policy

- Save policy every N updates (choose N such that ~20 snapshots span training)
- Pool size capped at 20, oldest-evicted — with a few preserved "strong historical" snapshots based on eval performance
- Periodic eval computes a win-rate matrix across the pool to detect cyclic strategies

### Why this matters

Latest-vs-latest only produces cyclic strategies and catastrophic forgetting. Older snapshots anchor the policy against regression. Scripted bots anchor basic skills that all-learned-play can drift away from. OpenAI Five's 80/20 current/past mix is the template; this is the smaller-scale version.

## 8. Starting hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | Recurrent MAPPO |
| Optimizer | Adam |
| γ | 0.997 |
| GAE λ | 0.95 |
| PPO clip ε | 0.10 |
| PPO epochs per batch | 4 |
| `minibatch_size` / `num_minibatches` | `minibatch_size` = segments per minibatch; `num_minibatches = ceil(num_segments / minibatch_size)` |
| Entropy coef | 0.01, slow anneal |
| Value loss coef | 0.5 |
| Gradient clipping | 0.5 |
| Value normalization | yes (per-rollout return-space z-scoring; value target normalized and value clip applied in normalized space) |
| Advantage normalization | yes |
| RNN type | GRU |
| RNN hidden size | 256 |
| Unroll length | 64–96 |
| Action repeat | 2–3 ticks |
| Parallel envs | 128 |

Conservative by intent. MAPPO-paper findings that drove these: keep PPO clip well under 0.2, limit epochs on hard problems, normalize values, avoid aggressive minibatching on on-policy data.

The table above targets the Phase 4+ MAPPO configuration at scaled training; Phase 3 runs single-agent recurrent PPO with the same hyperparameter profile minus the centralized critic. Several parameters have phase-specific overrides for the short-horizon Phase 1–3 ladder (see `experiments/configs/phase3/baseline/phase3_ranger_recurrent.yaml`):

- **γ = 0.99 for Phase 1–3** short-horizon episodes (30 s rounds ≈ 100 decisions; 0.99 gives effective horizon ≈ 100). Ramp to γ = 0.997 at Phase 4+ when round length grows.
- **RNN hidden size = 64 for Phase 1–3** flat-obs runs (31-dim obs, shorter memory horizon). Scale up toward 256 with observation complexity at Phase 5+ and to 512 if underfitting.
- **Unroll length = full episode for Phase 1–3** — episodes are ≤ 100 decisions at 10 Hz over 30 s rounds, which naturally brackets the 64–96 target; the trainer does not truncate BPTT within a segment. Truncate to 64–96 at Phase 5+ once episodes grow.
- **Parallel envs = 16 for Phase 3** on the interactive-training machine; 128 is the scaled-training target for Phase 4+ and beyond.

## 9. Tech stack

**C++ simulation core + raylib viewer + Python trainer.**

### Sim (C++)
- Pure deterministic game state update
- No rendering, no wall-clock time, no networking inside the sim core
- Exposed to Python via **`pybind11`** bindings (or `nanobind` as a lighter alternative)
- Batched interface: accept actions for N parallel envs, return obs/rewards/dones for N
- Shared-memory numpy arrays on the boundary where possible (zero-copy via `py::array_t` / buffer protocol)
- Single logical process; sim is internally thread-safe to support parallel env stepping
- Build system: **CMake**. Modern C++ (C++20) is fine; avoid exotic features that complicate cross-compilation to WASM if the browser-viewer stretch is pursued.

### Determinism discipline (C++-specific)

See `determinism_rules.md` for the canonical compiler/build flags, source-level rules, PRNG helpers, `state_hash()` manifest, and golden-replay policy. **Do not duplicate that list here.** At a summary level: MVP guarantees same-machine, same-binary, same-compiler reproducibility only; a WASM viewer is treated as visual-only and not expected to replay bit-identically across targets.

### Viewer (C++, raylib)
- **raylib** as the rendering library — simple immediate-mode 2D, tiny dependency footprint, fits the project's minimalist aesthetic
- Reads the same deterministic sim state — the viewer links against the sim as a library, no duplicate game logic
- Must support debug overlays: per-agent vision cones, fog of war, raycasts, shields, cooldowns, last-seen ghosts, reward event flashes, current weapon state (Mender), Vanguard barrier state
- Human input path: keyboard + mouse → action struct → sim (same action schema the RL agents use)
- Stretch: compile viewer + sim to WASM via emscripten for a browser-shareable demo (raylib supports this). Replay exactness is **not** guaranteed across native ↔ WASM; treat the browser viewer as visual-only. See `determinism_rules.md` and `replay_format.md`.

### Trainer (Python / PyTorch)
- **Start with CleanRL-style single-file PPO** (feedforward, flat obs) — the smallest trainer that can run. This is Phase 1 of the curriculum.
- Ladder up: add RNN support for Phase 2–3, centralized-critic / MAPPO for Phase 4, entity attention + grid encoders for Phases 5–6. Each addition is a diff on the prior trainer, not a rewrite.
- Not TorchRL or RLlib — those are over-abstract for a non-standard algorithm and make debugging harder.
- 128 parallel envs via batched C++ sim calls (preferred) or Python multiprocessing (fallback)
- Snapshot pool management (Phase 9+)
- Evaluation harness (scripted-bot opponents, behavioral metrics)
- Metrics logging — wandb or tensorboard

### Rationale summary

C++ is ergonomic for the sim's complex systems (LoS raycasting, ability state machines, per-agent visibility) and gives direct control over determinism, memory layout, and cache behavior. **raylib** is the right viewer library for this project: small, immediate-mode, 2D-first, trivial to link against a C++ sim, and compiles cleanly to WASM for the browser-demo stretch. Human-play client is trivial because the viewer and sim are the same binary — no marshaling, same data structures.

Throughput at this scale is not the bottleneck (~10–50k sim steps/sec at 128 parallel envs is plausible — enough for a 100M-step baseline in days, not weeks). JAX (JaxMARL) was considered and rejected: the sim's data-dependent control flow and variable-count entity state are painful to express in JAX, and the human-play requirement makes a GPU-resident sim awkward.

The main C++-specific hazard is floating-point determinism (addressed above under "Determinism discipline"); it is strictly more work than Rust here, but fully achievable with discipline.

## 10. Implementation dangers

### Hidden-enemy info leak prevention (highest priority)

Any leak of hidden enemy state into the actor silently degrades the entire research contribution with no error. Required explicit tests, run in CI:

```
- actor_obs does not change when a hidden enemy:
      moves
      aims
      changes cooldowns
      reloads
      swaps weapon
      fires                 # unless explicit hidden-fire perception is enabled
- actor_obs changes only when:
      a hidden enemy becomes visible via LoS
      a public event fires: objective contested flag flips,
          kill feed entry, score change, objective ownership change
- critic_state DOES see hidden enemies (sanity check the centralized side works)
- actor and critic observation builders are in separate top-level code paths
      (shared low-level utilities must not touch hidden state — see
       observation_spec.md)
- RNN hidden state is reset to zero on episode boundary
- RNN hidden state is not reused across PPO epochs — stored at rollout, replayed exactly during training
```

"Public events" is an explicit, small allowlist. There is no generic "enemy triggered an event" loophole: any new event that could reveal hidden enemy presence needs an explicit design decision and an updated leak test. In particular, **muzzle traces from hidden enemy fire are renderer-only in MVP** — they appear in the omniscient viewer but are not part of any actor observation. A deliberate hidden-fire perception ablation (approximate direction / distance band, no exact position) may be added later.

### Recurrent MAPPO silent failure modes

Any of these silently degrades training to effectively-feedforward with no error:

- Stale hidden state across PPO epochs
- BPTT truncation boundary `detach()` bugs
- Episode reset not zeroing hidden state
- Hidden state divergence between rollout sampling pass and training pass

### Memory sanity test

**Before scaled training, build a toy environment that provably requires memory** — e.g., "a cue appeared 2 seconds ago and has since disappeared; you must act on it now." If the recurrent policy can't solve that toy, recurrent MAPPO is broken and will silently fail on the real game. This test catches ~80% of the failure modes above. In the curriculum, this is **Phase 2**.

### Golden replay determinism tests

Determinism bugs in the sim silently invalidate everything downstream (replay, eval, reproducibility). See `determinism_rules.md` for canonical policy. Summary:

```
- Sim golden replay: feeds a recorded canonical action stream into a
  fresh Sim and asserts state_hash matches every sim tick
  (hash_mode = dense_golden). Does NOT call bot policy code —
  bot refactors must not break this.
- Bot regression test (separate): runs scripted bots from seed/config
  and checks bot behavior. Allowed to change when bot logic changes.
- Intra-process determinism: run the same seed twice in the same
  process; hashes must match.
- Cross-process determinism (future): spawn two fresh processes; compare.
- Regenerating the golden requires explicit human confirmation.
```

Running these in CI prevents a class of "the training loop got weirdly unstable three weeks ago and we don't know why" incidents.

### Variable entity counts

Attention-over-entities requires explicit masking for dead agents and variable visible-enemy counts. Test edge cases:

- All enemies dead
- All teammates dead
- 0 visible enemies
- Maximum visible enemies

### Reward-hack detection

Build replay viewer before training at scale. Periodically watch replays of the current policy; look for:

- Objective ignoring
- Kill-farming
- Shield-damage farming
- Healing-overheal spam
- Spawn-camping

## 11. Evaluation protocol

Self-play win rate always trends to 50%. It is an **uninformative** metric. Non-self-play evaluation is required from day one.

### Anchored baselines
- Win rate vs fixed scripted-bot suite (walk-to-objective bot, shoot-visible-enemy bot, defensive-hold bot)
- Win rate vs frozen reference policies (snapshots from prior phases)

### Behavioral metrics

Track continuously during training:

- Objective contest time
- Average teamfight duration
- Ally deaths while isolated (measures staggering / feeding)
- Team grouping distance (variance of teammate positions)
- Support survival time during fights (measures peel)
- Damage dealt into shields vs damage past shields
- Healing efficiency (heal delivered / heal potential, excluding overheal)
- Team-reveal ability value (expected fight-outcome swing in the seconds after a reveal)
- First-pick conversion rate (fights won conditional on scoring the first kill)
- Comeback rate (wins from a score deficit)

### Invalid-action and cooldown-waste metrics

A common failure mode is the policy spamming unavailable abilities. Track:

- **Ability-on-cooldown activation rate** — fraction of `ability_1` / `ability_2` rising edges issued while the ability was on cooldown. Should trend toward ~0 with training. High values indicate the cooldown observation feature is not being used.
- **Fire-while-dead rate** — fraction of `primary_fire == 1` emissions while the agent is dead. Should be near 0.
- **Fire-while-shielding rate** (Vanguard) — fraction of `primary_fire == 1` emissions while Barrier is active. Should trend to 0 once the policy learns the mutual-exclusion rule.
- **Out-of-ammo fire rate** (Ranger, when ammo is introduced) — fraction of primary_fire rising edges while magazine empty.

These are not reward signals; they are diagnostic. A policy that keeps spamming cooldowns is failing to condition on its own state, which usually indicates an observation or architecture bug rather than a reward-shaping bug.

### Human-play checkpoints

At every major phase transition, play the latest policy yourself for ~10 games. No substitute for "does it feel like a real player?"

### Replay inspection

Record replays of every eval game. The replay viewer is a first-class tool, not an afterthought — this is where emergent behavior gets debugged.

## 12. Baselines and ablations

For research legitimacy, plan to run (after Phase 5 is stable):

| Comparison | Question |
|---|---|
| MAPPO recurrent vs IPPO recurrent | Is the centralized critic helping? |
| MAPPO recurrent vs MAPPO feedforward | Is memory actually learning useful things? |
| Partial-obs actor vs full-info actor | What's the cost of fog on the decentralized side? |
| Shared policy + hero embedding vs per-role policies | Does parameter sharing help or hurt? |
| Snapshot self-play vs latest-only self-play | Does the opponent pool matter? |
| Symmetrized shaping vs per-role asymmetric shaping | Does reward symmetrization matter? |
| Per-episode map randomization vs fixed map | Overfitting quantification |

**Most research-valuable:**
1. **MAPPO recurrent vs MAPPO feedforward** — does the LSTM/GRU actually learn to track hidden enemies and cooldowns, or is it decorative?
2. **Centralized critic vs independent critic** — does MAPPO help beyond IPPO at 3v3 scale?
3. **Snapshot self-play vs latest-only** — the canonical self-play stability question.
