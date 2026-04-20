# Xushi2 — RL Design

**Project:** Xushi2 (虚实 2)
**Date:** 2026-04-20
**Status:** MVP design (Phase 1)

Companion to `2026-04-20-game-design.md`. This document covers the training algorithm, observation and action spaces, neural architecture, curriculum, self-play setup, evaluation, and implementation hazards.

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
- **`target_slot` is omitted in Phase 1–9.** All Phase 1 heroes use direction-based or aim-cone targeting: Barrier points where Vanguard aims, Combat Roll dashes along movement direction, Mender's Beam auto-locks the nearest ally in the aim cone, Mender's Tether zips to the ally in the aim cone. No ability requires explicit ally-slot selection. At Phase 10 (second heroes), ally-targeted abilities (Warden's Projected Guard, a classic Burst Heal) enable `target_slot` via attention over the entity tokens already encoded for observation, with a valid-target mask (see `observation_spec.md`).
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
| Ranger | ability_2 | — (deferred) | — |
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
+0.01  per objective score point own team gains
-0.01  per objective score point enemy team gains
+0.25  enemy kill
-0.25  ally death
+small useful healing delivered (overheal excluded, capped)
+small damage blocked near ally (Vanguard shield, capped)

total non-terminal shaping clipped to [-3.0, +3.0] per episode per team
```

Every shaped event is applied as `team_reward = own_events − enemy_events`. This symmetrization preserves zero-sum at the reward level, which in turn preserves `V_B ≈ −V_A` and avoids per-role reward engineering complexity.

### Why these magnitudes

Under worst-case shaping accumulation (full-round objective hold + many kills + useful healing), shaping sums to ~3.0. Terminal is ±10.0. The cap ensures terminal always dominates, so agents cannot learn to lose-but-farm-shaping.

Note: objective shaping is per **score point** gained, not per tick. Since scoring is +1/sec under the rules in game-design §3, this is effectively 0.01/sec while controlled. Max over a full-round hold ≈ 1.8, already comfortably under the shaping cap.

### Anti-hack guardrails

- Shaped-reward magnitude clipped per-episode; terminal reward always dominates
- No reward for raw damage without context
- No reward for overhealing full-HP allies
- **Objective-score reward is always awarded when team score increases**, even if only one ally is alive. Score is part of winning; suppressing it would create a weirder exception than it fixes. No extra per-agent reward for merely standing on point. A diagnostic metric tracks solo objective time while allies are dead, so stagger-feeding behaviors can be detected in eval.
- No reward for shield-damage farming
- Periodic evaluation with shaping disabled — if win rate collapses, shaping is too strong

## 6. Training curriculum

Phases are gates. Do not proceed until the prior phase produces stable, interpretable behavior.

**Principle: ladder up complexity.** The first learning run should not have CNN + attention + GRU + hybrid action space + MAPPO + self-play all at once. If it fails, the failure mode is unidentifiable. Start with the smallest learning setup that can fail, prove it works, then add one dimension of complexity at a time.

**Also: fixed map first, randomized map later.** Per-episode map randomization (game-design §5) is a capability of the sim from day one but should be **disabled** during early phases. Enable it only once the phase in question has solved the fixed map. Randomization is an anti-overfitting measure; adding it before the policy can solve anything just slows learning.

### Ladder

**Phase 0 — Sim determinism smoke test.** No learning. Scripted bot vs scripted bot. Run the same seed twice, assert identical trajectories bit-for-bit. Catches determinism bugs before they can poison training.

**Phase 1 — Feedforward PPO, flat obs, 1v1 Ranger.** No RNN, no attention, no grid — a flat vector observation and a small MLP. Self-play (or scripted opponent) is fine here since the state space is small. Fixed map. Goal: validate the entire learning pipeline (env, PPO, logging) on the easiest possible version of the problem.

**Phase 2 — Recurrent PPO on a 1v1 memory toy.** Not the real game — a stripped-down environment that *provably requires* recurrence (e.g., cue visible for 10 ticks, must act on it 30 ticks later). Purpose: validate GRU training (hidden state handling, BPTT, rollout/training consistency) in isolation from game complexity. This is the "memory sanity test" from §10 turned into a phase.

**Phase 3 — Recurrent PPO, 1v1 Ranger, flat obs.** Bring recurrence into the real game at minimal scale. Still flat obs, still fixed map.

**Phase 4 — Recurrent IPPO or MAPPO, 2v2, flat obs.** Introduce multi-agent training and the centralized critic. Compositions to try: tank + damage, then damage + support. Still flat obs, still fixed map.

**Phase 5 — Add entity attention.** Swap flat obs for entity-tokens + attention pooling. 2v2 or 3v3. Fixed map. No grid yet.

**Phase 6 — Add the egocentric grid.** Concat a small CNN feature with the entity features. 3v3 Vanguard / Ranger / Mender. Fixed map. Full vision.

**Phase 7 — Partial observation.** Enable per-agent fog of war on the 3v3 setup. Goal: scouting, rotations, emergent coordination through positioning. *Note: Phase 1 heroes (Vanguard/Ranger/Mender) have no team-level reveal abilities — see game-design §4. Coordination here must emerge from the ally-through-walls vision channel alone, making this the hardest form of the partial-observation problem.*

**Phase 8 — Map randomization.** Turn on per-episode wall randomization. Overfitting mitigation.

**Phase 9 — Snapshot self-play league.** Switch opponent sampling to the snapshot pool (§7).

**Phase 10 — Second heroes per role and missing abilities.** Introduce alternative heroes (e.g., Warden alt-tank, Specter flanker DPS, a dedicated utility support) and fill in deferred Phase-1 abilities (Ranger's ability_2 candidate — Flashbang or similar; a Mender alternate with Damage Boost; Vanguard's Charge / Fire Strike). `target_slot` action enabled. Composition-specific strategy learning.

### Why this ordering matters

The phases are ordered so that when a phase fails to converge, the delta from the previous phase is a single component: RNN (Phase 2→3), multi-agent (3→4), attention (4→5), grid (5→6), partial obs (6→7), randomization (7→8), snapshot pool (8→9). Debugging is a binary search on the delta. If you combine deltas, debugging is combinatorial and you'll burn weeks on a training run you can't diagnose.

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
| PPO epochs per batch | 5 |
| Minibatches per epoch | 1–2 |
| Entropy coef | 0.01, slow anneal |
| Value loss coef | 0.5 |
| Gradient clipping | 0.5 |
| Value normalization | yes |
| Advantage normalization | yes |
| RNN type | GRU |
| RNN hidden size | 256 |
| Unroll length | 64–96 |
| Action repeat | 2–3 ticks |
| Parallel envs | 128 |

Conservative by intent. MAPPO-paper findings that drove these: keep PPO clip well under 0.2, limit epochs on hard problems, normalize values, avoid aggressive minibatching on on-policy data.

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