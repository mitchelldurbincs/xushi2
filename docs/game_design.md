# Xushi2 — Game Design

**Project:** Xushi2 (虚实 2)
**Date:** 2026-04-20
**Status:** MVP design (Phase 1)

## 1. Project framing

Xushi2 is a deterministic, top-down 2D, 3v3 control-point team shooter designed as a multi-agent reinforcement learning environment that is also playable by humans. Teamfight behavior should emerge from a small number of interacting systems — fog of war, line-of-sight, cooldown timing, and role interdependence — without hero ultimates, projectile physics, or verticality.

Named after the Sun Tzu concept of *虚实* (feint and substance, emptiness and fullness): information warfare and deception under partial observation.

**Design philosophy:** the complexity budget is spent on **team interaction**, not content volume. One map topology, three heroes, one mode, deterministic simulation, great replay tools.

Research framing: *A deterministic multi-agent RL benchmark for studying teamfight coordination under partial observability, built as a small top-down 2D hero shooter with 3v3 role-based combat, fog of war, fixed-tick simulation, and objective-driven win conditions.*

## 2. Design pillars

**Pillar 1 — Teamfight emergence, not mechanical overload.** Behaviors like regrouping, scouting, rotation, focus fire, peel, and disengage should emerge from the interaction of visibility, cover, objective, roles, and cooldowns. No ultimates, no inventories, no complicated terrain in MVP.

**Pillar 2 — Deterministic simulation first, rendering second.** The engine is a headless deterministic core. Rendering only visualizes; it never affects simulation. This enables fast rollouts, reproducibility, exact replay, and debuggable emergent behavior.

**Pillar 3 — Short episodes with clear win conditions.** 180–240 second rounds, first team to 100 objective score. Training signal arrives quickly; human matches stay tight.

## 3. Core game spec

| Property | Value |
|---|---|
| Team size | 3v3 |
| Composition | 1 tank, 1 damage, 1 support per team |
| Simulation tick | 30 Hz, fixed |
| Action repeat | 2–3 ticks (≈10–15 Hz effective decisions) |
| Round length | 180–240 seconds |
| Win condition | First team to 100 objective score, or higher score at timeout |
| Objective unlock | 15 seconds after round start |
| Respawn | 6–8 seconds, wave-respawn preferred |
| Movement | Continuous 2D |
| Aim | Angular delta, ±45° per **policy decision** (not per sim tick) |
| Weapons | Hitscan only |
| Physics | Fixed tick, deterministic |
| Mode | Control point (single neutral circle) |

**Escort mode** is deferred out of MVP. The engine should be built to support a pluggable "mode" abstraction, but escort is not implemented in Phase 1.

### Objective scoring rules (summary)

- **+1 score per second** while the point is controlled by one team and uncontested
- **No score while contested** (at least one living hero from each team inside the point)
- **Occupancy count does not matter**: one hero on point scores or captures at the same rate as three
- **Control flips after capture progress reaches 1.0** (not instantly)
- **Objective unlocks 15s after round start**; before that, the state machine below is paused

### Control-point state machine

The objective has four state variables, all integer-typed for exact determinism at 30 Hz. The UI converts to floats for display; sim math stays in integer ticks.

```
owner              ∈ {Neutral, A, B}                   # who currently holds the point
cap_team           ∈ {None, A, B}                      # which team currently has capture progress
cap_progress_ticks : integer, 0..CAPTURE_TICKS         # cap_team's progress in sim ticks
team_score_ticks   : integer per team, 0..WIN_TICKS    # win progression in sim ticks
```

Per-tick occupancy (only *living* heroes inside the capture circle count):
```
present_A = (count of living A heroes on point) > 0
present_B = (count of living B heroes on point) > 0
contested = present_A AND present_B
```

Constants (at TICK_HZ = 30):
```
CAPTURE_TICKS  = 8  · TICK_HZ = 240     # 8 seconds to fully capture from 0
DECAY_TICKS    = 8  · TICK_HZ = 240     # 8 seconds to fully decay from 1 while empty
WIN_TICKS      = 100 · TICK_HZ = 3000   # 100 seconds of scoring = win
SCORE_STEP     = 1                      # +1 score tick per sim tick while scoring
```

Display values are derived:
```
display_cap_progress = cap_progress_ticks / CAPTURE_TICKS   ∈ [0.0, 1.0]
display_score         = team_score_ticks / TICK_HZ          ∈ [0.0, 100.0]
```

All state transitions below are written in integer-tick arithmetic. There is no fractional `+= 1/30` accumulator; rounding edge cases do not exist, and the match outcome is exact and deterministic.

Initial state at round start: `owner = Neutral`, `cap_team = None`, `cap_progress_ticks = 0`, `team_score_ticks = 0` for both teams. Scoring is frozen during the first 15 seconds (objective lock). After unlock, each tick applies exactly one of the following five cases:

#### Case 1 — Contested (both teams present)

```
no scoring this tick
cap_progress_ticks freezes (no change)
cap_team unchanged
```

Freezing (rather than resetting) rewards defending a near-complete capture against brief contests without punishing the attacker's commitment.

#### Case 2 — Empty (neither team present)

```
no scoring this tick
cap_progress_ticks = max(0, cap_progress_ticks - 1)
if cap_progress_ticks == 0:
    cap_team = None
```

Ownership (`owner`) does **not** change when the point goes empty. A team that captured the point keeps it after stepping off.

#### Case 3 — Uncontested occupation by the owner

```
precondition: present_X only, and owner == X, and X ∈ {A, B}

team_X_score_ticks += SCORE_STEP
if team_X_score_ticks >= WIN_TICKS:
    X wins immediately; round ends at this tick

# Clean up any stale opposing-team capture progress:
if cap_team == opponent(X):
    cap_progress_ticks = max(0, cap_progress_ticks - 1)
    if cap_progress_ticks == 0:
        cap_team = None
else:
    cap_progress_ticks = 0
    cap_team = None
```

#### Case 4 — Uncontested occupation by a non-owner team

```
precondition: present_X only, and owner != X, and X ∈ {A, B}
(owner is either opponent(X) or Neutral)

no scoring this tick

if cap_team == X:
    cap_progress_ticks = min(CAPTURE_TICKS, cap_progress_ticks + 1)
    if cap_progress_ticks == CAPTURE_TICKS:
        owner = X
        cap_progress_ticks = 0
        cap_team = None
        # Next tick falls into Case 3 (new owner scoring)

elif cap_team == opponent(X):
    # Stale progress from the other team — reset first, don't let X build progress through it
    cap_progress_ticks = 0
    cap_team = X
    # Next tick, X starts capturing from 0

else:  # cap_team == None
    cap_team = X
    cap_progress_ticks = 0
    # Next tick, X starts accumulating capture progress
```

Rationale for resetting stale opposing progress rather than decaying it first: it is simpler, it matches the user-visible rule "uncapped contested progress resets on empty" generalized to "resets on the other team taking over," and it avoids a long dead period where the point is held by nobody in the capture bar.

### Tick-order placement

The control-point state machine runs at tick-pipeline step 15 ("Update objective control"), after deaths (step 13) and respawn updates (step 14). This ensures a hero who dies on the point is correctly removed from the occupancy count before the state machine evaluates.

### Edge cases covered

- **Kill on point at the moment of capture**: the killed hero is removed from the occupancy count *before* the state machine runs, so a 1v1 trade on the point that leaves only one team standing is correctly credited.
- **Exact-tick capture completion**: the tick that brings `cap_progress_ticks` to `CAPTURE_TICKS` does not score. The *next* tick is Case 3 and begins scoring. One-tick latency on capture; negligible at 30 Hz but explicit.
- **Exact-tick score reaches WIN_TICKS**: win condition is checked inside Case 3 after the score increment. The round ends immediately at that tick.
- **Contested while owner near win**: Case 1 applies (contested). No scoring. The near-win does not decay; owner keeps their `team_score_ticks`. They just can't cross `WIN_TICKS` while contested.

### Timeout resolution

If neither team reaches `WIN_TICKS` before the round timer expires, the team with the higher `team_score_ticks` at the timeout tick wins. If `team_A_score_ticks == team_B_score_ticks` exactly, the round is a draw (terminal reward 0.0 for both teams).

## 4. Fog of war model

The fog model is the strongest single determinant of this project's research identity.

- **Vision is per-agent, not team-shared.** Each hero computes its own line-of-sight. Ranger and Mender may see different enemies on the same tick.
- **Allies are visible through walls** (role, position, HP, alive/dead, respawn timer). This is a quality-of-life concession matching hero-shooter UI conventions; it is not the research-interesting form of partial observation.
- **Enemies are visible only via direct line-of-sight** from that specific agent. No team-shared enemy vision.
- **Last-seen memory:** for ~1.5 seconds after an enemy leaves an agent's line-of-sight, that agent retains a decaying positional estimate of the enemy's last known location.
- **No sound-based perception** in MVP.
- **No learned agent-to-agent communication** in MVP.
- **No team-level enemy-reveal abilities in MVP.** The Phase 1 hero kits (Vanguard, Ranger, Mender) do not include any ability that shares enemy vision across allies. This is a deliberate Phase-1 simplification and makes the partial-observation problem strictly harder: the *only* cross-agent information channel is positioning (allies visible through walls). Reveal abilities can be reintroduced in later phases (e.g., a Cassidy-style Flashbang with a vision-reduction effect, or a Ranger "Mark Target" returning in Phase 10) and become a natural ablation target.
- **Muzzle traces are renderer-only in MVP.** The omniscient viewer/debug view may draw a short line where any weapon fires, but muzzle traces are **not** part of actor observations and do not reveal hidden enemies. A deliberate "hidden-fire perception" ablation (approximate direction / distance band, no exact position) may be added in a later phase.

### Why per-agent fog matters

Because vision is not team-shared and no hero can broadcast enemy information to allies, *positioning itself becomes the only communication channel*. If Ranger advances into a flank, Mender cannot know what Ranger sees — she can only infer from Ranger's movement pattern and her own observations. This turns coordination into a genuine partial-information problem and is the purest version of the research question this project is designed to study.

### Shields and vision

Shields block **hitscan only**. Shields do **not** block line-of-sight (they are transparent barriers). This is easier to reason about and reduces visibility edge cases.

## 5. Map

**Single hand-designed symmetric arena.**

```
Team A Spawn
     |
  Left flank   Main choke   Right flank
     \             |            /
      \            |           /
       \      Control Point   /
        \          |         /
  Right flank   Main choke   Left flank
     |             |            |
Team B Spawn
```

- Three meaningful routes: main choke, left flank, right flank
- Walls block movement, line-of-sight, and hitscan
- Low cover / pillars create peek angles
- Spawn rooms are safe zones
- No health packs in MVP (would complicate support behavior analysis)

### Per-episode map randomization — supported from day one, enabled after fixed-map learning

Wall and cover-pillar positions within the fixed topology can be perturbed each episode (±2 units on pillars, minor shuffles within cover type, subject to symmetry preservation). This prevents the policy from memorizing exact wall coordinates without the cost of hand-building multiple maps. Map seed / layout ID is an input to the critic.

**The sim supports this from the start, but it is not turned on during early training.** Per rl-design §6, training runs on a **fixed map** through Phase 7 and randomization is only enabled at Phase 8. Enabling randomization before the policy can solve a fixed map just slows learning without teaching anything useful.

## 6. Hero roster (Phase 1)

Three heroes total, one per role. No hero selection yet — team composition is fixed. Each hero is loosely modeled on a hero-shooter archetype: Vanguard ≈ Reinhardt, Ranger ≈ Cassidy, Mender ≈ Mercy. Names are generic to avoid IP issues; kits are "basically the same abilities" at archetype level with MVP simplifications.

### Vanguard — shield tank (Reinhardt archetype)

| Stat | Value |
|---|---|
| HP | 300 |
| Move speed | 3.6 u/s |
| Vision radius | 13 u |
| Hitbox | Large |

**Primary: Warhammer.** Short-range hitscan cone. Range 3.0 u, arc 45–60°, 2 strikes/sec, moderate damage. *Held* (strikes while `primary_fire` is held, gated by fire rate). **Suppressed while Barrier is active** — the Warhammer cannot strike while the shield is up. If `primary_fire` is held while Barrier is also held, the primary input is ignored.

**Ability 1: Barrier.** Forward-facing rectangular shield that blocks **damage rays** (enemy and allied damage hitscan). **Does not block healing rays** — Mender's heal beam passes through allied Barriers. Barrier HP 250, width 3.0 u, *held* (active while `ability_1` is held), −30% movement while active, 2-second regen delay after release, 3-second redeploy lockout after break.

*Barrier and Warhammer are mutually exclusive. This is a Reinhardt-style commitment: to deal damage, you must drop the shield and expose your team.*

**Ability 2: Guard Step.** Short forward dash, 2.0 u, 5-second cooldown. No damage, no knockback. *Edge-triggered* (one dash per `ability_2` rising edge).

*Deferred Rein-style abilities: Charge, Fire Strike, Earth Shatter (ult).*

### Ranger — revolver DPS (Cassidy archetype)

| Stat | Value |
|---|---|
| HP | 150 |
| Move speed | 4.2 u/s |
| Vision radius | 15 u |
| Magazine | 6 shots |

**Primary: Revolver.** Long-range hitscan, range 22 u, high per-shot damage, low spread, mild falloff. *Held* (fires while `primary_fire` is held, gated by fire rate). **6-shot magazine.** When empty, `primary_fire` is a no-op until reloaded.

**Reload behavior (explicit state machine):**

```
magazine              ∈ {0..6}
reload_state          ∈ {Ready, ReloadWindup}
ticks_since_last_shot : integer (reset on successful shot)
reload_ticks_left     : integer
```

Constants:

```
AUTO_RELOAD_DELAY_TICKS = 2.0 · TICK_HZ = 60   # inactivity before auto-reload begins
RELOAD_DURATION_TICKS   = 1.5 · TICK_HZ = 45   # time to fill the magazine
```

Transitions per sim tick:

```
On successful primary_fire shot (held, fire-rate gate open, magazine > 0):
    magazine -= 1
    ticks_since_last_shot = 0
    reload_state = Ready

If reload_state == Ready, magazine < 6,
   and ticks_since_last_shot >= AUTO_RELOAD_DELAY_TICKS:
    reload_state = ReloadWindup
    reload_ticks_left = RELOAD_DURATION_TICKS

If reload_state == ReloadWindup:
    primary_fire does nothing (no-op)
    reload_ticks_left -= 1
    if reload_ticks_left == 0:
        magazine = 6
        reload_state = Ready

On Combat Roll impulse (ability_1):
    magazine = 6
    reload_state = Ready
    reload_ticks_left = 0          # cancels any in-progress auto-reload

Else (ready, magazine > 0, not firing this tick):
    ticks_since_last_shot += 1
```

Clarifications:
- **Holding `primary_fire` with empty magazine counts as "not firing."** `ticks_since_last_shot` continues to advance; auto-reload begins after the usual delay.
- **Auto-reload reloads the full magazine**, not a partial amount. There is no per-bullet refill.
- **Combat Roll cancels an in-progress auto-reload** and instantly fills to 6.
- **Taking damage does not interrupt reload.** The only interrupters are Combat Roll (which completes it instantly) and death.

Current ammo count and `reload_state` are exposed in the agent's observation.

**Ability 1: Combat Roll.** Dashes 3.0 u in the current **movement-input direction**, or in the aim direction if no movement input is being given. Instantly refills the Revolver magazine to 6 shots. 5-second cooldown. *Edge-triggered* (one roll per `ability_1` rising edge).

**Ability 2:** *Deferred for MVP.* Cassidy-archetype candidates for later phases: Flashbang (short-range AoE with silence or vision-reduction), Magnetic Grenade (sticky damage), Deadeye (ult).

### Mender — heal-beam support (Mercy archetype)

| Stat | Value |
|---|---|
| HP | 150 |
| Move speed | 4.0 u/s |
| Vision radius | 14 u |
| Weapon state | STAFF or SIDEARM (toggled) |

Mender has two weapons, toggled via Weapon Swap. `primary_fire` fires whichever weapon is currently equipped.

**Primary (STAFF state): Caduceus Beam.** A locked-on healing beam. *Held.*

Beam targeting rules:
1. On rising edge of `primary_fire`, the sim finds the nearest ally within a **45° cone** and **12 u range** along Mender's aim direction, with line-of-sight.
2. If such an ally exists, the beam **locks** onto them. The beam persists while `primary_fire` is held.
3. While locked, the beam delivers **50 HP/sec** to the target ally. Overheal is discarded.
4. The beam **breaks** if: `primary_fire` is released, the target moves out of 12 u range, line-of-sight is broken, the target dies, or Mender weapon-swaps.
5. The beam **passes through allied Barriers** (so Mender can heal Vanguard through his own shield).
6. The beam has **no damage** component, even if aimed at an enemy. No damage boost in MVP ("for now" per scope).

**Primary (SIDEARM state): Sidearm.** Hitscan pistol. Range 15 u, low-medium per-shot damage, medium fire rate. *Held* (fires while `primary_fire` is held, gated by fire rate). Standard hitscan — single-target, first-hit wins, blocked by walls and enemy Barriers.

**Ability 1: Weapon Swap.** Toggles between STAFF and SIDEARM. *Edge-triggered.* 0.5-second swap cooldown to prevent bounce-toggling. Current weapon state is exposed in the observation.

**Ability 2: Tether.** Zips Mender to an aimed-at ally within 18 u and line-of-sight. 8-second cooldown. *Impulse.* If no valid ally is aimed at, the ability is wasted (observable through the cooldown-waste metric in rl-design.md §11).

**Sim semantics (instantaneous):** on a valid Tether activation, Mender's position snaps at the activation tick to the nearest legal point within 1.5 u of the target ally, along the line from Mender to the ally, not inside a wall. Collision is resolved immediately. Mender is not invulnerable or immaterial in flight — there is no flight.

**Viewer semantics (cosmetic only):** the renderer may interpolate Mender's sprite between old and new positions over 4 render frames for visual readability. The interpolation is purely visual and never affects sim state, LoS, hitscan, damage timing, or anything else the sim observes.

This split (instant sim, interpolated visual) avoids having to define mid-flight collision, vulnerability, LoS, and damage timing — and keeps the sim simple and deterministic.

*Deferred Mercy-style abilities: Damage Boost (alternate staff mode), Resurrect, Valkyrie (ult).*

### Rationale for Phase 1 composition

- Vanguard's shield/weapon mutual exclusion is the Reinhardt-style commitment that drives the tank's decision-making: you cannot both protect and attack.
- Ranger's ammo + Combat Roll reload creates a micro-rhythm of "shoot six, roll, reposition, shoot six" that is natural for human play and gives the policy a meaningful resource-management learning target.
- Mender's beam + pistol toggle forces a mode-switch decision: heal commitment is a weapon-swap away from self-defense. No damage boost in Phase 1 keeps the healer loop pure.
- No team-level enemy-reveal ability exists in this roster (see §4). This is a deliberate simplification for MVP and makes partial observation maximally pure.

## 7. Combat model

### Hitscan resolution

Each weapon fire creates a ray or cone. Per shot:

1. Check walls (ray terminates on first wall hit)
2. Check active shields (barrier absorbs, reduces barrier HP)
3. Check hero hitboxes
4. Apply damage or healing
5. Accumulate into per-tick damage buffer

**All damage and healing for a tick is applied simultaneously at tick end.** If two agents kill each other on the same tick, both die. Deterministic tie-break by entity ID.

### Shields

- Block **damage rays** (both enemy and allied damage hitscan)
- Do **not** block **healing rays** (Mender's Caduceus Beam passes through allied shields)
- Do not block movement
- Do not block line-of-sight
- Integer HP and integer-tick regen timer
- Deterministic HP decrement ordering by ray-hit time within a tick

Implementation note: when resolving a hitscan ray, each intersected shield is evaluated against the ray's effect type. A damage ray that hits a shield is absorbed (shield HP reduced by damage amount, ray terminates). A healing ray that hits a shield passes through unmodified.

## 8. Movement and physics

```
move_vector ∈ [-1, 1]²       (normalized if |v| > 1)
velocity = move_vector · hero_speed
position += velocity · dt
```

- Constant velocity — no acceleration in MVP
- **Hero vs wall:** solid collision
- **Hitscan vs wall:** blocked
- **Hero vs hero:** no collision (prevents stuck-agent bugs and door-clogging)
- **Shield vs hero:** no collision
- No knockback, stun, or pin mechanics

## 9. Aim model

Agents output an **angular delta** once per **policy decision** (not once per sim tick). The delta is capped at ±45° per decision.

### Applying the delta

On a decision tick, the policy's aim delta is applied to the current aim direction:
```
aim_angle[decision] = aim_angle[prev decision] + clip(delta, -π/4, π/4)
```

During the action-repeat window between decisions, the aim direction **does not change**. Aim is held. It only advances at the next policy decision.

### Turn-rate math

| Sim tick | Action repeat | Policy rate | Max delta | Max turn rate |
|---|---|---|---|---|
| 30 Hz | 3 | 10 Hz | ±45° | 450°/sec |
| 30 Hz | 2 | 15 Hz | ±45° | 675°/sec |

Both are well above typical human aim-tracking speeds. A 180° flick takes 4 decisions (400 ms at 10 Hz, 267 ms at 15 Hz).

### Observations

Absolute aim direction is simulator state and is exposed in each agent's observation. Agents need to know where they are currently aiming.

### Rationale

Angular delta avoids the wraparound discontinuity of absolute-angle parameterizations and the magnitude-ambiguity of 2D-vector parameterizations. It matches mouse/gamepad input behavior and produces smooth tracking of moving targets.

## 10. Determinism rules

### Fixed tick

| | |
|---|---|
| Simulation tick | 30 Hz |
| Render FPS | independent from simulation |
| RL action repeat | 2–3 ticks |
| Match duration | 5,400–7,200 ticks (180–240 s) |

### Deterministic state

- All randomness from a seeded PRNG
- No wall-clock time in simulation
- No unordered map/hashmap iteration affecting results
- Stable entity IDs
- Deterministic tie-break on simultaneous events (by entity ID)
- Cooldowns stored as integer ticks
- Floats internally; quantize at tick-end (position to 1/1000 unit, HP to 1/100)

Fixed-point math is deferred. Same-machine reproducibility is sufficient for MVP; cross-machine determinism is a later concern.

## 11. Tick pipeline

Never silently reordered. Explicit, stable.

```
for each tick:
    1.  Read actions for all living agents
    2.  Clamp and validate actions
    3.  Update aim directions (apply angular deltas)
    4.  Apply movement
    5.  Resolve wall collisions
    6.  Update cooldown timers
    7.  Activate/deactivate abilities
    8.  Rebuild spatial index
    9.  Compute per-agent visibility (independent line-of-sight per hero)
    10. Resolve weapon fire and ability effects
    11. Accumulate damage, healing, shield damage, status
    12. Apply accumulated effects simultaneously
    13. Process deaths
    14. Process respawn timers
    15. Update objective control
    16. Compute rewards
    17. Emit observations (actor-side partial, critic-side full)
    18. Log state/action/reward frame
```

## 12. Scope exclusions (MVP)

Explicitly out of scope in Phase 1:

- Hero ultimates
- Projectile travel time (hitscan only)
- Gravity, jumping, verticality
- Ammo economy *across the roster* — Ranger has a 6-shot revolver magazine with auto-reload + Combat Roll reload as a deliberate Cassidy-archetype mechanic; no other hero has ammo
- Hero-body blocking
- Knockback, stun, pin effects
- Complex status effects
- Destructible terrain
- Escort / payload mode
- Multiple hand-built maps (single topology with per-episode randomization instead)
- Hero selection, drafting, banning
- Items, leveling
- Sound-based perception
- Ping system, emotes, voice lines
- Agent-to-agent learned communication
- Health packs

None of these is required to prove the core concept. Deferred items may enter later phases.

## 13. Design risks and mitigations

| Risk | Mitigation |
|---|---|
| Agents ignore objective and deathmatch | Terminal reward dominates, short round timer, objective reward scaled appropriately |
| Shield tank causes stalemates | Shield slows tank, finite HP, regen delay, flank routes bypass |
| Support makes fights endless | Mender beam is single-target, LoS/range-limited, no overheal, no damage boost, breaks on weapon swap; focus fire must exceed beam HPS. Explicit eval metrics track beam uptime, effective / potential healing, overheal discarded, beam break reason, time-to-kill healed target, time-to-kill shielded + healed Vanguard, and fight duration after first pick. |
| Fog causes camping | Objective forces exposure; no team-reveal abilities in MVP means coordinated pressure must come from positioning, not scouting spells. **Muzzle traces are renderer-only** and do not reveal hidden enemies to actor observations (see §4 and rl-design.md §10). |
| RL agents learn reward hacks | Terminal win reward dominates, shaped rewards capped, evaluate with shaping disabled, watch replays |
| Single map overfitting | Implement map randomization early; train fixed-map first; enable randomization in Phase 8 |

## 14. Setting and aesthetic

**Not yet decided. Placeholder: generic sci-fi.**

Important downstream decision: committing to a setting and aesthetic lane early avoids "looks like Overwatch" framing at every subsequent art decision. Candidate directions:

- Blueprint / schematic minimalism
- Neon-on-dark tron-grid
- Biopunk wetware
- Post-collapse salvage crews
- Abandoned-megastructure AI factions
- Claymation / papercraft stop-motion

**Deadline:** before first human-facing renderer work.

## 15. Engine architecture

Five layers:

1. **Simulation core** — **C++** (C++20, CMake). Deterministic, headless, pure game state update. No rendering, no wall-clock, no I/O. Strict float-determinism discipline (see rl-design.md §9).
2. **Environment API** — Gymnasium-style wrapper exposed to Python via **`pybind11`**. Multi-agent observation/action dicts. Actor-side and critic-side observation builders are separate code paths.
3. **Bot layer** — scripted bots (C++, linked against the sim) for testing, early training opponents, and anchored evaluation baselines.
4. **Viewer / debugger** — 2D visualizer built on **raylib**, linked against the same C++ sim (no duplicate game logic). Must show true state + per-agent vision overlay + fog + raycasts + shields + cooldowns + last-seen ghosts + reward events. Human-play uses the same action schema the RL agents emit.
5. **Training / evaluation** — Python / PyTorch. Self-play, scripted opponents, snapshot league, metrics, replay inspection.

See `2026-04-20-rl-design.md` for details on layers 3, 5, and the contract of layer 2.
