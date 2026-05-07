# Xushi2 — Observation Specification

Companion to `rl_design.md` §2–§3. Defines the exact layout of the
actor-side (per-agent, partial) and critic-side (centralized, full)
observation tensors for each curriculum phase.

**Coding philosophy:** Observation builders are Tier 0 code (see
`coding_philosophy.md`). Actor-observation leak checks are mandatory
in CI — any code path that iterates hidden enemies must not be
callable from `actor_obs_builder`.

**Status: Phase 4 critic obs live for 3v3 Ranger; Phase 1 actor obs
unchanged.** The actor builder remains at the Phase-1 layout (31 floats,
1v1-flavored — see Phase 1 table below). The critic builder has been
widened to the Phase-4 (3v3) layout: `build_critic_obs` (no phase suffix;
the previous `build_critic_obs_phase1` has been retired). The Phase-4
critic builder requires `MatchConfig::team_size == 3`; passing
`team_size == 1` will assert. Phase 1–3 trainers do not call the critic
builder at all (they use a shared-trunk actor-critic), so this swap is
not load-bearing for the single-agent phases. See
`docs/plans/2026-05-07-phase4-critic-obs-design.md` for the layout
rationale.

Totals: **actor = 31 floats** (Phase 1 table below),
**critic = 135 floats** (Phase 4 table further below). Dim constants are
mirrored across `python/xushi2/obs_manifest.py`,
`src/sim/include/xushi2/sim/obs.h`, and tested in lockstep by
`tests/observations/test_obs_dims.cpp` and
`python/tests/test_obs_manifest.py`.

## Observation invariants

Two invariants hold across every phase:

1. **Actor and critic observations have separate top-level builders and
   separate manifests.** They MAY share low-level pure utilities that
   cannot access hidden entities — team-frame coordinate transform,
   angle normalization, scalar normalization, enum encoding. The hard
   rule is: **no function that iterates over hidden enemies or full
   state may be called by `actor_obs_builder`.** This is the single
   most important rule in the project (see `rl_design.md` §10). Sharing
   a code path that touches hidden state is the most likely way to
   accidentally leak enemy information to the actor.
2. **All spatial features are in team-relative coordinates.** Team A sees
   the map from the A-side orientation; Team B sees it mirrored. The
   sim runs in world coordinates; the mirroring happens in the obs
   builder. See `rl_design.md` §6 "Team-relative coordinate normalization."

## Phase-by-phase layout

### Phase 1 (feedforward PPO, 1v1 Ranger, flat obs)

Minimal flat vector. One agent controls a single Ranger vs a scripted
opponent or a symmetric Ranger.

| Field                                        | Dim | Range / Encoding |
|----------------------------------------------|-----|------------------|
| own HP (normalized)                          | 1   | [0, 1] |
| own velocity                                 | 2   | [-1, 1] team-frame |
| own aim direction                            | 2   | unit vector `(sin θ, cos θ)` |
| own position                                 | 2   | team-frame, [-1, 1] normalized to map extent |
| own revolver ammo                            | 1   | [0, 1] = magazine / 6 |
| own reloading                                | 1   | {0, 1} |
| own combat-roll cd                           | 1   | [0, 1] = ticks_remaining / max_cd |
| enemy alive                                  | 1   | {0, 1} |
| enemy respawn timer                          | 1   | [0, 1] |
| enemy relative position                      | 2   | team-frame, delta of map-normalized positions; nominal range [-1, 1] in practice, can reach ±2 at opposite map edges |
| enemy HP (normalized)                        | 1   | [0, 1] |
| enemy velocity                               | 2   | team-frame, normalized by `ranger_max_speed()` (same convention as own velocity) |
| objective owner one-hot                      | 3   | {Neutral, Us, Them} |
| cap_team one-hot                             | 3   | {None, Us, Them} |
| cap progress                                 | 1   | [0, 1] |
| contested                                    | 1   | {0, 1} |
| objective unlocked                           | 1   | {0, 1} |
| own score                                    | 1   | [0, 1] |
| enemy score                                  | 1   | [0, 1] |
| self on point                                | 1   | {0, 1} |
| enemy on point (public / no-fog phase)       | 1   | {0, 1} |
| round timer                                  | 1   | [0, 1] = elapsed / total |

Total: 31 floats (see canonical totals at top). Feedforward MLP, no memory required.

Presence/alive fields:

- `enemy_alive = 0` zeros out enemy position / HP / velocity features.
- `enemy_respawn_timer` is `ticks_remaining / max_respawn_ticks`, or 0
  when alive.
- In no-fog phases, `enemy_alive` is ground truth. In fog phases (Phase
  7+), it is replaced with `(enemy_visible, enemy_last_seen_valid,
  enemy_alive_public_if_known)` — see Phase 7.

**Leak-prevention at Phase 1:** No fog of war yet, so no leak risk. But
the actor / critic builders must already be separate top-level
functions; this is the phase that establishes the contract.

**Phase 1 critic builder retired.** The 1v1 critic builder
(`build_critic_obs_phase1`, dim 45) has been removed. The current
`build_critic_obs` requires `MatchConfig::team_size == 3` and emits
the 135-float Phase-4 layout; phases 2 and 3 do not consume a
centralized critic at all (shared-trunk actor-critic), so retiring the
1v1 builder costs nothing operationally. Old training checkpoints
remain valid since they store actor weights only.

### Phase 2 (recurrent PPO on 1v1 memory toy)

Toy env spec lives elsewhere (`docs/memory_toy.md`, TBD). Purpose is to
validate RNN training, not to exercise the real obs pipeline.

### Phase 3 (recurrent PPO, 1v1 Ranger, flat obs)

Same flat obs as Phase 1. The only difference is the policy has a GRU.

### Phase 4 (recurrent MAPPO, 3v3 Ranger, flat obs)

Phase 4 is centralized-training / decentralized-execution (CTDE) MAPPO
with shared actor weights across the three Ranger slots per team. The
**actor obs** at Phase 4 is the existing 31-float Phase-1 layout, run
once per agent slot (no per-slot widening — all three Rangers are
role-identical and share the same actor network).

The **critic obs** is the only obs surface that grows. Layout (135 floats):

```
[0   ..  31)   actor_obs(team_perspective, own slot 0)   31 floats, team-frame
[31  ..  62)   actor_obs(team_perspective, own slot 1)   31 floats, team-frame
[62  ..  93)   actor_obs(team_perspective, own slot 2)   31 floats, team-frame
[93  .. 105)   enemy_world_block(enemy slot 0)           12 floats, world-frame
[105 .. 117)   enemy_world_block(enemy slot 1)           12 floats, world-frame
[117 .. 129)   enemy_world_block(enemy slot 2)           12 floats, world-frame
[129 .. 133)   cap_progress_ticks, team_a_score_ticks,
               team_b_score_ticks, tick_raw              4 floats, raw counters
[133 .. 135)   seed_hi, seed_lo                          2 floats, normalized
```

**Per-enemy world block (12 floats), world-frame, no team mirroring:**

| Field             | Dim | Notes                                              |
|-------------------|-----|----------------------------------------------------|
| world_position    | 2   | raw `(x, y)` from `HeroState::position`            |
| world_velocity    | 2   | raw `(vx, vy)` from `HeroState::velocity`          |
| world_aim_unit    | 2   | `(sin(aim_angle), cos(aim_angle))`, no mirror      |
| hp_normalized     | 1   | `health_centi_hp / max_health_centi_hp`            |
| alive_flag        | 1   | `{0, 1}`                                           |
| respawn_timer     | 1   | `(respawn_tick − now) / respawn_ticks`, clamped, 0 when alive |
| ammo              | 1   | `weapon.magazine / kRangerMaxMagazine`             |
| reloading         | 1   | `{0, 1}` from `weapon.reloading`                   |
| combat_roll_cd    | 1   | `cd_ability_1 / kRangerCombatRollCooldownTicks`    |

**Frame conventions:**

- The 3 own-team actor mirrors are **team-frame** (Team A as-is, Team B
  mirrored across map center) — exactly what each agent's actor would
  see. This makes the CTDE "critic sees ≥ actor" contract trivial to
  assert in tests.
- The 3 enemy world blocks are **world-frame, no mirror**, regardless of
  `team_perspective`. The critic must learn the team-frame ↔ world-frame
  bridge — same convention as Phase 1.

**Slot order:** within each team, the three Ranger slots are emitted in
ascending index order. Team A occupies slots 0–2, Team B slots 3–5; the
critic for `team_perspective == Team::A` sees own slots in the order
0,1,2 and enemy slots in order 3,4,5, and vice versa for Team B. The
sim does not permute slot identities, so this offset coupling is stable
in practice.

**Sim prerequisite:** the Phase-4 critic builder requires
`MatchConfig::team_size == 3`. The sim's spawn logic gates 3v3 spawning
on this field; the default (`team_size == 1`) preserves the 1v1
Phase-1/2/3 path bit-identically.

The canonical field list lives in `python/xushi2/obs_manifest.py`'s
`CRITIC_FIELDS` and is mirrored by the C++ implementation at
`src/sim/src/critic_obs.cpp`. See
`docs/plans/2026-05-07-phase4-critic-obs-design.md` for layout rationale
and trade-offs (in particular: why concat-actor-mirrors over a pure
team-level layout, and why per-enemy aim/ammo/cooldowns are privileged).

### Phase 5 (add entity attention)

Switch from fixed flat slots to variable-count entity tokens. See
`rl_design.md` §2 "Observation encoder."

### Phase 6 (add egocentric grid)

Add a 32×32 local spatial grid alongside the entity tokens.

### Phase 7 (partial observation)

Enable per-agent line-of-sight (game-design §4). From here, the actor's
visible-enemy slots only fill when that specific agent has LoS, and
last-seen decay begins.

Enemy presence becomes three separate fields:

- `enemy_visible` — this agent has current LoS
- `enemy_last_seen_valid` — last-seen ghost is still within the
  ~1.5-second decay window
- `enemy_alive_public_if_known` — alive/dead status derived from public
  kill feed only (never from hidden enemy state)

This is the phase where leak-prevention tests start earning their keep.
See `tests/observations/test_actor_leak.cpp`.

### Phase 8+ (map randomization, snapshot self-play, second heroes)

No observation-space changes from Phase 7 except:
- At Phase 10 (second heroes), enable the `target_slot` action and add
  entity-attention-based target selection for ally-targeted abilities.
  Also surface the `target_slot` valid-target mask:

  ```
  target_slot valid-target mask:
      fixed order over entity tokens
      invalid/dead/hidden targets masked
      no hidden enemy target slots exposed to actor
  ```

## Field canonical order

Observation tensors must be laid out in a fixed, documented field order.
A manifest file `python/xushi2/obs_manifest.py` will enumerate the order.
Changing the order is a breaking change — old checkpoints become invalid.

## Critic-side additions

Critic always sees (full target, Phase 4+ once the full roster is in play):
- True hidden-enemy positions (regardless of per-agent LoS)
- All cooldowns, ammo, weapon states, beam-lock targets
- Objective state machine internals (`cap_team`, `cap_progress_ticks`,
  `team_score_ticks` — see `game_design.md` §3)
- Map layout / seed
- Team side (A or B)

**Phase 4 subset (currently implemented):** the critic sees the
team-frame actor obs of all three own-team Ranger slots in ascending
slot order, then a 12-float world-frame block per enemy Ranger
(position, velocity, `(sin, cos)` of aim, hp, alive flag, respawn
timer, ammo, reloading flag, combat-roll cooldown), then raw objective
tick counters (`cap_progress_ticks`, `team_a_score_ticks`,
`team_b_score_ticks`, `tick_raw`), then `seed_hi`/`seed_lo`. Mender
weapon state / beam-lock target and Vanguard Barrier state enter when
those heroes land at Phase 10+. No explicit `team_side` scalar — team
perspective is implicit in the actor-mirror prefix.

Never exposed to actor:
- Hidden enemy positions (when outside this agent's LoS)
- Hidden enemy cooldowns, ammo, weapon state, internal ability state
- Enemy team's objective progress intent (only the displayed objective
  state is visible — progress bar, contested flag)
