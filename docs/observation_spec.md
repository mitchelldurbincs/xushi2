# Xushi2 — Observation Specification

Companion to `rl_design.md` §2–§3. Defines the exact layout of the
actor-side (per-agent, partial) and critic-side (centralized, full)
observation tensors for each curriculum phase.

**Coding philosophy:** Observation builders are Tier 0 code (see
`coding_philosophy.md`). Actor-observation leak checks are mandatory
in CI — any code path that iterates hidden enemies must not be
callable from `actor_obs_builder`.

**Status: Phase 1b — actor + critic builders live for 1v1 Ranger.** The
Phase 1 layouts below are canonical and implemented
(`src/sim/src/actor_obs.cpp`, `src/sim/src/critic_obs.cpp`). Dim constants
are mirrored by `python/xushi2/obs_manifest.py` and
`src/sim/include/xushi2/sim/obs.h`. Phase 2+ layouts remain spec-only.

Totals at Phase 1: **actor = 31 floats**, **critic = 45 floats**.
The "~28" mentioned in the Phase 1 table below was an early estimate;
the canonical totals above are computed from the field list and kept in
lockstep by tests (`tests/observations/test_obs_dims.cpp` and
`python/tests/test_obs_manifest.py`).

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

### Phase 2 (recurrent PPO on 1v1 memory toy)

Toy env spec lives elsewhere (`docs/memory_toy.md`, TBD). Purpose is to
validate RNN training, not to exercise the real obs pipeline.

### Phase 3 (recurrent PPO, 1v1 Ranger, flat obs)

Same flat obs as Phase 1. The only difference is the policy has a GRU.

### Phase 4 (recurrent IPPO/MAPPO, 2v2, flat obs)

Expand flat obs to cover 2 allies and up-to-2 enemies. Fixed-size slots
(no variable-length — that's Phase 5). Missing allies/enemies are zero-
padded with a presence flag.

| Added fields                    | Dim per slot | Count |
|---------------------------------|--------------|-------|
| ally presence flag              | 1            | × 1 (one teammate) |
| ally HP, position, velocity, aim| ≈ 7          | × 1 |
| enemy presence/alive flag       | 1            | × 1 |
| enemy HP, pos, vel, aim         | ≈ 7          | × 1 |

Critic obs at Phase 4: flat vector containing the full sim state
(positions, HP, cooldowns, ability states, RNG-independent counters).
Critic-side fields MUST include ground-truth enemy positions regardless
of visibility.

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

**Phase 3 subset (currently implemented):** the critic sees the flat actor
prefix for team-perspective, plus world-frame own/enemy position and
velocity, `cap_progress_ticks`, `team_a_score_ticks`,
`team_b_score_ticks`, the raw tick counter, and `seed_hi`/`seed_lo`.
Per-hero cooldown / ammo / weapon-state / beam-lock-target fields enter
with the corresponding heroes as the roster grows in Phase 4+ (Mender
weapon state and beam-lock target; Vanguard Barrier state). No explicit
`team_side` scalar today — team perspective is implicit in which slot's
actor prefix the critic consumes (see code-side follow-up).

Never exposed to actor:
- Hidden enemy positions (when outside this agent's LoS)
- Hidden enemy cooldowns, ammo, weapon state, internal ability state
- Enemy team's objective progress intent (only the displayed objective
  state is visible — progress bar, contested flag)
