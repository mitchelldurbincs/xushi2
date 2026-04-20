# Xushi2 — Observation Specification

Companion to `rl_design.md` §2–§3. Defines the exact layout of the
actor-side (per-agent, partial) and critic-side (centralized, full)
observation tensors for each curriculum phase.

**Status: Phase 0 — skeleton. Fill in as each curriculum phase gates.**

## Observation invariants

Two invariants hold across every phase:

1. **Actor and critic observations are built by separate functions.**
   Never share code between them. This is the single most important
   rule in the project (see `rl_design.md` §10). Sharing a code path is
   the most likely way to accidentally leak hidden enemy state to the
   actor.
2. **All spatial features are in team-relative coordinates.** Team A sees
   the map from the A-side orientation; Team B sees it mirrored. The
   sim runs in world coordinates; the mirroring happens in the obs
   builder. See `rl_design.md` §6 "Team-relative coordinate normalization."

## Phase-by-phase layout

### Phase 1 (feedforward PPO, 1v1 Ranger, flat obs)

Minimal flat vector. One agent controls a single Ranger vs a scripted
opponent or a symmetric Ranger.

| Field                   | Dim | Range / Encoding |
|-------------------------|-----|------------------|
| own HP (normalized)     | 1   | [0, 1] |
| own velocity            | 2   | [-1, 1] team-frame |
| own aim direction       | 2   | unit vector `(sin θ, cos θ)` |
| own position            | 2   | team-frame, [-1, 1] normalized to map extent |
| own revolver ammo       | 1   | [0, 1] = magazine / 6 |
| own reloading           | 1   | {0, 1} |
| own combat-roll cd      | 1   | [0, 1] = ticks_remaining / max_cd |
| enemy relative position | 2   | team-frame, [-1, 1] |
| enemy HP (normalized)   | 1   | [0, 1] |
| enemy velocity          | 2   | team-frame |
| objective progress      | 1   | signed, [-1, 1]: +1 = fully owned by us |
| round timer             | 1   | [0, 1] = elapsed / total |

Total: ~17 floats. Feedforward MLP, no memory required.

**Leak-prevention at Phase 1:** No fog of war yet, so no leak risk. But
the actor / critic code paths must already be separate; this is the
phase that establishes the contract.

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
| enemy presence flag             | 1            | × 1 |
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

This is the phase where leak-prevention tests start earning their keep.
See `tests/observations/test_actor_leak.cpp`.

### Phase 8+ (map randomization, snapshot self-play, second heroes)

No observation-space changes from Phase 7 except:
- At Phase 10 (second heroes), re-enable `target_slot` action and add
  entity-attention-based target selection for ally-targeted abilities.

## Field canonical order

Observation tensors must be laid out in a fixed, documented field order.
A manifest file `python/xushi2/obs_manifest.py` will enumerate the order.
Changing the order is a breaking change — old checkpoints become invalid.

## Critic-side additions (all phases)

Critic always sees:
- True hidden-enemy positions (regardless of per-agent LoS)
- All cooldowns, ammo, weapon states, beam-lock targets
- Objective state machine internals (`cap_team`, `cap_progress`)
- Map layout / seed
- Team side (A or B)

Never exposed to actor:
- Hidden enemy positions (when outside this agent's LoS)
- Hidden enemy cooldowns, ammo, weapon state, internal ability state
- Enemy team's objective progress intent (only the displayed objective
  state is visible — progress bar, contested flag)
