# Xushi2 — Action Specification

Companion to `rl_design.md` §4 and `game_design.md` §6, §9. Defines the
exact action schema emitted by RL agents and the human viewer.

## Schema

One fixed-size `Action` struct per agent per policy decision.

| Field            | Type                | Range                | Notes |
|------------------|---------------------|----------------------|-------|
| `move_x`         | float32             | [-1, 1]              | tanh-squashed Gaussian from policy |
| `move_y`         | float32             | [-1, 1]              | tanh-squashed Gaussian |
| `aim_delta`      | float32             | [-π/4, π/4]          | applied once per policy decision (NOT per tick); ±45° cap |
| `primary_fire`   | bool (Bernoulli)    | {0, 1}               | held or edge-triggered depending on hero + ability |
| `ability_1`      | bool (Bernoulli)    | {0, 1}               | |
| `ability_2`      | bool (Bernoulli)    | {0, 1}               | |
| `target_slot`    | uint8 (Categorical) | {0..K-1}             | Phase 1: unused (always 0); enabled at Phase 10 |

C++ definition: `xushi2::common::Action` in
`src/common/include/xushi2/common/types.h`. Python mirror exposed via
`pybind11` as `xushi2.Action`.

## Per-decision vs per-tick

The action is emitted **once per policy decision**. Each decision lasts
`action_repeat` sim ticks. During the action-repeat window:

- `move_x`, `move_y` are applied every tick (continuous movement)
- `aim_delta` is applied **only once** at the start of the window.
  Aim does not advance during the window.
- `primary_fire` / `ability_1` / `ability_2` are re-evaluated every tick
  for held abilities; rising edges are detected only across window
  boundaries, not within.

| Action repeat | Policy rate | Max aim rate |
|---------------|-------------|--------------|
| 2 ticks @ 30 Hz | 15 Hz     | 675°/sec     |
| 3 ticks @ 30 Hz | 10 Hz     | 450°/sec     |

## Held vs edge-triggered (per hero)

See `rl_design.md` §4. Summary:

| Hero    | primary_fire          | ability_1              | ability_2       |
|---------|------------------------|-------------------------|------------------|
| Vanguard| Held (suppressed while Barrier is active) | Held (Barrier) | Edge (Guard Step) |
| Ranger  | Held (no-op if empty mag) | Edge (Combat Roll + reload) | *deferred* |
| Mender  | Held (current weapon)  | Edge (Weapon Swap)      | Edge (Tether)    |

The sim tracks previous-tick button state per agent to detect rising
edges. The previous-tick button state is **not** part of the policy
observation — the agent can spam a button without penalty; only 0→1
transitions trigger edge-triggered effects.

## Invalid actions

The sim treats invalid actions as no-ops, not errors. Invalid means:

- Firing while dead
- Firing an edge-triggered ability while it is on cooldown
- Firing Warhammer while Barrier is held (Vanguard mutual exclusion)
- Firing Revolver with an empty magazine and not currently reloading
- Using Mender Weapon Swap within the 0.5s swap cooldown
- Using Mender Tether with no valid ally in the aim cone

The cooldown state (and ammo, weapon state, Barrier state) are part of
the observation, so the policy can learn to avoid these waste cases.
Wasted actions are tracked via the `cooldown-waste` and
`fire-while-shielding` metrics in `rl_design.md` §11.

## Action distribution factorization

The policy's action distribution factorizes:

```
π(a | obs) =
      π_move(move_x, move_y)
    · π_aim(aim_delta)
    · π_primary(primary_fire)
    · π_ability_1(ability_1)
    · π_ability_2(ability_2)
  [ · π_target(target_slot) ]    # only when target_slot is enabled (Phase 10+)
```

Continuous heads use tanh-squashed Gaussian. Binary heads use Bernoulli.
Target-slot uses Categorical with masking over legal targets.

## Human-player action mapping (viewer)

The viewer translates keyboard + mouse into the same `Action` schema:

| Input                  | Action field           |
|------------------------|------------------------|
| WASD                   | `move_x`, `move_y`     |
| mouse movement (Δx)    | accumulated into `aim_delta` (clamped at decision time) |
| left mouse button      | `primary_fire`         |
| shift / space          | `ability_1`            |
| E / right mouse button | `ability_2`            |

Exact bindings are configurable in `viewer` but the emitted `Action`
struct is identical to what the RL agents emit. Human demonstrations can
be recorded as normal replays.
