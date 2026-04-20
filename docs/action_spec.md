# Xushi2 — Action Specification

Companion to `rl_design.md` §4 and `game_design.md` §6, §9. Defines the
exact action schema emitted by RL agents and the human viewer.

**Coding philosophy:** Action canonicalization is a Tier 0 boundary
(see `coding_philosophy.md`). All action values are clamped, quantized,
and validated exactly once at this boundary before entering the sim.
The canonical stored form and the live-sim form are bit-identical.

## Schema

One fixed-size `Action` struct per agent per policy decision.

| Field            | Type                | Range                | Notes |
|------------------|---------------------|----------------------|-------|
| `move_x`         | float32             | [-1, 1]              | tanh-squashed Gaussian from policy |
| `move_y`         | float32             | [-1, 1]              | tanh-squashed Gaussian |
| `aim_delta`      | float32             | [-π/4, π/4]          | applied once per policy decision (NOT per tick); ±45° cap |
| `primary_fire`   | bool (Bernoulli)    | {0, 1}               | held-semantics (see below) |
| `ability_1`      | bool (Bernoulli)    | {0, 1}               | held or impulse depending on hero + ability |
| `ability_2`      | bool (Bernoulli)    | {0, 1}               | held or impulse depending on hero + ability |
| `target_slot`    | uint8 (Categorical) | {0..K-1}             | Phase 1–9: omitted or always 0. Phase 10+: enabled for explicit ally/enemy targeted abilities. |

C++ definition: `xushi2::common::Action` in
`src/common/include/xushi2/common/types.h`. Python mirror exposed via
`pybind11` as `xushi2.Action`.

All actions are canonicalized before entering the sim: continuous fields
are quantized to the replay representation (`int16`, scale `1/10000`),
booleans are packed, `target_slot` is clamped. Training, human play, and
replay all consume the same canonicalized `Action`. See
`replay_format.md` for the canonical byte layout.

## Per-decision vs per-tick

The action is emitted **once per policy decision**. Each decision lasts
`action_repeat` sim ticks. During the action-repeat window:

- `move_x`, `move_y` are applied every tick (continuous movement)
- `aim_delta` is applied **only once** at the start of the window.
  Aim does not advance during the window.
- Held-semantics booleans (`primary_fire`, Vanguard `ability_1`) are
  re-evaluated every tick of the window against the same decision
  action — they behave exactly as if the button were held for the full
  window.
- Impulse-semantics booleans (see below) are evaluated **once**, at the
  start of the decision window.

| Action repeat | Policy rate | Max aim rate |
|---------------|-------------|--------------|
| 2 ticks @ 30 Hz | 15 Hz     | 675°/sec     |
| 3 ticks @ 30 Hz | 10 Hz     | 450°/sec     |

## Held vs impulse semantics (per hero)

To keep the action interface Markovian from the policy's perspective,
edge-triggered abilities are exposed as **impulses**, not as physical
buttons. The sim does **not** track previous-tick button state for
impulse fields.

- **Held** fields: value `= 1` means "hold the button for this decision
  window." Re-evaluated every sim tick in the window.
- **Impulse** fields: value `= 1` means "attempt the ability once at the
  start of this decision window." Value `= 0` means "do not attempt."
  No previous-button state is required; identical observations plus
  identical actions produce identical outcomes.

| Hero    | primary_fire          | ability_1                | ability_2           |
|---------|------------------------|---------------------------|----------------------|
| Vanguard| Held (suppressed while Barrier is active) | Held (Barrier) | Impulse (Guard Step) |
| Ranger  | Held (no-op if empty mag) | Impulse (Combat Roll + reload) | *deferred* |
| Mender  | Held (current weapon)  | Impulse (Weapon Swap)     | Impulse (Tether)     |

The viewer converts real human key/mouse rising edges into single
one-decision impulses, so human play and RL play emit identical
`Action` structs.

**Why impulse, not held-with-edge-detect:** if the sim tracked previous
button state, then the same observation plus the same action could
produce different outcomes (Combat Roll triggers if the previous
decision's `ability_1` was 0; no-ops if it was 1). That hidden
dependency makes the environment non-Markovian for the policy — a real
hazard for feedforward PPO in Phase 1. Impulse semantics remove hidden
action-interface state from the sim.

## Invalid actions

The sim treats invalid actions as no-ops, not errors. Invalid means:

- Firing while dead
- Firing an impulse ability while it is on cooldown
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
| left mouse button      | `primary_fire` (held while pressed) |
| shift / space          | `ability_1` (impulse: rising edge → 1 on the next decision) |
| E / right mouse button | `ability_2` (impulse: rising edge → 1 on the next decision) |

For impulse inputs, the viewer sets the field to 1 on the decision
immediately following a rising edge and back to 0 on all subsequent
decisions until the next rising edge. Held inputs reflect the current
physical button state at decision time.

Exact bindings are configurable in `viewer` but the emitted `Action`
struct is identical to what the RL agents emit. Human demonstrations can
be recorded as normal replays.
