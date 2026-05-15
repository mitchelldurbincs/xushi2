# Phase 4 Next Experiment Recommendation

Date: 2026-05-15

## Decision

Do not run `weak_basic_v2` as the next Phase 4 experiment.

Run an instrumentation-first diagnostic and use its result to scope the next
structural experiment. The implemented diagnostic is:

- `python/scripts/analyze_replay_combat.py`
- Output directory: `runs/phase4_replay_combat_diagnostics/`

This is the recommended next step because the last post-escalation result
falsified simple aim-retention as the missing ingredient. `aim_target_bc_v1`
preserved `90/96` synthetic aim hits after BC, but the full 3v3 policy still
settled into a scoreless `5/5` weak_basic draw. More opponent tuning would not
explain why preserved synthetic aim fails to convert into full-env damage,
focused kills, or score.

## Why Not `weak_basic_v2`

The proposed `weak_basic_v2` changes opponent aim noise, bot cooldown, bot
damage, round length, learning rate, entropy, and BC warm start. Those axes are
already heavily implicated by falsified Phase 4 variants:

- `weak_basic_v1` tested a weaker-aim opponent and still produced scoreless
  draws.
- Reduced bot fire rate and longer/shorter round variants did not escape the
  basin.
- Damage and opponent-strength variants have repeatedly changed how bad the
  outcome looks without producing a reliable learner scoring policy.
- LR and entropy changes have produced transient kill edges at best, not score
  conversion.

`weak_basic_v2` might create wins by making the scripted bot much less
dangerous, but it would still leave the core failure unmeasured: Team A fires
often yet does not convert fire into damage and score at the same rate as the
bot. Under the Escape Protocol, the next experiment should reduce uncertainty
about the failure mode rather than spend another run on a multi-axis nerf.

## Diagnostic Result

The replay analyzer reconstructs each text replay through the C++ sim, detects
episode resets by tick rollback, and aggregates per-slot fire commands,
visible-fire commands, damage-producing hit deltas, kill deltas, damage, nearest
visible target aim error, and nearest-target distribution.

Recent Phase 4 stochastic replay results:

| Replay | A hit/fire | B hit/fire | A kills | B kills | A aim error | B aim error |
|---|---:|---:|---:|---:|---:|---:|
| `aim_freeze_bc_v1_ckpt0050` | `0.0144` | `0.0231` | `0` | `9` | `1.684` | `1.470` |
| `aim_target_bc_v1_ckpt0050` | `0.0219` | `0.0455` | `2` | `30` | `1.466` | `1.074` |
| `aim_transfer_v1_ckpt0500` | `0.0102` | `0.0336` | `0` | `14` | `1.554` | `1.039` |
| `aux_aim_v1_ckpt0500` | `0.0100` | `0.0373` | `0` | `20` | `1.559` | `1.016` |
| `invalid_fire_mask_v1_ckpt0500` | `0.0100` | `0.0448` | `0` | `24` | `1.548` | `0.974` |
| `per_action_entropy_v1_ckpt0500` | `0.0160` | `0.0368` | `0` | `19` | `1.556` | `1.190` |
| `weak_basic_v1_ckpt0500` | `0.0096` | `0.0432` | `0` | `23` | `1.550` | `0.807` |

Behavioral autopsy:

- Team A usually fires almost continuously, so this is not primarily a
  no-fire failure.
- Team A usually fires while a target is visible, so this is not primarily a
  no-visibility failure.
- Team A's damage-producing hit conversion is consistently much worse than the
  bot's, even in the stronger `aim_target_bc_v1` probe.
- Team A's nearest-visible-target aim error is consistently high, usually
  around `1.5` radians.
- Target attribution is diffuse and does not show a reliable focus-fire policy.
  For example, `aim_target_bc_v1` Team A attributed nearest targets across
  slots `{3: 440, 4: 2036, 5: 1235}` while the bot still converted roughly twice
  the hit rate.

This points to a full-env combat composition failure: the policy can emit fire
commands and can preserve synthetic aim in isolation, but it does not produce
accurate, focused, timed shots in the moving 3v3 environment.

## Proposed Next Experiment

Next run should be a structural combat-composition probe, not another
opponent-strength variant:

1. Add the replay-combat metrics to the eval path for the next probe so every
   checkpoint reports Team A/B hit conversion, visible-fire rate, aim error, and
   target concentration alongside win/loss/draw.
2. Implement a target-conditioned combat head or target-selection head that
   explicitly predicts the enemy slot to focus, then conditions aim/fire on that
   target while keeping the existing movement/objective pathway intact.
3. Gate the run before PPO with a full-env BC diagnostic: require preserved
   synthetic aim and improved replay-combat hit conversion against weak_basic.
4. Stop at update 50 unless hit/fire, kill concentration, or score improves
   beyond the current `aim_target_bc_v1` baseline.

Falsification criteria for the next structural probe:

- If Team A hit/fire remains below `0.025` and aim error remains near `1.5`
  radians after BC or update 50, target conditioning is not solving the combat
  composition failure.
- If hit/fire improves but target distribution remains diffuse and score stays
  zero, prioritize explicit focus-fire reward/instruction or a simplified 1v1
  combat environment.
- If hit/fire and focus improve but score remains zero, the next bottleneck is
  objective timing or survival, not combat mechanics.

## Artifact Notes

The analyzer does not change sim rules, training, reward, obs/action spaces, or
the replay format. It reconstructs existing `.replay` files and should be safe
to run on old artifacts. Final match state is reported for the last detected
episode; aggregates span all detected episodes.
