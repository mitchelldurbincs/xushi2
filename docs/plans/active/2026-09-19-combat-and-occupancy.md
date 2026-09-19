# Combat while holding the objective

Date: 2026-09-19. Status: next implementation/experiment assignment; not launched.

## Basis

[Exact sampled replay diagnosis](../../reports/2026-09-19-sampled-replays/README.md)
verified 480 episodes and 288,000 decision hashes. V5 is still the reference.
Its six weak-bot losses have zero deaths and much more time off point than its
wins. Its draws mostly occupy the point but do not convert enough contested
time into scoring. L3/L4 do not retain both objective and combat performance.
The prior plan to start with isolated aim is narrowed to a cap-duel task that
also requires objective occupation. A fixed network-capacity ceiling remains
unproven.

## One implementation

Port the existing cap-duel mini-task to the current native entity-grid actor
observations. Keep the existing mini-task and runtime factory as the entry
points; do not add a separate trainer. Use the same ObservationEngine semantics
and tensor layout as v5. Preserve the current action and checkpoint shapes,
including inactive-slot handling. Load v5 without replacing or widening learned
layers. Do not revive missing flat-observation teacher artifacts.

The point is to learn to clear a contest and stay/return on objective while
fighting. This is not permission to rewrite the full-game reward or add target
lead observations. Any necessary mini-task config changes must be explicit in
the implementation PR; full-game evaluation stays at the pinned settings.

Preflight must prove:
- checkpoint loading preserves every v5 parameter before updates;
- mini-task actor observations use the same native visible-entity path and
  expected masks, including inactive enemies;
- action/team frames and recurrent resets remain correct;
- observation/leak suites and replay reconstruction pass;
- a zero-update full-game evaluation still matches the v5 reference.

## Bounded pilot after preflight

Use one training seed (`0xC0FFEE`) and one candidate, starting from v5. Cap the
pilot at **100 updates**, evaluating updates 0, 25, 50, 75 and 100. Determine and
record rollout width/horizon and therefore the absolute interaction budget in
the reviewed config before launch. Reuse existing anchor/retention machinery;
keep intervention knobs fixed for the whole pilot. Log training in W&B with
commit, config, seed, URL, checkpoints and exact evaluation replays.

Every checkpoint uses the committed matrix evaluator: 96 games per cell,
eight envs, sampled actions, 600t/15s/8s, relative snapshot paths unchanged,
both `0xA11CE` and `0xBEEF`. Evaluate vs weak_basic_v2 and sampled frozen v5.
No promotion based on the mini-task reward or a best small-sample checkpoint.

Stop at the budget. Stop earlier if two consecutive checkpoints score below
2 against weak_basic_v2 on either reference seed. If mini-task combat improves
but objective retention or full-game occupancy falls, record a negative
transfer result and inspect it before extending training.

A candidate advances to confirmation only if it preserves mean weak-bot score
>=2 on both reference seeds and wins more games than it loses against sampled
v5 on both. Track alive-but-off-point duration, uncontested occupation,
score, kills, and sampled failure replays so improvement has an interpretable
mechanism. Confirm a selected candidate without further tuning on fresh seeds
`0xD00D` and `0xFACE` (96 games per cell each).

This is an experiment acceptance rule, not a phase-clearance rule. Promotion
still needs human viewer inspection under AGENTS.md. This task has not launched the pilot or advanced a phase.
