# Exact sampled replay diagnosis — 2026-09-19

**Exact replay capture is implemented. V5 remains the reference. The next intervention should train combat while retaining objective occupancy, not assume isolated aim explains every failure.**

## What was established

- Evaluation source: `c386f8f7dad7036f8080b268da1702f8f1585bc8`; implementation in [PR #120](https://github.com/mitchelldurbincs/xushi2/pull/120), resolving [#118](https://github.com/mitchelldurbincs/xushi2/issues/118).
- Repeated all five first-seed reference cells: **480 games**, 96 per cell, base seed `0xA11CE`, eight envs, sampled learners and sampled snapshot opponents, 600-tick respawns, 15s unlock / 8s capture, 60s rounds.
- All five aggregate JSON files are **byte-identical** to the preceding reference report. These repeats add replay evidence, not independent statistical trials.
- Reconstructed all 480 initial states and **288,000 decision states** from recorded actions; every hash, final score and kill count matched.
- Full C++ and Python 3.10/3.11/3.12 CI passed on the implementation commit ([run](https://github.com/mitchelldurbincs/xushi2/actions/runs/35441928700)); 66 focused local Python tests and import boundaries passed.
- No training run or W&B run was launched. No new champion, phase clearance, or human viewer approval is claimed.

Exact commands, checkpoint hashes, source config paths, platform and artifact identity are in [manifest.json](manifest.json). Current behavior is measured on the same Linux/Torch/build setup as the preceding reference.

## What the failures show

Objective durations below are measured from native post-decision state, after the 15-second unlock. They approximate occupancy at the 0.1s decision interval. “Off point while alive” means at least one learner is alive and **no** learner occupies the point. All 480 episodes reached the point at least once; failing to arrive at all is not the issue in this sample.

| Policy / opponent / outcome | Games | Mean kills A/B | Seconds off point while alive | Seconds with learner on point | Seconds uncontested for learner |
|---|---:|---:|---:|---:|---:|
| v5 / weak / wins | 66 | 4.52 / 0.00 | 3.53 | 41.47 | 12.29 |
| v5 / weak / losses | 6 | 3.50 / 0.00 | 16.90 | 28.10 | 7.75 |
| v5 / weak / draws | 24 | 4.96 / 0.00 | 3.60 | 41.40 | 6.89 |
| L3 / weak / losses | 35 | 0.54 / 0.71 | 14.04 | 30.96 | 0.00 |
| L3 / weak / draws | 60 | 0.52 / 0.20 | 4.30 | 40.70 | 0.12 |
| L4 / v5 / losses | 75 | 0.12 / 1.04 | 39.04 | 5.96 | 0.70 |

**V5 has both an occupancy problem and a contest-clearing problem.** It never loses an agent against the weak bot in these 96 games, including all six losses. Those losses leave the point unoccupied much longer than wins. Its draws, however, occupy the point for most of the unlocked period and achieve almost five kills per game, but get only 6.89 seconds of uncontested occupation on average. Eliminations do not reliably translate into enough capture and scoring time before enemies return. This is evidence for the next diagnostic/training target, not proof of a single causal mechanism.

**L3's anchor failure is not simply forgetting how to approach the objective.** Its draws spend 40.70 of 45 unlocked seconds on point but almost never clear the contest. **L4 against sampled v5 has a different access/occupancy failure:** its losing games occupy the point for only 5.96 seconds, with very few kills. There is no evidence here that either successor has a combat advantage worth preferring over v5.

Capture progress freezes while contested and decays while the point is empty; it does not require an uninterrupted eight-second clear interval. The analysis's longest-clear interval is a diagnostic only. Hit-per-fire-command and pre-action nearest-target aim error from the existing combat analyzer are also diagnostics, not weapon shot accuracy or proof that aim is the sole bottleneck.

## Concrete episodes

Selection was fixed as the **first completed episode of each outcome in each cell**, giving 15 episodes. Their full timelines and existing combat-tool output are in [analysis.json](analysis.json); all 480 episode diagnostics are in [episode_diagnostics.json.gz](episode_diagnostics.json.gz).

1. `captures/v5-vs-weak-a11ce/cell-000/episode-0003.replay` — first v5 loss. V5 kills three enemies without a death, leads 1.20–0 at 45s, and has all three agents alive but none on point at 55s and 60s. The opponent finishes ahead, 1.97–1.20.
2. `captures/v5-vs-weak-a11ce/cell-000/episode-0002.replay` — first v5 draw. V5 finishes with five kills and no deaths, but zero score. At 30s both teams are off the point and the enemy team is dead; at 35s a respawned enemy is already contesting the returning learner. Occupancy and timing matter alongside damage.
3. `captures/l4-vs-weak-a11ce/cell-000/episode-0000.replay` — first L4 loss. L4 leads 5.17–0 with three kills and no deaths, then all its living agents leave the point. It loses 5.97–5.17.

These are programmatic trajectory inspections. Viewer judgment has not been supplied. To inspect the first example after extraction, build the existing viewer and run:

```bash
build/bin/xushi2_viewer --replay docs/reports/2026-09-19-sampled-replays/captures/v5-vs-weak-a11ce/cell-000/episode-0003.replay
```

## Replay archive and reproduction

All 480 original `.replay` files, per-decision hash sidecars and cell indices are in the separately delivered **`xushi2-sampled-replays-2026-09-19.tar.gz`** (29,357,879 bytes). Large raw traces are not committed to git. Archive SHA-256:

`cc844e12a0c79f17f6f6f8719e2972895efd9087ad0c9ec7e8b3502b5b851747`

Extract it into this report directory, then run from the repository root:

```bash
OMP_NUM_THREADS=1 python docs/reports/2026-09-19-sampled-replays/analyze.py
```

[analyze.py](analyze.py) reuses the existing replay parser, C++ simulator and combat analyzer. It verifies every decision, calculates objective diagnostics, and regenerates the two analysis JSON files. Bitwise replay is guaranteed only for the recorded same-machine/binary/compiler conditions; a different platform must verify rather than assume equality.

To produce new captures, add `--replay-dir NEW_DIRECTORY` to any reference matrix command. The directory must be new. Each counted game is written exactly once, with resolved configuration and both teams' actual C++ actions. Existing metric JSON and RNG streams are unchanged.

## Next experiment

The [combat-and-occupancy plan](../../plans/active/2026-09-19-combat-and-occupancy.md) specifies one current-entity-grid cap-duel port starting from v5, then one bounded pilot. It preserves the existing retention requirement (weak-bot mean score at least 2 on both reference seeds) and the sampled-v5 comparison. Occupancy and contest-clearing diagnostics determine whether an apparent isolated skill improvement transfers.

No training experiment has been started on the strength of this single-seed diagnosis. A candidate must be evaluated on both reference seeds, confirmed on fresh seeds, and reviewed in the viewer before any promotion decision.
