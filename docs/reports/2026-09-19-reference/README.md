# Reference evaluation — 2026-09-19

The pending architecture branch was merged in [PR #117](https://github.com/mitchelldurbincs/xushi2/pull/117). This report establishes a fresh Linux reference for the three preserved policies after the native-observation cutover. **v5-0300 remains the reference policy. Neither L3-0800 nor L4-0300 is a replacement champion.**

## Results

Each cell below combines two independent evaluation seeds, 96 episodes each. Counts are wins / losses / draws from the learner (Team A) perspective. Mean score is averaged over all 192 episodes. All policies sample actions, including the frozen v5 opponent.

| Learner | Opponent | W / L / D | Mean score A / B |
|---|---|---:|---:|
| v5-0300 | weak_basic_v2 | 126 / 14 / 52 | 2.358 / 0.596 |
| L3-0800 | weak_basic_v2 | 1 / 66 / 125 | 0.001 / 1.900 |
| L4-0300 | weak_basic_v2 | 50 / 81 / 61 | 1.872 / 4.064 |
| L3-0800 | v5-0300 | 1 / 100 / 91 | 0.027 / 3.718 |
| L4-0300 | v5-0300 | 1 / 141 / 50 | 0.001 / 6.069 |

### Per-seed results

| Learner | Opponent | Base seed | W / L / D | Score A / B |
|---|---|---|---:|---:|
| L3-0800 | v5-0300 | 0xA11CE | 1 / 45 / 50 | 0.055 / 2.909 |
| L3-0800 | v5-0300 | 0xBEEF | 0 / 55 / 41 | 0.000 / 4.526 |
| L3-0800 | weak_basic_v2 | 0xA11CE | 1 / 35 / 60 | 0.001 / 2.198 |
| L3-0800 | weak_basic_v2 | 0xBEEF | 0 / 31 / 65 | 0.000 / 1.602 |
| L4-0300 | v5-0300 | 0xA11CE | 1 / 75 / 20 | 0.001 / 6.630 |
| L4-0300 | v5-0300 | 0xBEEF | 0 / 66 / 30 | 0.000 / 5.508 |
| L4-0300 | weak_basic_v2 | 0xA11CE | 22 / 39 / 35 | 1.694 / 3.926 |
| L4-0300 | weak_basic_v2 | 0xBEEF | 28 / 42 / 26 | 2.049 / 4.201 |
| v5-0300 | weak_basic_v2 | 0xA11CE | 66 / 6 / 24 | 2.373 / 0.530 |
| v5-0300 | weak_basic_v2 | 0xBEEF | 60 / 8 / 28 | 2.344 / 0.662 |

## Interpretation

- v5 retained objective conversion on both seeds: score 2.373 and 2.344 against weak_basic_v2, above the existing retention threshold of 2. Its pooled win rate was 65.6% (126/192).
- L3 has effectively lost its anchor objective game. L4 retains some objective behavior, but its scores were 1.694 and 2.049; it did not meet the threshold on both seeds and lost more anchor games than it won.
- Both successors won only 1/192 games against sampled v5. The preserved filename `best_fighter` is a historical label, not the current ranking. Neither candidate meets the dual requirement of retention plus a win-count edge over v5.
- This establishes reference measurements, not a new phase clearance. It does not prove a fixed network-capacity ceiling, generalization to other maps/teams/opponents, or reliable performance at 240-tick respawns.

## Evaluation contract

- Integration commit: `3da4c143542f7d99fbfcaa4146887a187670f2ea`.
- Evaluation source: `4e18bb745d797b7859a3944a946a0f4e5be77885`. Its only executable change from the merge is an optional matrix-CLI batch-width argument; game, model, reward, observation, and evaluation-loop implementations are unchanged.
- All three original checkpoint files are unchanged. Their SHA-256 hashes, source training-config paths, embedded-setting contract, exact commands, outputs, and output hashes are in [manifest.json](manifest.json).
- 60-second fixed-map rounds; fog disabled; action repeat 3; unlock 15 seconds; capture 8 seconds; respawn **600 ticks**. The CLI uses `--as-trained`; its default canonical mode would use 240 ticks.
- Eight evaluation environments, one Torch thread, CPU inference; Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.5.3, Gymnasium 1.3.0, GCC 13.3.0, Linux x86-64. Full platform and extension hash are recorded.
- Base seeds `0xA11CE` and `0xBEEF`. Each cell is a separate CLI call. The existing CLI adds 1000 to the learner sampling seed for snapshot matchups; the manifest records this effective seed.
- Run from the repository root and preserve the exact relative opponent checkpoint path. Snapshot sampling currently incorporates the path string in its RNG seed; renaming the file changes the sample.
- The 960 reference episodes exclude a further 96-episode repeat of v5-vs-weak at `0xA11CE`; that repeat produced byte-identical JSON. It verifies reproducibility on this build and is not additional independent evidence.
- These are offline checkpoint evaluations. No W&B training run was created.

Example (repeat with the second seed; all ten commands are in the manifest):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python python/scripts/eval_mappo_matrix.py \
  --checkpoint data/checkpoints/phase4_v5_upd300_stochastic_600t_converter.pt \
  --anchor-bot weak_basic_v2 --episodes 96 --num-envs 8 \
  --seed 0xA11CE --as-trained --stochastic --output /tmp/v5-reference.json
```

## Historical comparability

The August 9 bot reference was 77/3/16, score 2.572. The current first-seed result is 66/6/24, score 2.373. Earlier artifacts did not fully record hardware, Torch version, or evaluation batch width, and snapshot RNG depends on its path. Cross-platform bitwise equality is not promised by this simulator. These numbers are a new pinned reference; their difference alone cannot identify a code regression.

Snapshot observations also intentionally changed on August 12 to each checkpoint's training-time semantics. Pre-cutover head-to-head standings must not be carried forward. The current runs show v5 decisively ahead of these two successors under the recorded setup.

## Replays and validation

Three separate **greedy** diagnostic episodes against weak_basic_v2 are preserved in [diagnostic_replays](diagnostic_replays/), with hashes and headers in its index. Each has 600 decisions and the expected 600-tick respawn setting. Decompress a `.replay.gz` file before opening it with the existing viewer. They are illustrative episodes, not the sampled games in this matrix; no human replay review or subjective behavior approval is claimed.

Exact per-episode matrix replay export is explicitly deferred to [issue #118](https://github.com/mitchelldurbincs/xushi2/issues/118). This is the remaining gap against the training checklist's replay requirement.

Validation before integration: all four GitHub CI jobs passed (C++ and Python 3.10–3.12); local 163/163 C++ tests, 29 focused Python tests, import-boundary check and training smoke passed. The evaluation CLI change passed eight matrix tests, including identical bot and snapshot results when host CPU count is changed from 1 to 32 with width pinned.

## Next experiment

1. Keep v5-0300 and this evaluation contract as the frozen reference. Start the next candidate from v5, preserving its learned objective behavior.
2. Complete #118 and inspect actual sampled failures before selecting the combat intervention: distinguish poor approach, weak aim, and failure to occupy the objective.
3. Port one existing aim/combat mini-task to the current entity-grid observation format. Preserve the current checkpoint lineage; use isolated skill pretraining plus the existing retention/anchor machinery. The historical flat-observation teacher artifacts are unavailable and incompatible with the current lineage.
4. Run one bounded pilot, with the intervention and stop rule fixed in advance. Success requires improved combat and objective retention together: mean score at least 2 vs weak_basic_v2 on both reference seeds, and a win-count edge over sampled v5 in the committed evaluation path. Confirm a selected candidate on separate held-out seeds before promotion.
5. Stop the pilot if isolated combat improves while full-game retention fails; inspect transfer rather than extending the same league run. Wider heads remain an unanswered hypothesis because the earlier migration probes erased skills before testing capacity.

This report does not launch the next training campaign.
