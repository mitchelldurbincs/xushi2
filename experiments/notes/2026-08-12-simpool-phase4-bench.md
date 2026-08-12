# SimPool Phase-4 bench — env boundary fixed; training is now model-bound (2026-08-12)

Follow-up to `2026-08-11-native-obs-simpool-baseline.md`. Same machine, same
stripped probe-v5 config; pool runs add `env.native_entity_obs: true` +
`ppo.vector_env: sim_pool`. Suites at time of measurement: ctest 163/163,
pytest 623/623 (includes pool-vs-legacy full-episode parity across
auto-resets: obs/critic byte-identical, rewards ≤ 1e-6).

## Full training loop (train.benchmark.run, warmup 1, measured 5)

| config | env_sps (rollout) | learner_sps | rollout s/iter | update s/iter |
|---|---|---|---|---|
| 16 envs, sync | 2,210 | 3,903 | 0.93 | 0.52 |
| 16 envs, sim_pool | 2,202 | 3,872 | 0.93 | 0.53 |
| 64 envs, sync | 2,259 | 4,212 | 3.63 | 1.94 |
| 64 envs, sim_pool | 2,234 | 4,275 | 3.67 | 1.92 |
| 64 envs, sim_pool, torch_num_threads 8 | 2,053 | 3,104 | 3.99 | 2.64 |

## Pure env stepping (no model), 64 envs × 200 vector steps

| backend | ms / vector step | env-steps / s |
|---|---|---|
| sync (legacy per-env, native obs) | 9.73 | 6,581 |
| sim_pool | 1.71 | **37,384** |

## Reading

- The batched boundary did what it was designed to do: **5.7× on env
  stepping**, landing inside the design doc's "10–50k sim steps/sec"
  target (rl_design.md §9). One GIL-released FFI call per vector step,
  persistent bots, reward features in one block.
- End-to-end training throughput did NOT move, because at this model size
  the rollout was already ~90% torch: `sample_action` for 64×3 agents per
  step (GRU + entity attention + 3×32×32 grid conv) plus rollout
  bookkeeping is ~27 ms of the ~28.5 ms vector step. The env was ~9.7 ms
  of it and is now ~1.7 ms.
- `torch_num_threads: 8` makes it *worse* (small tensors; intra-op
  threading overhead), so the model cost is many small sequential
  forwards, not a parallelizable matmul bound.
- Consequence for the plan: Phase-5 SimPool threading will NOT improve
  training throughput at current model size — the lever moved to the
  torch side (bigger per-forward batches, torch.compile, or accelerator
  offload). Note the capacity campaign wants *wider* models, which makes
  the model share grow further.
- Consequence that still stands: eval and any env-heavy tooling (replay
  sweeps, matrix evals with cheap policies, BC data generation) get the
  full 5.7×, and the 128-env width is now practically free on the env
  side.
