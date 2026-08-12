# Phase-0 baseline — native obs ownership + SimPool campaign (2026-08-11)

Baseline throughput before any change, per the approved plan
(`/Users/aspect/.claude/plans/1-make-the-c-shimmering-marble.md`). Every later
phase's bench must beat these numbers.

Config: `phase4_cap_headwidth_probe_v5.yaml` stripped for benching (wandb off,
no `init_from_checkpoint`, no BC pretrain), `vector_env: sync`,
`torch_num_threads: 1`. Bench: `python -m train.benchmark.run --target mappo
--warmup-iterations 1 --measured-iterations 5`. Machine: 18-core arm64 macOS,
CPython 3.14, CPU only. Git: 5a11b13 (+ dirty settings.local.json only).

| num_envs | env_steps/sec (rollout) | learner_steps/sec (update) | rollout s/iter | update s/iter |
|---|---|---|---|---|
| 16 | 2,210 | 3,841 | 0.93 | 0.53 |
| 64 | 2,264 | 4,306 | 3.62 | 1.90 |

env_sps flat from 16 → 64 envs: the sync vector env is a serial Python loop,
so width buys nothing today. Suite state at baseline: ctest 143/143,
pytest 606/606.
