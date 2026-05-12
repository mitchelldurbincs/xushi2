# C++ Benchmarking Guide

This guide defines the minimum standard for simulator/runtime benchmark runs.

## Build requirements

- Benchmarks **must** be compiled in `Release` mode.
- Configure with benchmark targets enabled:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DXUSHI2_BUILD_BENCHMARKS=ON
cmake --build build -j --target benchmarks
```

## Run guidance

- Use repeated runs to reduce noise (`--benchmark_repetitions=5` baseline).
- Set a minimum timing window (`--benchmark_min_time=0.1` baseline).
- Prefer JSON output for machine-readable trend comparison.

Example:

```bash
./build/benchmarks/<benchmark_binary> \
  --benchmark_repetitions=5 \
  --benchmark_min_time=0.1 \
  --benchmark_out=build/benchmarks/results/<benchmark_binary>.json \
  --benchmark_out_format=json
```

## Baseline artifacts

- Store baseline benchmark results under:
  - `build/benchmarks/results/*.json`
- For local smoke checks, use a separate artifact suffix such as:
  - `build/benchmarks/results/*.smoke.json`

## Determinism invariant

Benchmark code is performance instrumentation only.

- Benchmark harnesses and fixtures **must preserve deterministic simulation invariants**.
- Benchmark setup must not mutate production sim behavior in ways that alter deterministic outcomes.
- Any optimization proposed from benchmark findings must still pass determinism and test gates before merge.
