# Benchmark Baselines and Drift Interpretation

This folder defines a lightweight, contributor-friendly benchmark discipline that
is robust to noisy environments.

## Normalization guidance

When collecting benchmark results used for baseline or comparison:

1. **Use Release builds only.**
   - C++: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
2. **Pin CPU governor to a fixed performance profile where possible.**
   - Linux example (requires privileges on host):
     - `cpupower frequency-set -g performance`
   - If not possible (managed CI, container limitations), record the governor and
     only compare results against runs from the same environment class.
3. **Run N repetitions and report median.**
   - Recommended `N=7` (minimum 5) for quick checks.
   - Store raw values in artifacts when possible; baseline uses median for
     comparison stability.

## Baseline update policy

Update `baseline.json` when all are true:

- The change intentionally affects performance (algorithm, memory layout,
  batching, environment complexity), **or**
- Repeated reruns in normalized conditions show stable drift outside warn band,
  **and**
- You include context in the PR about why the shift is expected.

Do **not** update baseline from a single noisy CI run.

## Interpreting drift

- **< 10% drift**: typically noise, no action unless persistent.
- **10–15% drift**: warning zone. Investigate and rerun normalized measurements.
- **> 15% drift**: likely meaningful regression or workload shift; investigate,
  and either fix or explicitly re-baseline with rationale.

## Comparison script

Use `python/scripts/compare_benchmarks.py`:

```bash
python -m scripts.compare_benchmarks \
  --baseline ../docs/benchmarks/baseline.json \
  --current artifacts/benchmarks/current.json
```

Behavior:

- Warns when regression is above `warn_regression_pct` (default 12%).
- Does **not** fail by default, to avoid blocking all environments on noise.
- Can be made strict by setting a fail threshold:

```bash
python -m scripts.compare_benchmarks \
  --baseline ../docs/benchmarks/baseline.json \
  --current artifacts/benchmarks/current.json \
  --fail-threshold-pct 15
```
