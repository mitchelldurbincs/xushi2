# phase_gate

Configurable phase-gate evaluation. Given a phase config, a run's evidence, and
(optionally) a human review, emits a structured `GateDecision` answering: did
this run clear the gate?

## When to use

Use it once a training run has finished and you need a reproducible, machine-
readable verdict on whether the phase is cleared. The evaluator does not look
at W&B or the filesystem itself — it consumes a `RunEvidence` JSON you produce
from whatever sources the run wrote to (W&B, local CSVs, replay manifests).

## Decision flow

`evaluate_phase_gate` checks, in order:

1. **Blockers** — crash / NaN / import error / timeout-before-evidence
   → `BLOCKED`.
2. **Identity & artifacts** — git commit, config path, seeds, W&B URL,
   replay artifacts → `EVIDENCE_INSUFFICIENT` if anything required is missing.
3. **Objective checks** — for each configured metric, aggregate the series
   (mean/stddev/min/max/median over `last_n` or `all`) and compare against a
   threshold. Missing samples honor `on_missing` (`EVIDENCE_INSUFFICIENT` or
   `FAIL`). Any failing check → `NOT_CLEARED`.
4. **Subjective review** — if required and not yet supplied,
   `HUMAN_INSPECTION_REQUIRED`; if supplied and not `approved`, `NOT_CLEARED`.
5. Otherwise → `CLEARED`.

Exit codes from the CLI: `0` on `CLEARED`, `2` otherwise.

## Config

The evaluator loads two YAML layers and deep-merges defaults under the phase
config. Phase config overrides win.

**`experiments/configs/_gate_defaults.yaml`** — repo-wide defaults under
`phase_gate_defaults:`. Edit here to change the bar for all phases.

**Per-phase config** — anywhere under `experiments/configs/`, with a top-level
`phase_gate:` block. Example:

```yaml
phase_gate:
  phase: phase4_mappo_basic
  identity_requirements:
    min_unique_seeds: 3
  objective_checks:
    - id: eval_winrate
      metric: eval/winrate
      source: wandb
      aggregation: { type: mean, window: last_n, n: 5 }
      comparator: ">="
      threshold: 0.55
      min_samples: 5
      on_missing: EVIDENCE_INSUFFICIENT
    - id: train_stability
      metric: train/policy_loss
      aggregation: { type: stddev, window: last_n, n: 20 }
      comparator: "<="
      threshold: 0.5
      min_samples: 20
  subjective_checks:
    required: true
    trigger_if_objective_passed: true
    questions:
      - id: replay_sanity
        prompt: "Do the replays show purposeful target selection?"
    approval_rule: all_yes
```

Comparators: `>=`, `<=`, `>`, `<`, `==`, `!=`.
Aggregations: `mean`, `stddev`, `min`, `max`, `median`.
Windows: `last_n` (use `n`), `all`.

## Run evidence

A JSON file with the run's facts. Shape (see `models.RunEvidence`):

```json
{
  "run_id": "abc123",
  "git_commit": "deadbeef",
  "config_path": "experiments/configs/phase4/baseline/phase4_mappo_basic.yaml",
  "seeds": [1, 2, 3],
  "wandb_run_url": "https://wandb.ai/.../runs/abc123",
  "replay_artifacts": ["runs/abc/replay_000.bin"],
  "viewer_command": "xushi2-viewer --replay runs/abc/replay_000.bin",
  "crashed": false,
  "saw_nan": false,
  "import_error": false,
  "timed_out_before_evidence": false,
  "metrics": {
    "eval/winrate": [0.51, 0.54, 0.58, 0.57, 0.60],
    "train/policy_loss": [0.42, 0.39, 0.41, 0.40, 0.38]
  }
}
```

`metrics` is a dict of metric name → time-ordered list of floats. The series is
what aggregations operate on.

## Human review (optional)

YAML, passed via `--human-review`:

```yaml
decision: approved          # or rejected / needs_changes
checks:
  replay_sanity: yes
comment: "Replays look fine — clean target switching."
```

Omit the flag (or point at a missing path) to leave subjective review
unavailable; the gate then returns `HUMAN_INSPECTION_REQUIRED` when subjective
checks are required.

## CLI

```powershell
python -m python.train.phase_gate.cli `
  --phase-config experiments/configs/phase4/baseline/phase4_mappo_basic.yaml `
  --run-evidence runs/abc123/evidence.json `
  --human-review  runs/abc123/review.yaml `
  --output        runs/abc123/gate_decision.json
```

`--gate-defaults` defaults to `experiments/configs/_gate_defaults.yaml`.

The decision JSON (`--output`) is the canonical artifact — attach it to the
run, the PR, or the phase result doc.

## Programmatic

```python
from pathlib import Path
from python.train.phase_gate import evaluate_phase_gate
from python.train.phase_gate.io import (
    load_phase_gate_config, load_run_evidence, load_human_review, save_decision,
)

cfg = load_phase_gate_config(
    Path("experiments/configs/phase4/baseline/phase4_mappo_basic.yaml"),
    Path("experiments/configs/_gate_defaults.yaml"),
)
run = load_run_evidence(Path("runs/abc123/evidence.json"))
review = load_human_review(Path("runs/abc123/review.yaml"))

decision = evaluate_phase_gate(cfg, run, review)
save_decision(Path("runs/abc123/gate_decision.json"), decision)
print(decision.status, decision.final_reason)
```

## Statuses

| Status | Meaning |
|---|---|
| `CLEARED` | All objective checks passed; subjective review approved or not required. |
| `NOT_CLEARED` | Objective check(s) failed, or human review rejected. |
| `BLOCKED` | Run crashed / NaN / import error / timeout before evidence. |
| `EVIDENCE_INSUFFICIENT` | Required identity, artifacts, or samples missing. |
| `HUMAN_INSPECTION_REQUIRED` | Objectives passed; awaiting subjective review. |
