from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import evaluate_phase_gate
from .io import load_human_review, load_phase_gate_config, load_run_evidence, save_decision


def main() -> int:
    parser = argparse.ArgumentParser("phase-gate-eval")
    parser.add_argument("--phase-config", required=True, type=Path)
    parser.add_argument("--run-evidence", required=True, type=Path)
    parser.add_argument("--gate-defaults", type=Path, default=Path("experiments/configs/_gate_defaults.yaml"))
    parser.add_argument("--human-review", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_phase_gate_config(args.phase_config, args.gate_defaults)
    run = load_run_evidence(args.run_evidence)
    review = load_human_review(args.human_review) if args.human_review else None

    decision = evaluate_phase_gate(config, run, review)
    save_decision(args.output, decision)

    print(f"phase={decision.phase}")
    print(f"status={decision.status.value}")
    print(f"reason={decision.final_reason}")
    print(f"decision_artifact={args.output}")

    return 0 if decision.status.value == "CLEARED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
