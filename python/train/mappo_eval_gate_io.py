from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from train.mappo_evaluate import eval_stats_dict
from train.mappo_model import MappoEvalStats
from xushi2.mappo_eval_gate import check_eval_gate


@dataclass(frozen=True)
class EvalGateConfig:
    output: str = "eval_gate.json"
    thresholds: dict | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalGateConfig":
        data = dict(payload)
        output = str(data.pop("output", "eval_gate.json"))
        return cls(output=output, thresholds=data)


def write_json_artifact(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json_artifact(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_eval_gate(*, phase_label: str, stats: MappoEvalStats, gate_cfg: EvalGateConfig, output_dir: Path) -> dict:
    thresholds = gate_cfg.thresholds or {}
    gate = check_eval_gate(eval_stats_dict(stats), thresholds)
    gate_path = output_dir / gate_cfg.output
    write_json_artifact(gate_path, gate)
    print(f"[{phase_label}/mappo] eval_gate {'pass' if gate['passed'] else 'fail'} wrote {gate_path}", flush=True)
    if not gate["passed"]:
        raise RuntimeError("MAPPO eval gate failed: " + "; ".join(gate["failures"]))
    return gate
