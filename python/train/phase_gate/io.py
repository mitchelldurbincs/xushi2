from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import GateDecision, HumanReview, PhaseGateConfig, RunEvidence


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_phase_gate_config(phase_config_path: Path, defaults_path: Path | None = None) -> PhaseGateConfig:
    phase_config = yaml.safe_load(phase_config_path.read_text(encoding="utf-8")) or {}
    gate_config = phase_config.get("phase_gate", {})

    if defaults_path is not None and defaults_path.exists():
        defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
        defaults_gate = defaults.get("phase_gate_defaults", {})
        gate_config = _deep_merge(defaults_gate, gate_config)

    return PhaseGateConfig(**gate_config)


def load_run_evidence(path: Path) -> RunEvidence:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunEvidence(**payload)


def load_human_review(path: Path) -> HumanReview:
    if not path.exists():
        return HumanReview(available=False)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return HumanReview(
        available=True,
        decision=payload.get("decision"),
        checks=payload.get("checks", {}),
        comment=payload.get("comment"),
    )


def save_decision(path: Path, decision: GateDecision) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
