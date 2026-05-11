from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from train.mappo import train_phase4_from_config
from xushi2.mappo_eval_gate import check_eval_gate


def test_eval_gate_passes_thresholds() -> None:
    summary = check_eval_gate(
        {
            "episodes": 2,
            "wins": 2,
            "losses": 0,
            "draws": 0,
            "mean_reward": 13.0,
            "mean_score_a": 7.0,
            "mean_score_b": 0.0,
            "mean_final_tick": 900.0,
        },
        {
            "min_episodes": 2,
            "min_win_rate": 1.0,
            "max_draw_rate": 0.0,
            "min_mean_reward": 10.0,
            "min_mean_score_a": 7.0,
            "max_mean_score_b": 0.0,
        },
    )

    assert summary["passed"] is True
    assert summary["failures"] == []
    assert summary["metrics"]["win_rate"] == 1.0


def test_eval_gate_reports_failures() -> None:
    summary = check_eval_gate(
        {
            "episodes": 2,
            "wins": 1,
            "losses": 0,
            "draws": 1,
            "mean_reward": 2.0,
            "mean_score_a": 1.0,
            "mean_score_b": 1.0,
        },
        {
            "min_win_rate": 1.0,
            "max_draw_rate": 0.0,
            "min_mean_reward": 10.0,
            "max_mean_score_b": 0.0,
        },
    )

    assert summary["passed"] is False
    assert any("min_win_rate" in failure for failure in summary["failures"])
    assert any("max_draw_rate" in failure for failure in summary["failures"])
    assert any("min_mean_reward" in failure for failure in summary["failures"])
    assert any("max_mean_score_b" in failure for failure in summary["failures"])


def test_train_config_eval_gate_writes_artifact(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4_eval_gate")
    config["run"]["eval_gate"] = {
        "min_episodes": 1,
        "max_draw_rate": 1.0,
        "output": "eval_gate.json",
    }

    train_phase4_from_config(config)

    output = tmp_path / "phase4_eval_gate" / "mappo" / "eval_gate.json"
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["metrics"]["episodes"] == 1


def test_train_config_eval_gate_fails_run(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase4/smoke/phase4_mappo_smoke.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4_eval_gate_fail")
    config["run"]["eval_gate"] = {
        "min_win_rate": 1.0,
        "max_draw_rate": 0.0,
        "output": "eval_gate.json",
    }

    with pytest.raises(RuntimeError, match="MAPPO eval gate failed"):
        train_phase4_from_config(config)
