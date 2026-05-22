from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from _paths import config_path, script_path

from scripts.diagnose_full_env_teacher import run_teacher_diagnostic


def _diagnostic_config() -> dict:
    with open(
        config_path("phase4/probe/phase4_mappo_full_env_rehearsal_v2.yaml"),
        encoding="utf-8",
    ) as fh:
        cfg = yaml.safe_load(fh)
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["opponent_bot"] = "noop"
    cfg["env"]["sim"] = dict(cfg["env"]["sim"])
    cfg["env"]["sim"]["round_length_seconds"] = 2
    return cfg


def test_full_env_teacher_diagnostic_returns_finite_metrics() -> None:
    summary = run_teacher_diagnostic(
        _diagnostic_config(),
        episodes=1,
        seed=123,
        max_decisions=4,
    )

    assert summary["teacher"] == "actor_obs_scripted"
    assert summary["opponent_bot"] == "noop"
    metrics = summary["metrics"]
    for key in (
        "team_a_hit_fire",
        "team_a_visible_fire_rate",
        "objective_on_point",
        "wins",
        "losses",
        "mean_score_a",
        "mean_score_b",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_full_env_teacher_diagnostic_cli_writes_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "diag_config.yaml"
    output = tmp_path / "teacher_diag.json"
    cfg_path.write_text(yaml.safe_dump(_diagnostic_config()), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(script_path("diagnose_full_env_teacher.py")),
            "--config",
            str(cfg_path),
            "--episodes",
            "1",
            "--seed",
            "123",
            "--max-decisions",
            "4",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["episodes"] == 1
    assert payload["metrics"]["losses"] >= 0.0


def test_full_env_teacher_diagnostic_supports_cpp_bot_teacher() -> None:
    summary = run_teacher_diagnostic(
        _diagnostic_config(),
        episodes=1,
        seed=123,
        teacher="cpp_basic",
        max_decisions=4,
    )

    assert summary["teacher"] == "cpp_basic"
    assert summary["metrics"]["team_a_visible_fire_rate"] >= 0.0


def test_full_env_teacher_diagnostic_supports_multi_enemy_visible_teacher() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"),
        encoding="utf-8",
    ) as fh:
        cfg = yaml.safe_load(fh)
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["opponent_bot"] = "noop"
    cfg["env"]["sim"] = dict(cfg["env"]["sim"])
    cfg["env"]["sim"]["round_length_seconds"] = 2

    summary = run_teacher_diagnostic(
        cfg,
        episodes=1,
        seed=123,
        teacher="multi_enemy_visible",
        max_decisions=4,
    )

    assert summary["teacher"] == "multi_enemy_visible"
    assert summary["metrics"]["team_a_visible_fire_rate"] >= 0.0
