from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from xushi2.mappo_matrix_gate import check_matrix_gate
from _paths import script_path


def _rows() -> list[dict]:
    return [
        {"opponent_type": "bot", "win_rate": 1.0, "draw_rate": 0.0},
        {"opponent_type": "snapshot", "win_rate": 0.8, "draw_rate": 0.1},
    ]


def test_matrix_gate_passes_thresholds() -> None:
    summary = check_matrix_gate(
        _rows(),
        {
            "min_rows": 2,
            "min_win_rate": {"bot": 1.0, "snapshot": 0.75},
            "max_draw_rate": {"bot": 0.0, "snapshot": 0.25},
        },
    )
    assert summary["passed"] is True
    assert summary["counts_by_type"] == {"bot": 1, "snapshot": 1}


def test_matrix_gate_reports_failures() -> None:
    summary = check_matrix_gate(
        _rows(),
        {
            "min_rows": 3,
            "min_win_rate": {"snapshot": 0.9, "selfplay": 0.1},
            "max_draw_rate": {"snapshot": 0.0},
        },
    )
    assert summary["passed"] is False
    assert any("row_count" in item for item in summary["failures"])
    assert any("win_rate" in item for item in summary["failures"])
    assert any("draw_rate" in item for item in summary["failures"])
    assert any("missing opponent_type 'selfplay'" in item for item in summary["failures"])


def test_check_mappo_matrix_cli_writes_summary(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    gate_path = tmp_path / "gate.json"
    matrix_path.write_text(json.dumps(_rows()) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("check_mappo_matrix.py")),
            "--matrix",
            str(matrix_path),
            "--min-rows",
            "2",
            "--min-win-rate",
            "bot=1.0",
            "--min-win-rate",
            "snapshot=0.75",
            "--output",
            str(gate_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS" in result.stdout
    summary = json.loads(gate_path.read_text(encoding="utf-8"))
    assert summary["passed"] is True
