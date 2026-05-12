from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from _paths import script_path


def test_phase4_walk_objective_dump_replay_writes_six_slot_text_replay(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "phase4_walk_objective.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("diag_phase4_walk_objective.py")),
            "--dump-replay",
            str(replay_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "phase4_walk_objective" in result.stdout
    assert replay_path.exists()
    lines = replay_path.read_text(encoding="utf-8").splitlines()
    assert "team_size=3" in lines[0]
    first_decision = lines[1].split()
    assert len(first_decision) == 1 + 6 * 6
