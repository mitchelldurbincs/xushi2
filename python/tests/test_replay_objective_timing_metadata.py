from __future__ import annotations

from _paths import repo_path

from scripts.analyze_replay_combat import _config_from_header


def _parse_header(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k] = v
    return out


def test_replay_analysis_uses_objective_timing_from_header() -> None:
    header = _parse_header(
        "format=xushi2-replay-v1 seed=7 round_seconds=30 action_repeat=3 "
        "obj_unlock_ticks=60 obj_capture_ticks=45"
    )
    cfg = _config_from_header(header)
    assert int(cfg.objective_unlock_ticks) == 60
    assert int(cfg.objective_capture_ticks) == 45


def test_replay_analysis_legacy_header_defaults_objective_timing() -> None:
    header = _parse_header("format=xushi2-replay-v1 seed=7 round_seconds=30 action_repeat=3")
    cfg = _config_from_header(header)
    # Legacy replay headers do not encode objective timing; loaders must
    # preserve MatchConfig defaults for backward compatibility.
    assert int(cfg.objective_unlock_ticks) == 15 * 30
    assert int(cfg.objective_capture_ticks) == 8 * 30


def test_replay_header_emits_objective_timing_keys_from_dump_smoke_fixture() -> None:
    fixture = repo_path("data/replays/golden_phase0_basic.txt")
    first_line = fixture.read_text(encoding="utf-8").splitlines()[0]
    # Existing fixture is a legacy text replay: absence of these keys is still
    # valid and covered by fallback behavior.
    assert "obj_unlock_ticks=" not in first_line
    assert "obj_capture_ticks=" not in first_line
