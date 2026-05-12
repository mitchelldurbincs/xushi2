from __future__ import annotations

from train.train import NormalizedEntryConfig, format_phase_banner, run_phase


def _normalized(phase_int: int) -> NormalizedEntryConfig:
    return NormalizedEntryConfig(
        phase_int=phase_int,
        phase_label=f"phase{phase_int}",
        sim_cfg={"seed": 1},
        env_cfg={"opponent_bot": "basic", "learner_team": "A"},
        run_cfg={"episodes": 2, "team_a_bot": "basic", "team_b_bot": "noop"},
        base_seed=7,
    )


def test_format_phase_banner_phase0_default() -> None:
    banner = format_phase_banner(_normalized(0), "phase0")
    assert "episodes=2" in banner
    assert "bots=basic vs noop" in banner


def test_format_phase_banner_phase3() -> None:
    banner = format_phase_banner(_normalized(3), "phase3")
    assert "opponent=basic" in banner
    assert "learner_team=A" in banner


def test_format_phase_banner_phase11() -> None:
    banner = format_phase_banner(_normalized(11), "phase11")
    assert "match_type=current" in banner
    assert "learner_team=both" in banner


def test_run_phase_unsupported() -> None:
    rc = run_phase(_normalized(99), {"phase": "phase99"})
    assert rc == 2


def test_run_phase_phase0_without_determinism() -> None:
    n = _normalized(0)
    n = NormalizedEntryConfig(**{**n.__dict__, "run_cfg": {"assert_determinism": False}})
    rc = run_phase(n, {"phase": "phase0"})
    assert rc == 2
