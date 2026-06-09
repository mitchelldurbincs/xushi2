from __future__ import annotations

from train.train import NormalizedEntryConfig, format_phase_banner, run_phase


def _normalized(
    phase_int: int | None,
    *,
    learner_kind: str = "scripted_determinism",
    env_kind: str = "scripted_determinism",
) -> NormalizedEntryConfig:
    return NormalizedEntryConfig(
        phase_int=phase_int,
        phase_label=f"phase{phase_int}" if phase_int is not None else env_kind,
        sim_cfg={"seed": 1},
        env_cfg={"opponent_bot": "basic", "learner_team": "A"},
        run_cfg={"episodes": 2, "team_a_bot": "basic", "team_b_bot": "noop"},
        base_seed=7,
        learner_kind=learner_kind,
        env_kind=env_kind,
    )


def test_format_phase_banner_phase0_default() -> None:
    banner = format_phase_banner(_normalized(0), "phase0")
    assert "episodes=2" in banner
    assert "bots=basic vs noop" in banner


def test_format_phase_banner_phase11() -> None:
    n = _normalized(11, learner_kind="mappo", env_kind="mappo_match")
    n = NormalizedEntryConfig(**{**n.__dict__, "env_cfg": {"n_agents": 6}})
    banner = format_phase_banner(n, "phase11")
    assert "match_type=current" in banner
    assert "mappo" in banner


def test_format_phase_banner_phase4_selfplay() -> None:
    n = _normalized(4, learner_kind="mappo", env_kind="mappo_match")
    n = NormalizedEntryConfig(
        **{
            **n.__dict__,
            "env_cfg": {"self_play": {"enabled": True}},
        }
    )
    banner = format_phase_banner(n, "phase4")
    assert "match_type=current" in banner
    assert "mappo" in banner


def test_format_phase_banner_uses_experiment_phase_metadata() -> None:
    n = _normalized(4, learner_kind="mappo", env_kind="mappo_match")
    banner = format_phase_banner(n, "unknown")
    assert "phase=phase4" in banner
    assert "phase=unknown" not in banner


def test_run_phase_unsupported() -> None:
    rc = run_phase(
        _normalized(99, learner_kind="unknown", env_kind="unknown"),
        {"phase": "phase99"},
    )
    assert rc == 2


def test_run_phase_phase0_without_determinism() -> None:
    n = _normalized(0)
    n = NormalizedEntryConfig(**{**n.__dict__, "run_cfg": {"assert_determinism": False}})
    rc = run_phase(n, {"phase": "phase0"})
    assert rc == 2
