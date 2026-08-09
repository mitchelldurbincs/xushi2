"""Boundary-validation tests for the pybind11 layer.

The sim enforces its preconditions with X2_REQUIRE, which calls
``std::abort()``. That is correct inside the sim -- a Tier-0 invariant
violation means the state is untrustworthy -- but unusable at the Python
boundary: an abort gives no traceback, cannot be caught, and skips
``finally`` blocks. Under the async vector env it also strands the parent
waiting on a pipe that will never be written to.

Every case below reaches an abort path in the sim. Each must surface as a
``ValueError`` instead. If any of these tests *crashes* the interpreter
rather than failing, the binding-layer guard for that path has regressed.

Invariant under test: an X2_REQUIRE cannot be reached from Python.
"""

from __future__ import annotations

import numpy as np
import pytest

from xushi2 import xushi2_cpp as _cpp
from xushi2.obs_manifest import CRITIC_DIM


def _valid_mechanics() -> _cpp.Phase1MechanicsConfig:
    m = _cpp.Phase1MechanicsConfig()
    m.revolver_damage_centi_hp = 7500
    m.revolver_fire_cooldown_ticks = 15
    m.revolver_hitbox_radius = 0.75
    m.respawn_ticks = 240
    return m


def _valid_config(team_size: int = 1) -> _cpp.MatchConfig:
    cfg = _cpp.MatchConfig()
    cfg.seed = 1
    cfg.round_length_seconds = 30
    cfg.fog_of_war_enabled = False
    cfg.randomize_map = False
    cfg.team_size = team_size
    cfg.mechanics = _valid_mechanics()
    return cfg


def test_valid_config_still_constructs() -> None:
    """Guard against the validator rejecting good configs."""
    sim = _cpp.Sim(_valid_config())
    assert sim.tick == 0


def test_default_constructed_config_raises_not_aborts() -> None:
    # Every mechanics field is at its sentinel. This is the single most
    # likely way a user reaches the sim's abort path.
    with pytest.raises(ValueError, match="mechanics"):
        _cpp.Sim(_cpp.MatchConfig())


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("revolver_damage_centi_hp", 0, "revolver_damage_centi_hp"),
        ("revolver_fire_cooldown_ticks", 0, "revolver_fire_cooldown_ticks"),
        ("revolver_hitbox_radius", -1.0, "revolver_hitbox_radius"),
        ("revolver_hitbox_radius", float("nan"), "revolver_hitbox_radius"),
        ("respawn_ticks", 0, "respawn_ticks"),
    ],
)
def test_invalid_mechanics_field_raises(field: str, value: float, needle: str) -> None:
    cfg = _valid_config()
    m = _valid_mechanics()
    setattr(m, field, value)
    cfg.mechanics = m
    with pytest.raises(ValueError, match=needle):
        _cpp.Sim(cfg)


@pytest.mark.parametrize("unset_field", ["revolver_damage_centi_hp", "respawn_ticks"])
def test_unset_uint_mechanics_field_raises(unset_field: str) -> None:
    cfg = _valid_config()
    m = _valid_mechanics()
    setattr(m, unset_field, 0xFFFFFFFF)  # the sentinel meaning "unset"
    cfg.mechanics = m
    with pytest.raises(ValueError, match="unset"):
        _cpp.Sim(cfg)


@pytest.mark.parametrize("action_repeat", [0, 1, 4, 255])
def test_invalid_action_repeat_raises(action_repeat: int) -> None:
    cfg = _valid_config()
    cfg.action_repeat = action_repeat
    with pytest.raises(ValueError, match="action_repeat"):
        _cpp.Sim(cfg)


@pytest.mark.parametrize("team_size", [0, 2, 4, 6])
def test_invalid_team_size_raises(team_size: int) -> None:
    cfg = _valid_config()
    cfg.team_size = team_size
    with pytest.raises(ValueError, match="team_size"):
        _cpp.Sim(cfg)


@pytest.mark.parametrize("timing_field", ["objective_unlock_ticks", "objective_capture_ticks"])
def test_zero_objective_timing_raises(timing_field: str) -> None:
    cfg = _valid_config()
    setattr(cfg, timing_field, 0)
    with pytest.raises(ValueError, match=timing_field):
        _cpp.Sim(cfg)


def test_inverted_map_bounds_raise() -> None:
    cfg = _valid_config()
    cfg.map.max_x = cfg.map.min_x
    with pytest.raises(ValueError, match="map"):
        _cpp.Sim(cfg)


def test_cover_circle_outside_map_raises() -> None:
    cfg = _valid_config()
    cover = _cpp.CoverCircle()
    center = _cpp.Vec2()
    center.x = 5.0
    center.y = 5.0
    cover.center = center
    cover.radius = 100.0  # extends far outside the 50x50 arena
    cfg.cover_circles = [cover]
    with pytest.raises(ValueError, match="cover_circles"):
        _cpp.Sim(cfg)


def test_zero_length_wall_raises() -> None:
    cfg = _valid_config()
    wall = _cpp.WallSegment()
    a = _cpp.Vec2()
    a.x = a.y = 10.0
    b = _cpp.Vec2()
    b.x = b.y = 10.0  # identical endpoints
    wall.a = a
    wall.b = b
    wall.half_width = 0.25
    cfg.wall_segments = [wall]
    with pytest.raises(ValueError, match="wall_segments"):
        _cpp.Sim(cfg)


def test_build_critic_obs_on_team_size_1_raises() -> None:
    # The critic builder asserts team_size == 3; that assert aborts.
    sim = _cpp.Sim(_valid_config(team_size=1))
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    with pytest.raises(ValueError, match="team_size"):
        _cpp.build_critic_obs(sim, _cpp.Team.A, out)


def test_build_critic_obs_on_team_size_3_succeeds() -> None:
    sim = _cpp.Sim(_valid_config(team_size=3))
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, out)
    assert np.isfinite(out).all()


def test_scripted_bot_action_unknown_name_raises() -> None:
    sim = _cpp.Sim(_valid_config())
    with pytest.raises(ValueError, match="unknown bot_name"):
        _cpp.scripted_bot_action(sim, 0, "definitely_not_a_bot")


@pytest.mark.parametrize(("bot_a", "bot_b"), [("nope", "basic"), ("basic", "nope")])
def test_run_scripted_episode_unknown_name_raises(bot_a: str, bot_b: str) -> None:
    # make_bot_by_name aborts on an unknown name; this binding previously
    # passed the string straight through to it.
    with pytest.raises(ValueError, match="unknown bot_"):
        _cpp.run_scripted_episode(_valid_config(), bot_a, bot_b)


def test_run_scripted_episode_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="mechanics"):
        _cpp.run_scripted_episode(_cpp.MatchConfig(), "basic", "basic")


def test_run_scripted_episode_valid_still_runs() -> None:
    hashes, final_tick, _a_kills, _b_kills, _winner = _cpp.run_scripted_episode(
        _valid_config(), "basic", "basic"
    )
    assert final_tick > 0
    assert len(hashes) > 0
