"""End-to-end tests for the pybind11 obs bindings.

Verifies that xushi2_cpp.build_actor_obs / build_critic_obs write correctly
into caller-provided numpy buffers, that the dim constants exposed via the
module match the Python manifest, and that malformed buffers raise
ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from xushi2 import xushi2_cpp as _cpp
from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    CRITIC_DIM,
    actor_field_slice,
    critic_field_slice,
)


def _fresh_sim(
    seed: int = 1,
    team_size: int = 1,
    *,
    fog_of_war_enabled: bool = False,
    cover_circles: list[_cpp.CoverCircle] | None = None,
):
    cfg = _cpp.MatchConfig()
    cfg.seed = seed
    cfg.round_length_seconds = 30
    cfg.fog_of_war_enabled = fog_of_war_enabled
    cfg.randomize_map = False
    cfg.team_size = team_size
    m = _cpp.Phase1MechanicsConfig()
    m.revolver_damage_centi_hp = 7500
    m.revolver_fire_cooldown_ticks = 15
    m.revolver_hitbox_radius = 0.75
    m.respawn_ticks = 240
    cfg.mechanics = m
    if cover_circles is not None:
        cfg.cover_circles = cover_circles
    return _cpp.Sim(cfg)


def _cover_between_default_spawns() -> _cpp.CoverCircle:
    cover = _cpp.CoverCircle()
    center = _cpp.Vec2()
    center.x = 25.0
    center.y = 25.0
    cover.center = center
    cover.radius = 1.0
    return cover


def test_module_dims_match_python_manifest():
    assert _cpp.ACTOR_OBS_PHASE1_DIM == ACTOR_PHASE1_DIM
    assert _cpp.CRITIC_OBS_DIM == CRITIC_DIM


def test_build_actor_obs_fills_buffer():
    sim = _fresh_sim()
    buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, buf)
    # own_hp is the first float; fresh Ranger at full HP -> 1.0.
    own_hp_slice = actor_field_slice("own_hp")
    assert buf[own_hp_slice.start] == pytest.approx(1.0)
    assert np.all(np.isfinite(buf))


def test_build_actor_obs_slot_three_is_team_b_view():
    sim = _fresh_sim()
    a_buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    b_buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, a_buf)
    _cpp.build_actor_obs(sim, 3, b_buf)
    own_pos_slice = actor_field_slice("own_position")
    assert a_buf[own_pos_slice][1] == pytest.approx(b_buf[own_pos_slice][1])


def test_build_actor_obs_rejects_too_small_buffer():
    sim = _fresh_sim()
    too_small = np.zeros(ACTOR_PHASE1_DIM - 1, dtype=np.float32)
    with pytest.raises(ValueError):
        _cpp.build_actor_obs(sim, 0, too_small)


def test_build_actor_obs_rejects_nd_buffer():
    sim = _fresh_sim()
    nd = np.zeros((2, ACTOR_PHASE1_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        _cpp.build_actor_obs(sim, 0, nd)


def test_build_actor_obs_accepts_larger_buffer():
    sim = _fresh_sim()
    over = np.full(ACTOR_PHASE1_DIM + 10, -99.0, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, over)
    assert over[ACTOR_PHASE1_DIM] == pytest.approx(-99.0)


def test_build_actor_obs_native_fog_hides_los_blocked_alive_enemy():
    sim = _fresh_sim(
        fog_of_war_enabled=True,
        cover_circles=[_cover_between_default_spawns()],
    )
    assert _cpp.line_of_sight(sim, 0, 3) is False
    buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, buf)
    assert buf[actor_field_slice("enemy_alive")][0] == pytest.approx(0.0)
    assert np.allclose(buf[actor_field_slice("enemy_relative_position")], 0.0)
    assert buf[actor_field_slice("enemy_hp")][0] == pytest.approx(0.0)


def test_build_actor_obs_fog_disabled_keeps_los_blocked_enemy_visible():
    sim = _fresh_sim(
        fog_of_war_enabled=False,
        cover_circles=[_cover_between_default_spawns()],
    )
    assert _cpp.line_of_sight(sim, 0, 3) is False
    buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, buf)
    assert buf[actor_field_slice("enemy_alive")][0] == pytest.approx(1.0)
    assert not np.allclose(buf[actor_field_slice("enemy_relative_position")], 0.0)
    assert buf[actor_field_slice("enemy_hp")][0] == pytest.approx(1.0)


def test_observable_enemy_slots_binding_uses_native_fog_mask():
    sim = _fresh_sim(
        team_size=3,
        fog_of_war_enabled=True,
        cover_circles=[_cover_between_default_spawns()],
    )
    mask = _cpp.observable_enemy_slots(sim, 1)
    assert len(mask) == 6
    assert mask[4] is False
    assert mask[1] is False


def test_build_actor_obs_3v3_uses_counterpart_enemy_slot():
    sim = _fresh_sim(team_size=3)
    buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 1, buf)
    # Slot 1 and its counterpart slot 4 spawn on the same x lane, so the
    # team-frame relative x is zero. The old first-enemy helper leaked slot 3
    # here, producing a negative relative x for slot 1.
    rel = buf[actor_field_slice("enemy_relative_position")]
    assert rel[0] == pytest.approx(0.0)
    assert rel[1] > 0.0


def test_build_critic_obs_fills_buffer():
    sim = _fresh_sim(team_size=3)
    buf = np.zeros(CRITIC_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, buf)
    assert np.all(np.isfinite(buf))
    assert buf.shape == (CRITIC_DIM,)
    # First own slot's own_hp slot still equals 1.0 (fresh sim).
    own_hp_slice = critic_field_slice("slot0/own_hp")
    assert buf[own_hp_slice.start] == pytest.approx(1.0)


def test_build_critic_obs_enemy_world_position_is_raw_team_b_spawn():
    sim = _fresh_sim(team_size=3)
    buf = np.zeros(CRITIC_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, buf)
    # Slot 4 (middle Team-B Ranger) on default 50x50 arena spawns at
    # (cx, 45) = (25, 45) — world frame, no mirror.
    enemy1_pos = buf[critic_field_slice("enemy1/world_position")]
    assert enemy1_pos[0] == pytest.approx(25.0)
    assert enemy1_pos[1] == pytest.approx(45.0)


def test_build_critic_obs_rejects_wrong_team_perspective():
    sim = _fresh_sim(team_size=3)
    buf = np.zeros(CRITIC_DIM, dtype=np.float32)
    with pytest.raises(Exception):
        _cpp.build_critic_obs(sim, _cpp.Team.Neutral, buf)


def test_build_critic_obs_rejects_too_small_buffer():
    sim = _fresh_sim(team_size=3)
    too_small = np.zeros(CRITIC_DIM - 1, dtype=np.float32)
    with pytest.raises(ValueError):
        _cpp.build_critic_obs(sim, _cpp.Team.A, too_small)


def test_sim_exposes_kills_and_deaths_by_slot_zeroed_on_fresh_sim():
    sim = _fresh_sim(team_size=3)
    kills = sim.kills_by_slot
    deaths = sim.deaths_by_slot
    assert hasattr(kills, "__len__")
    assert hasattr(deaths, "__len__")
    assert len(kills) == _cpp.AGENTS_PER_MATCH
    assert len(deaths) == _cpp.AGENTS_PER_MATCH
    assert all(k == 0 for k in kills)
    assert all(d == 0 for d in deaths)
