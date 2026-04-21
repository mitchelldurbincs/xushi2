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
    CRITIC_PHASE1_DIM,
    actor_field_slice,
    critic_field_slice,
)


def _fresh_sim(seed: int = 1):
    cfg = _cpp.MatchConfig()
    cfg.seed = seed
    cfg.round_length_seconds = 30
    cfg.fog_of_war_enabled = False
    cfg.randomize_map = False
    m = _cpp.Phase1MechanicsConfig()
    m.revolver_damage_centi_hp = 7500
    m.revolver_fire_cooldown_ticks = 15
    m.revolver_hitbox_radius = 0.75
    m.respawn_ticks = 240
    cfg.mechanics = m
    return _cpp.Sim(cfg)


def test_module_dims_match_python_manifest():
    assert _cpp.ACTOR_OBS_PHASE1_DIM == ACTOR_PHASE1_DIM
    assert _cpp.CRITIC_OBS_PHASE1_DIM == CRITIC_PHASE1_DIM


def test_build_actor_obs_fills_buffer():
    sim = _fresh_sim()
    buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, buf)
    # own_hp is the first float; fresh Ranger at full HP -> 1.0.
    own_hp_slice = actor_field_slice("own_hp")
    assert buf[own_hp_slice.start] == pytest.approx(1.0)
    # All entries must be finite.
    assert np.all(np.isfinite(buf))


def test_build_actor_obs_slot_three_is_team_b_view():
    sim = _fresh_sim()
    a_buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    b_buf = np.zeros(ACTOR_PHASE1_DIM, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, a_buf)
    _cpp.build_actor_obs(sim, 3, b_buf)
    # By 180° spawn symmetry, both teams see themselves at the same
    # team-frame y; the test is the complement of the C++ one.
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
    # Oversize buffer is allowed (caller may pool a batch buffer). Only the
    # first ACTOR_PHASE1_DIM floats are written.
    sim = _fresh_sim()
    over = np.full(ACTOR_PHASE1_DIM + 10, -99.0, dtype=np.float32)
    _cpp.build_actor_obs(sim, 0, over)
    # Sentinel past the written region survives.
    assert over[ACTOR_PHASE1_DIM] == pytest.approx(-99.0)


def test_build_critic_obs_fills_buffer():
    sim = _fresh_sim()
    buf = np.zeros(CRITIC_PHASE1_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, buf)
    assert np.all(np.isfinite(buf))
    # The actor-prefix's own_hp slot still equals 1.0.
    own_hp_slice = critic_field_slice("own_hp")
    assert buf[own_hp_slice.start] == pytest.approx(1.0)


def test_build_critic_obs_world_position_is_raw_team_a_spawn():
    sim = _fresh_sim()
    buf = np.zeros(CRITIC_PHASE1_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, buf)
    # Spawn for Team A on the default 50x50 arena is (25, 5) per
    # sim.cpp::reset_state — world frame (no mirror).
    world_own = buf[critic_field_slice("world_own_position")]
    assert world_own[0] == pytest.approx(25.0)
    assert world_own[1] == pytest.approx(5.0)


def test_build_critic_obs_rejects_wrong_team_perspective():
    sim = _fresh_sim()
    buf = np.zeros(CRITIC_PHASE1_DIM, dtype=np.float32)
    # Team.Neutral is not a valid perspective.
    with pytest.raises(Exception):
        _cpp.build_critic_obs(sim, _cpp.Team.Neutral, buf)


def test_build_critic_obs_rejects_too_small_buffer():
    sim = _fresh_sim()
    too_small = np.zeros(CRITIC_PHASE1_DIM - 1, dtype=np.float32)
    with pytest.raises(ValueError):
        _cpp.build_critic_obs(sim, _cpp.Team.A, too_small)
