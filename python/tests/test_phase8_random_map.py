from __future__ import annotations

import numpy as np
import pytest

from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
)
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.runner import _build_config


def _make_sim_cfg(round_length: int = 5) -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": round_length,
        "fog_of_war_enabled": False,
        "randomize_map": True,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def test_build_config_accepts_explicit_map_bounds() -> None:
    cfg = _build_config(
        {
            **_make_sim_cfg(),
            "map": {"min_x": -1.0, "min_y": 2.0, "max_x": 49.0, "max_y": 52.0},
        }
    )
    assert cfg.map.min_x == -1.0
    assert cfg.map.min_y == 2.0
    assert cfg.map.max_x == 49.0
    assert cfg.map.max_y == 52.0


def test_build_config_accepts_hero_kind_composition() -> None:
    cfg = _build_config(
        {
            **_make_sim_cfg(),
            "hero_kinds": ["Vanguard", "Ranger", "Mender", "Vanguard", "Ranger", "Mender"],
        }
    )
    assert [kind.name for kind in cfg.hero_kinds] == [
        "Vanguard",
        "Ranger",
        "Mender",
        "Vanguard",
        "Ranger",
        "Mender",
    ]


def test_build_config_accepts_cover_circles() -> None:
    cfg = _build_config(
        {
            **_make_sim_cfg(),
            "cover_circles": [{"x": 10.0, "y": 10.0, "radius": 1.25}],
        }
    )
    covers = cfg.cover_circles
    assert len(covers) == 1
    assert covers[0].center.x == 10.0
    assert covers[0].center.y == 10.0
    assert covers[0].radius == 1.25


def test_build_config_accepts_wall_segments() -> None:
    cfg = _build_config(
        {
            **_make_sim_cfg(),
            "wall_segments": [
                {"x1": 20.0, "y1": 20.0, "x2": 20.0, "y2": 30.0, "half_width": 0.3}
            ],
        }
    )
    walls = cfg.wall_segments
    assert len(walls) == 1
    assert walls[0].a.x == 20.0
    assert walls[0].a.y == 20.0
    assert walls[0].b.x == 20.0
    assert walls[0].b.y == 30.0
    assert walls[0].half_width == pytest.approx(0.3)


def test_randomized_map_bounds_are_deterministic_and_seed_dependent() -> None:
    opts = {"span_jitter": 5.0, "min_span": 45.0, "max_span": 55.0}
    a = randomized_map_bounds(123, opts)
    b = randomized_map_bounds(123, opts)
    c = randomized_map_bounds(124, opts)
    assert a == b
    assert a != c
    assert 45.0 <= a["max_x"] - a["min_x"] <= 55.0
    assert 45.0 <= a["max_y"] - a["min_y"] <= 55.0


def test_randomized_cover_markers_are_deterministic_symmetric_and_seed_dependent() -> None:
    opts = {
        "span_jitter": 5.0,
        "min_span": 45.0,
        "max_span": 55.0,
        "cover_count_per_side": 2,
        "cover_jitter": 1.0,
        "cover_radius": 1.25,
    }
    bounds = randomized_map_bounds(123, opts)
    a = randomized_cover_markers(123, opts)
    b = randomized_cover_markers(123, opts)
    c = randomized_cover_markers(124, opts)
    cx = 0.5 * (bounds["min_x"] + bounds["max_x"])
    cy = 0.5 * (bounds["min_y"] + bounds["max_y"])
    assert a == b
    assert a != c
    assert len(a) == 4
    for marker in a:
        assert bounds["min_x"] <= marker["x"] <= bounds["max_x"]
        assert bounds["min_y"] <= marker["y"] <= bounds["max_y"]
        assert marker["radius"] == 1.25
        mirrored = {"x": 2.0 * cx - marker["x"], "y": 2.0 * cy - marker["y"]}
        assert any(
            abs(other["x"] - mirrored["x"]) < 1e-5
            and abs(other["y"] - mirrored["y"]) < 1e-5
            for other in a
        )


def test_randomized_wall_segments_are_deterministic_symmetric_and_seed_dependent() -> None:
    opts = {
        "span_jitter": 5.0,
        "min_span": 45.0,
        "max_span": 55.0,
        "wall_count_per_side": 1,
        "wall_jitter": 1.0,
        "wall_half_width": 0.25,
        "wall_length": 5.0,
    }
    bounds = randomized_map_bounds(123, opts)
    a = randomized_wall_segments(123, opts)
    b = randomized_wall_segments(123, opts)
    c = randomized_wall_segments(124, opts)
    cx = 0.5 * (bounds["min_x"] + bounds["max_x"])
    cy = 0.5 * (bounds["min_y"] + bounds["max_y"])
    assert a == b
    assert a != c
    assert len(a) == 2
    for wall in a:
        assert bounds["min_x"] <= wall["x1"] <= bounds["max_x"]
        assert bounds["min_y"] <= wall["y1"] <= bounds["max_y"]
        assert bounds["min_x"] <= wall["x2"] <= bounds["max_x"]
        assert bounds["min_y"] <= wall["y2"] <= bounds["max_y"]
        assert wall["half_width"] == 0.25
        mirrored_x = 2.0 * cx - wall["x1"]
        mirrored_y1 = 2.0 * cy - wall["y2"]
        mirrored_y2 = 2.0 * cy - wall["y1"]
        assert any(
            abs(other["x1"] - mirrored_x) < 1e-5
            and abs(other["x2"] - mirrored_x) < 1e-5
            and abs(other["y1"] - mirrored_y1) < 1e-5
            and abs(other["y2"] - mirrored_y2) < 1e-5
            for other in a
        )


def test_map_layout_hash_is_deterministic_and_geometry_dependent() -> None:
    opts = {
        "span_jitter": 5.0,
        "min_span": 45.0,
        "max_span": 55.0,
        "cover_count_per_side": 2,
        "cover_jitter": 1.0,
        "cover_radius": 1.0,
    }
    bounds = randomized_map_bounds(123, opts)
    covers = randomized_cover_markers(123, opts)
    walls = randomized_wall_segments(123, opts)
    a = map_layout_hash(bounds, covers, walls)
    b = map_layout_hash(
        dict(bounds),
        [dict(marker) for marker in covers],
        [dict(wall) for wall in walls],
    )
    covers_changed = [dict(marker) for marker in covers]
    covers_changed[0]["radius"] = 1.25
    c = map_layout_hash(bounds, covers_changed, walls)
    walls_changed = [dict(wall) for wall in walls]
    walls_changed[0]["half_width"] = 0.4
    d = map_layout_hash(bounds, covers, walls_changed)
    assert a == b
    assert a.startswith("0x")
    assert len(a) == 18
    assert a != c
    assert a != d


def test_phase8_env_resets_with_seeded_randomized_map() -> None:
    env = Phase8RandomMapMappoEnv(
        _make_sim_cfg(),
        opponent_bot="noop",
        fog_mode="team_shared",
        visible_radius=0.65,
        map_randomization={
            "span_jitter": 5.0,
            "min_span": 45.0,
            "max_span": 55.0,
            "cover_count_per_side": 2,
            "cover_jitter": 1.0,
            "cover_radius": 1.0,
            "wall_count_per_side": 1,
            "wall_jitter": 1.0,
            "wall_half_width": 0.25,
            "wall_length": 5.0,
        },
    )
    try:
        obs_a, info_a = env.reset(seed=123)
        bounds_a = info_a["map_bounds"]
        covers_a = info_a["cover_markers"]
        walls_a = info_a["wall_segments"]
        layout_a = info_a["map_layout_hash"]
        obs_b, info_b = env.reset(seed=123)
        bounds_b = info_b["map_bounds"]
        covers_b = info_b["cover_markers"]
        walls_b = info_b["wall_segments"]
        layout_b = info_b["map_layout_hash"]
        obs_c, info_c = env.reset(seed=124)
        bounds_c = info_c["map_bounds"]
        covers_c = info_c["cover_markers"]
        walls_c = info_c["wall_segments"]
        layout_c = info_c["map_layout_hash"]

        assert obs_a.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert obs_b.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert obs_c.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert bounds_a == bounds_b
        assert bounds_a != bounds_c
        assert covers_a == covers_b
        assert covers_a != covers_c
        assert walls_a == walls_b
        assert walls_a != walls_c
        assert layout_a == layout_b
        assert layout_a == map_layout_hash(bounds_a, covers_a, walls_a)
        assert layout_a != layout_c
        assert len(covers_a) == 4
        assert covers_a[0]["radius"] == 1.0
        assert len(walls_a) == 2
        assert walls_a[0]["half_width"] == 0.25

        _next_obs, _reward, _term, _trunc, step_info = env.step(
            np.zeros((3, 6), dtype=np.float32)
        )
        assert step_info["map_layout_hash"] == layout_c
        assert step_info["wall_segments"] == walls_c

        critic_obs = np.zeros(CRITIC_DIM, dtype=np.float32)
        env.build_critic_obs(critic_obs)
        assert np.all(np.isfinite(critic_obs))
    finally:
        env.close()
