"""Entity-grid observation layout constants and map-frame helpers.

The tensors themselves are built natively by the C++ ObservationEngine
(src/sim/src/entity_obs.cpp) — docs/observation_spec.md invariant 1. This
module keeps the layout constants (lockstep with entity_obs.h), the
team-frame normalization helpers shared by replay/analysis tooling, and
entity_obs_self_position (used by BC pretraining).
"""

from __future__ import annotations

import numpy as np

ENTITY_TOKEN_DIM: int = 18
MULTI_ENEMY_TOKEN_COUNT: int = 5
GRID_SIZE: int = 32
GRID_CHANNELS: int = 3
GRID_FLAT_DIM: int = GRID_CHANNELS * GRID_SIZE * GRID_SIZE
MULTI_ENEMY_ENTITY_OBS_DIM: int = (
    MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM + MULTI_ENEMY_TOKEN_COUNT
)
MULTI_ENEMY_ENTITY_GRID_OBS_DIM: int = MULTI_ENEMY_ENTITY_OBS_DIM + GRID_FLAT_DIM


_KIND = slice(0, 3)
_TEAM = slice(3, 6)
_HP = 6
_ALIVE = 7
_POSITION = slice(8, 10)
_VELOCITY = slice(10, 12)
_AIM = slice(12, 14)
_AMMO = 14
_RELOADING = 15
_ABILITY_CD = 16
_AUX = 17

_SELF_TOKEN = 0
_FIRST_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 4
_OBJECTIVE_RADIUS = 3.0
_RANGER_MAX_SPEED = 4.2


def map_bounds_from_sim_cfg(sim_cfg: dict) -> dict[str, float]:
    raw = dict(sim_cfg.get("map", {}))
    return {
        "min_x": float(raw.get("min_x", 0.0)),
        "min_y": float(raw.get("min_y", 0.0)),
        "max_x": float(raw.get("max_x", 50.0)),
        "max_y": float(raw.get("max_y", 50.0)),
    }


def normalize_world_for_team(
    world_xy: np.ndarray,
    map_bounds: dict[str, float],
    *,
    team_b_view: bool,
) -> np.ndarray:
    xy = np.asarray(world_xy, dtype=np.float32)
    min_x = float(map_bounds["min_x"])
    min_y = float(map_bounds["min_y"])
    max_x = float(map_bounds["max_x"])
    max_y = float(map_bounds["max_y"])
    center = np.array(
        [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)],
        dtype=np.float32,
    )
    team_xy = 2.0 * center - xy if team_b_view else xy
    half = np.array(
        [0.5 * (max_x - min_x), 0.5 * (max_y - min_y)],
        dtype=np.float32,
    )
    half = np.where(half > 0.0, half, 1.0)
    return ((team_xy - center) / half).astype(np.float32)


def denormalize_team_to_world(
    team_xy: np.ndarray,
    map_bounds: dict[str, float],
    *,
    team_b_view: bool,
) -> np.ndarray:
    """Inverse of :func:`normalize_world_for_team`.

    Actor-side positions are stored map-normalized to [-1, 1]; enemy blocks in
    the critic tensor are stored in raw world units. Anything comparing the
    two -- distances, bearings -- must first put them in a single frame, and
    world is the honest choice: normalizing divides x and y by separate
    half-extents, so on a non-square map it distorts angles.
    """
    xy = np.asarray(team_xy, dtype=np.float32)
    min_x = float(map_bounds["min_x"])
    min_y = float(map_bounds["min_y"])
    max_x = float(map_bounds["max_x"])
    max_y = float(map_bounds["max_y"])
    center = np.array(
        [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)],
        dtype=np.float32,
    )
    half = np.array(
        [0.5 * (max_x - min_x), 0.5 * (max_y - min_y)],
        dtype=np.float32,
    )
    half = np.where(half > 0.0, half, 1.0)
    world = xy * half + center
    return (2.0 * center - world if team_b_view else world).astype(np.float32)


def entity_obs_self_position(obs: np.ndarray) -> np.ndarray:
    """Return the self-token team-frame position from flattened token observations."""
    obs = np.asarray(obs, dtype=np.float32)
    min_dim = ENTITY_TOKEN_DIM
    if obs.shape[-1] < min_dim:
        raise ValueError(f"entity obs last dim must be at least {min_dim}, got {obs.shape}")
    return obs[..., _POSITION]
