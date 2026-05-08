"""Small egocentric grid observation adapter for Phase 6 diagnostics."""

from __future__ import annotations

import numpy as np

from xushi2.entity_obs import (
    ENTITY_OBS_DIM,
    MULTI_ENEMY_ENTITY_OBS_DIM,
    actor_obs_to_entity_obs,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

GRID_SIZE: int = 32
GRID_CHANNELS: int = 3
GRID_FLAT_DIM: int = GRID_CHANNELS * GRID_SIZE * GRID_SIZE
ENTITY_GRID_OBS_DIM: int = ENTITY_OBS_DIM + GRID_FLAT_DIM
MULTI_ENEMY_ENTITY_GRID_OBS_DIM: int = MULTI_ENEMY_ENTITY_OBS_DIM + GRID_FLAT_DIM


def _paint(grid: np.ndarray, channel: int, xy: np.ndarray, value: float) -> None:
    # xy is in the current actor's team-relative [-1, 1] map frame.
    ix = int(np.clip(round((float(xy[0]) + 1.0) * 0.5 * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    iy = int(np.clip(round((1.0 - (float(xy[1]) + 1.0) * 0.5) * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    grid[channel, iy, ix] = max(grid[channel, iy, ix], float(value))


def actor_obs_to_entity_grid_obs(obs: np.ndarray) -> np.ndarray:
    """Convert Phase-4 flat actor observations to entity tokens plus grid.

    Grid channels are intentionally tiny for the first Phase-6 probe:
    objective-relative marker, self marker, and visible/alive enemy marker.
    The policy still receives full-vision entity tokens; fog/LoS is a later
    phase delta.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] != ACTOR_PHASE1_DIM:
        raise ValueError(
            f"actor obs last dim must be {ACTOR_PHASE1_DIM}, got {obs.shape}"
        )
    entity_obs = actor_obs_to_entity_obs(obs)
    flat = obs.reshape(-1, ACTOR_PHASE1_DIM)
    grids = np.zeros((flat.shape[0], GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    own_pos_sl = actor_field_slice("own_position")
    enemy_pos_sl = actor_field_slice("enemy_relative_position")
    enemy_alive_sl = actor_field_slice("enemy_alive")
    for i, row in enumerate(flat):
        own_pos = row[own_pos_sl]
        _paint(grids[i], 0, -own_pos, 1.0)
        _paint(grids[i], 1, np.array([0.0, 0.0], dtype=np.float32), 1.0)
        if float(row[enemy_alive_sl][0]) > 0.5:
            _paint(grids[i], 2, row[enemy_pos_sl], 1.0)

    grid_flat = grids.reshape(*obs.shape[:-1], GRID_FLAT_DIM)
    return np.concatenate((entity_obs, grid_flat), axis=-1).astype(np.float32)
