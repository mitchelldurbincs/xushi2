"""Partial-observation adapter for Phase 7 diagnostics."""

from __future__ import annotations

import numpy as np

from xushi2.entity_obs import ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM
from xushi2.grid_obs import (
    ENTITY_GRID_OBS_DIM,
    GRID_CHANNELS,
    GRID_FLAT_DIM,
    GRID_SIZE,
    actor_obs_to_entity_grid_obs,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

_ENEMY_TOKEN = 1
_ENEMY_REL_POS = actor_field_slice("enemy_relative_position")
_ENEMY_ALIVE = actor_field_slice("enemy_alive")
_OWN_POSITION = actor_field_slice("own_position")

_KIND_SELF = slice(0, 3)
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


def actor_obs_to_partial_entity_grid_obs(
    obs: np.ndarray,
    *,
    visible_radius: float,
    team_shared: bool,
    visible_override: np.ndarray | None = None,
    last_seen_enemy_position: np.ndarray | None = None,
    last_seen_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Convert flat actor obs to Phase-7 masked entity+grid observations.

    This is a diagnostic fog adapter. It masks the enemy token and enemy grid
    channel when the enemy is outside ``visible_radius`` in the actor's
    team-relative frame. In team-shared mode, visibility is unioned across the
    three learner slots before masking. If a stale team-frame enemy position is
    supplied, hidden agents receive a last-seen marker instead of a live enemy.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] != ACTOR_PHASE1_DIM:
        raise ValueError(f"actor obs last dim must be {ACTOR_PHASE1_DIM}, got {obs.shape}")
    if visible_radius <= 0.0:
        raise ValueError("visible_radius must be positive")

    out = actor_obs_to_entity_grid_obs(obs)
    flat_obs = obs.reshape(-1, ACTOR_PHASE1_DIM)
    flat_out = out.reshape(-1, ENTITY_GRID_OBS_DIM)

    enemy_rel = flat_obs[:, _ENEMY_REL_POS]
    enemy_alive = flat_obs[:, _ENEMY_ALIVE][:, 0] > 0.5
    visible = (np.linalg.norm(enemy_rel, axis=1) <= float(visible_radius)) & enemy_alive
    if visible_override is not None:
        override = np.asarray(visible_override, dtype=bool).reshape(-1)
        if override.shape != visible.shape:
            raise ValueError(
                f"visible_override shape must be {visible.shape}, got {override.shape}"
            )
        visible &= override
    if team_shared:
        if flat_obs.shape[0] % 3 != 0:
            raise ValueError("team_shared partial obs expects a multiple of 3 rows")
        visible = np.repeat(visible.reshape(-1, 3).any(axis=1), 3)

    token_width = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = flat_out[:, :token_width].reshape(-1, ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = flat_out[:, token_width : token_width + ENTITY_TOKEN_COUNT]
    grids = flat_out[:, -GRID_FLAT_DIM:].reshape(-1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    hidden = ~visible
    tokens[hidden, _ENEMY_TOKEN, :] = 0.0
    mask[hidden, _ENEMY_TOKEN] = 0.0
    grids[hidden, 2, :, :] = 0.0
    if last_seen_enemy_position is not None:
        last_seen = np.asarray(last_seen_enemy_position, dtype=np.float32).reshape(-1, 2)
        if last_seen.shape != (flat_obs.shape[0], 2):
            raise ValueError(
                "last_seen_enemy_position shape must be "
                f"{(flat_obs.shape[0], 2)}, got {last_seen.shape}"
            )
        valid = np.ones(flat_obs.shape[0], dtype=bool)
        if last_seen_valid is not None:
            valid = np.asarray(last_seen_valid, dtype=bool).reshape(-1)
            if valid.shape != (flat_obs.shape[0],):
                raise ValueError(
                    f"last_seen_valid shape must be {(flat_obs.shape[0],)}, got {valid.shape}"
                )
        marker = hidden & valid
        if np.any(marker):
            own_pos = flat_obs[:, _OWN_POSITION]
            rel_pos = last_seen - own_pos
            enemy_tokens = tokens[marker, _ENEMY_TOKEN, :]
            enemy_tokens[:, _KIND_SELF] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            enemy_tokens[:, _TEAM] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            enemy_tokens[:, _HP] = 0.0
            enemy_tokens[:, _ALIVE] = 0.0
            enemy_tokens[:, _POSITION] = rel_pos[marker]
            enemy_tokens[:, _VELOCITY] = 0.0
            enemy_tokens[:, _AIM] = 0.0
            enemy_tokens[:, _AMMO] = 0.0
            enemy_tokens[:, _RELOADING] = 0.0
            enemy_tokens[:, _ABILITY_CD] = 0.0
            enemy_tokens[:, _AUX] = 0.5
            tokens[marker, _ENEMY_TOKEN, :] = enemy_tokens
            mask[marker, _ENEMY_TOKEN] = 1.0
            for idx in np.flatnonzero(marker):
                x = int(
                    np.clip(
                        round((float(rel_pos[idx, 0]) + 1.0) * 0.5 * (GRID_SIZE - 1)),
                        0,
                        GRID_SIZE - 1,
                    )
                )
                y = int(
                    np.clip(
                        round((1.0 - (float(rel_pos[idx, 1]) + 1.0) * 0.5) * (GRID_SIZE - 1)),
                        0,
                        GRID_SIZE - 1,
                    )
                )
                grids[idx, 2, y, x] = max(grids[idx, 2, y, x], 0.5)
    return out
