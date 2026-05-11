"""Widened Phase-7+ entity-grid observations with three enemy tokens."""

from __future__ import annotations

import numpy as np

from xushi2.entity_obs import (
    ENTITY_TOKEN_DIM,
    MULTI_ENEMY_TOKEN_COUNT,
)
from xushi2.grid_obs import (
    GRID_CHANNELS,
    GRID_FLAT_DIM,
    GRID_SIZE,
    MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
)
from xushi2.obs_manifest import (
    ACTOR_PHASE1_DIM,
    CRITIC_DIM,
    actor_field_slice,
    critic_field_slice,
)

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


def _paint(grid: np.ndarray, channel: int, xy: np.ndarray, value: float) -> None:
    ix = int(np.clip(round((float(xy[0]) + 1.0) * 0.5 * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    iy = int(np.clip(round((1.0 - (float(xy[1]) + 1.0) * 0.5) * (GRID_SIZE - 1)), 0, GRID_SIZE - 1))
    grid[channel, iy, ix] = max(grid[channel, iy, ix], float(value))


def _enemy_world(critic: np.ndarray, enemy_idx: int, field: str) -> np.ndarray:
    return critic[critic_field_slice(f"enemy{enemy_idx}/{field}")]


def _actor(flat: np.ndarray, field: str) -> np.ndarray:
    return flat[actor_field_slice(field)]


def actor_obs_to_multi_enemy_entity_grid_obs(
    obs: np.ndarray,
    *,
    critic_obs: np.ndarray,
    map_bounds: dict[str, float],
    visible_radius: float,
    visible_override: np.ndarray | None = None,
    last_seen_enemy_position: np.ndarray | None = None,
    last_seen_valid: np.ndarray | None = None,
    team_b_view: np.ndarray | None = None,
) -> np.ndarray:
    """Build self, enemy0, enemy1, enemy2, objective tokens plus a 3-channel grid.

    ``critic_obs`` supplies all enemy world-state, but live enemy fields are
    emitted only for slots that pass the supplied visibility mask, or the local
    radius check when no mask is supplied. Hidden enemies are zeroed unless a
    stale last-seen marker is provided for that row and enemy slot.
    """
    obs = np.asarray(obs, dtype=np.float32)
    critic_obs = np.asarray(critic_obs, dtype=np.float32)
    if obs.shape[-1] != ACTOR_PHASE1_DIM:
        raise ValueError(f"actor obs last dim must be {ACTOR_PHASE1_DIM}, got {obs.shape}")
    flat_obs = obs.reshape(-1, ACTOR_PHASE1_DIM)
    flat_critic = critic_obs.reshape(-1, CRITIC_DIM)
    if flat_critic.shape[0] != flat_obs.shape[0]:
        raise ValueError(
            "critic_obs row count must match actor obs row count, "
            f"got {flat_critic.shape[0]} and {flat_obs.shape[0]}"
        )
    if visible_radius <= 0.0:
        raise ValueError("visible_radius must be positive")

    rows = flat_obs.shape[0]
    team_b = (
        np.zeros(rows, dtype=bool)
        if team_b_view is None
        else np.asarray(team_b_view, dtype=bool).reshape(-1)
    )
    if team_b.shape != (rows,):
        raise ValueError(f"team_b_view shape must be {(rows,)}, got {team_b.shape}")

    out = np.zeros(
        (rows, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
        dtype=np.float32,
    )
    token_width = MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = out[:, :token_width].reshape(rows, MULTI_ENEMY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = out[:, token_width : token_width + MULTI_ENEMY_TOKEN_COUNT]
    grids = out[:, -GRID_FLAT_DIM:].reshape(rows, GRID_CHANNELS, GRID_SIZE, GRID_SIZE)

    own_pos = flat_obs[:, actor_field_slice("own_position")]
    own_hp = flat_obs[:, actor_field_slice("own_hp")][:, 0]
    own_velocity = flat_obs[:, actor_field_slice("own_velocity")]
    own_aim = flat_obs[:, actor_field_slice("own_aim_unit")]
    own_ammo = flat_obs[:, actor_field_slice("own_ammo")][:, 0]
    own_reloading = flat_obs[:, actor_field_slice("own_reloading")][:, 0]
    own_cd = flat_obs[:, actor_field_slice("own_combat_roll_cd")][:, 0]
    self_on_point = flat_obs[:, actor_field_slice("self_on_point")][:, 0]
    owner_onehot = flat_obs[:, actor_field_slice("objective_owner_onehot")]
    cap_progress = flat_obs[:, actor_field_slice("cap_progress")][:, 0]

    self_tok = tokens[:, _SELF_TOKEN, :]
    self_tok[:, _KIND] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    self_tok[:, _TEAM] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    self_tok[:, _HP] = own_hp
    self_tok[:, _ALIVE] = 1.0
    self_tok[:, _POSITION] = own_pos
    self_tok[:, _VELOCITY] = own_velocity
    self_tok[:, _AIM] = own_aim
    self_tok[:, _AMMO] = own_ammo
    self_tok[:, _RELOADING] = own_reloading
    self_tok[:, _ABILITY_CD] = own_cd
    self_tok[:, _AUX] = self_on_point

    obj_tok = tokens[:, _OBJECTIVE_TOKEN, :]
    obj_tok[:, _KIND] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    obj_tok[:, _TEAM] = owner_onehot
    obj_tok[:, _ALIVE] = 1.0
    obj_tok[:, _POSITION] = -own_pos
    obj_tok[:, _AUX] = cap_progress

    mask[:, _SELF_TOKEN] = 1.0
    mask[:, _OBJECTIVE_TOKEN] = 1.0

    enemy_pos_norm = np.zeros((rows, 3, 2), dtype=np.float32)
    enemy_visible = np.zeros((rows, 3), dtype=bool)
    for row in range(rows):
        for enemy_idx in range(3):
            enemy_world_pos = _enemy_world(flat_critic[row], enemy_idx, "world_position")
            enemy_pos = normalize_world_for_team(
                enemy_world_pos,
                map_bounds,
                team_b_view=bool(team_b[row]),
            )
            enemy_pos_norm[row, enemy_idx] = enemy_pos
            alive = bool(_enemy_world(flat_critic[row], enemy_idx, "alive_flag")[0] > 0.5)
            rel = enemy_pos - own_pos[row]
            enemy_visible[row, enemy_idx] = alive and np.linalg.norm(rel) <= float(visible_radius)

    if visible_override is not None:
        override = np.asarray(visible_override, dtype=bool).reshape(rows, 3)
        enemy_visible = override

    stale_pos = None
    stale_valid = np.zeros((rows, 3), dtype=bool)
    if last_seen_enemy_position is not None:
        stale_pos = np.asarray(last_seen_enemy_position, dtype=np.float32).reshape(rows, 3, 2)
        if last_seen_valid is not None:
            stale_valid = np.asarray(last_seen_valid, dtype=bool).reshape(rows, 3)
        else:
            stale_valid[:] = True

    for row in range(rows):
        _paint(grids[row], 0, -own_pos[row], 1.0)
        _paint(grids[row], 1, np.array([0.0, 0.0], dtype=np.float32), 1.0)
        for enemy_idx in range(3):
            tok_idx = _FIRST_ENEMY_TOKEN + enemy_idx
            enemy_tok = tokens[row, tok_idx, :]
            enemy_tok[_KIND] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            enemy_tok[_TEAM] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if enemy_visible[row, enemy_idx]:
                rel = enemy_pos_norm[row, enemy_idx] - own_pos[row]
                enemy_tok[_HP] = _enemy_world(flat_critic[row], enemy_idx, "hp_normalized")[0]
                enemy_tok[_ALIVE] = 1.0
                enemy_tok[_POSITION] = rel
                vel = _enemy_world(flat_critic[row], enemy_idx, "world_velocity")
                enemy_tok[_VELOCITY] = (-vel if bool(team_b[row]) else vel) / _RANGER_MAX_SPEED
                enemy_tok[_AIM] = _enemy_world(flat_critic[row], enemy_idx, "world_aim_unit")
                enemy_tok[_AMMO] = _enemy_world(flat_critic[row], enemy_idx, "ammo")[0]
                enemy_tok[_RELOADING] = _enemy_world(flat_critic[row], enemy_idx, "reloading")[0]
                enemy_tok[_ABILITY_CD] = _enemy_world(
                    flat_critic[row], enemy_idx, "combat_roll_cd"
                )[0]
                dist = np.linalg.norm(
                    _enemy_world(flat_critic[row], enemy_idx, "world_position")
                    - np.array(
                        [
                            0.5 * (map_bounds["min_x"] + map_bounds["max_x"]),
                            0.5 * (map_bounds["min_y"] + map_bounds["max_y"]),
                        ],
                        dtype=np.float32,
                    )
                )
                enemy_tok[_AUX] = 1.0 if dist <= _OBJECTIVE_RADIUS else 0.0
                mask[row, tok_idx] = 1.0
                _paint(grids[row], 2, rel, 1.0)
            elif stale_pos is not None and stale_valid[row, enemy_idx]:
                rel = stale_pos[row, enemy_idx] - own_pos[row]
                enemy_tok[_ALIVE] = 0.0
                enemy_tok[_POSITION] = rel
                enemy_tok[_AUX] = 0.5
                mask[row, tok_idx] = 1.0
                _paint(grids[row], 2, rel, 0.5)

    return out.reshape(*obs.shape[:-1], MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
