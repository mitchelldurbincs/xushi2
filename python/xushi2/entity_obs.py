"""Entity-token observation adapter for Phase 5 MAPPO experiments."""

from __future__ import annotations

import numpy as np

from xushi2.obs_manifest import ACTOR_PHASE1_DIM, actor_field_slice

ENTITY_TOKEN_COUNT: int = 3
MULTI_ENEMY_TOKEN_COUNT: int = 5
ENTITY_TOKEN_DIM: int = 18
ENTITY_OBS_DIM: int = ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM + ENTITY_TOKEN_COUNT
MULTI_ENEMY_ENTITY_OBS_DIM: int = (
    MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM + MULTI_ENEMY_TOKEN_COUNT
)

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

_SELF_TOKEN = 0
_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 2


def _as_actor_obs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] != ACTOR_PHASE1_DIM:
        raise ValueError(
            f"actor obs last dim must be {ACTOR_PHASE1_DIM}, got {obs.shape}"
        )
    return obs


def actor_obs_to_entity_obs(obs: np.ndarray) -> np.ndarray:
    """Convert Phase-4 flat actor observations to Phase-5 token observations.

    The returned layout is ``tokens.flatten()`` followed by a valid-token mask.
    Tokens are self, enemy, objective. The enemy token is masked out when the
    current flat observation reports the enemy dead.
    """
    obs = _as_actor_obs(obs)
    out_shape = (*obs.shape[:-1], ENTITY_OBS_DIM)
    out = np.zeros(out_shape, dtype=np.float32)
    tokens = out[..., : ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM].reshape(
        *obs.shape[:-1], ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM
    )
    mask = out[..., ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM :]

    own_hp = obs[..., actor_field_slice("own_hp")]
    own_velocity = obs[..., actor_field_slice("own_velocity")]
    own_aim = obs[..., actor_field_slice("own_aim_unit")]
    own_position = obs[..., actor_field_slice("own_position")]
    own_ammo = obs[..., actor_field_slice("own_ammo")]
    own_reloading = obs[..., actor_field_slice("own_reloading")]
    own_cd = obs[..., actor_field_slice("own_combat_roll_cd")]
    enemy_alive = obs[..., actor_field_slice("enemy_alive")]
    enemy_position = obs[..., actor_field_slice("enemy_relative_position")]
    enemy_hp = obs[..., actor_field_slice("enemy_hp")]
    enemy_velocity = obs[..., actor_field_slice("enemy_velocity")]
    owner_onehot = obs[..., actor_field_slice("objective_owner_onehot")]
    cap_progress = obs[..., actor_field_slice("cap_progress")]
    self_on_point = obs[..., actor_field_slice("self_on_point")]
    enemy_on_point = obs[..., actor_field_slice("enemy_on_point")]

    self_tok = tokens[..., _SELF_TOKEN, :]
    self_tok[..., _KIND_SELF] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    self_tok[..., _TEAM] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    self_tok[..., _HP] = own_hp[..., 0]
    self_tok[..., _ALIVE] = 1.0
    self_tok[..., _POSITION] = own_position
    self_tok[..., _VELOCITY] = own_velocity
    self_tok[..., _AIM] = own_aim
    self_tok[..., _AMMO] = own_ammo[..., 0]
    self_tok[..., _RELOADING] = own_reloading[..., 0]
    self_tok[..., _ABILITY_CD] = own_cd[..., 0]
    self_tok[..., _AUX] = self_on_point[..., 0]

    enemy_tok = tokens[..., _ENEMY_TOKEN, :]
    enemy_tok[..., _KIND_SELF] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    enemy_tok[..., _TEAM] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    enemy_tok[..., _HP] = enemy_hp[..., 0]
    enemy_tok[..., _ALIVE] = enemy_alive[..., 0]
    enemy_tok[..., _POSITION] = enemy_position
    enemy_tok[..., _VELOCITY] = enemy_velocity
    enemy_tok[..., _AUX] = enemy_on_point[..., 0]

    obj_tok = tokens[..., _OBJECTIVE_TOKEN, :]
    obj_tok[..., _KIND_SELF] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    obj_tok[..., _TEAM] = owner_onehot
    obj_tok[..., _ALIVE] = 1.0
    obj_tok[..., _POSITION] = -own_position
    obj_tok[..., _AUX] = cap_progress[..., 0]

    mask[..., _SELF_TOKEN] = 1.0
    mask[..., _ENEMY_TOKEN] = (enemy_alive[..., 0] > 0.5).astype(np.float32)
    mask[..., _OBJECTIVE_TOKEN] = 1.0
    return out


def entity_obs_self_position(obs: np.ndarray) -> np.ndarray:
    """Return the self-token team-frame position from flattened entity obs."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape[-1] < ENTITY_OBS_DIM:
        raise ValueError(
            f"entity obs last dim must be at least {ENTITY_OBS_DIM}, got {obs.shape}"
        )
    tokens = obs[..., : ENTITY_TOKEN_COUNT * ENTITY_TOKEN_DIM].reshape(
        *obs.shape[:-1], ENTITY_TOKEN_COUNT, ENTITY_TOKEN_DIM
    )
    return tokens[..., _SELF_TOKEN, _POSITION]
