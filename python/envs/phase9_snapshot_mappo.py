"""Phase-9 MAPPO env with frozen snapshot opponent sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.phase4_mappo import Phase4MappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
    sim_cfg_with_map_bounds,
)
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    normalize_world_for_team,
)
from xushi2.obs_manifest import CRITIC_DIM, actor_field_slice, critic_field_slice
from xushi2.snapshot_policy import SnapshotLeague, SnapshotPolicy

__all__ = ["Phase9SnapshotMappoEnv"]


class Phase9SnapshotMappoEnv(gym.Env):
    """Phase-8 observation stack with a frozen snapshot driving Team B."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    n_agents: int = 3
    actor_obs_dim: int = MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    critic_obs_dim: int = CRITIC_DIM
    action_dim: int = 6

    def __init__(
        self,
        sim_cfg: dict,
        *,
        opponent_bot: str,
        learner_team: str = "A",
        reward_cfg: dict[str, Any] | None = None,
        fog_mode: str = "team_shared",
        visible_radius: float = 0.65,
        map_randomization: dict[str, Any] | None = None,
        snapshot_paths: list[str] | tuple[str, ...] | None = None,
        snapshot_league: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if learner_team != "A":
            raise ValueError("Phase9SnapshotMappoEnv currently supports learner_team='A'")
        self._base_sim_cfg = dict(sim_cfg)
        self._reward_cfg = dict(reward_cfg or {})
        self._fog_mode = fog_mode
        self._team_shared = fog_mode == "team_shared"
        self._visible_radius = float(visible_radius)
        self._map_randomization = dict(map_randomization or {})
        paths = tuple(str(Path(p)) for p in (snapshot_paths or ()))
        self._snapshot_league = SnapshotLeague.from_config(paths, snapshot_league)
        self._env: Phase4MappoEnv | None = None
        self._last_map_bounds: dict[str, float] | None = None
        self._last_cover_markers: list[dict[str, float]] = []
        self._last_wall_segments: list[dict[str, float]] = []
        self._last_layout_hash: str | None = None
        self._last_snapshot_path: str | None = None
        self._last_snapshot_group: str | None = None
        self._last_seen_enemy_position = np.zeros((3, 3, 2), dtype=np.float32)
        self._last_seen_valid = np.zeros((3, 3), dtype=bool)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM),
            dtype=np.float32,
        )
        low = np.tile(
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (3, 1),
        )
        high = np.tile(
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            (3, 1),
        )
        self.action_space = spaces.Box(low=low, high=high, shape=(3, 6), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        seed_int = 0 if seed is None else int(seed)
        bounds = randomized_map_bounds(seed_int, self._map_randomization)
        covers = randomized_cover_markers(seed_int, self._map_randomization)
        walls = randomized_wall_segments(seed_int, self._map_randomization)
        layout_hash = map_layout_hash(bounds, covers, walls)
        snapshot = self._snapshot_league.sample(seed_int)
        snapshot_path = snapshot.path
        self._last_map_bounds = bounds
        self._last_cover_markers = covers
        self._last_wall_segments = walls
        self._last_layout_hash = layout_hash
        self._last_snapshot_path = snapshot_path
        self._last_snapshot_group = snapshot.group
        self._last_seen_enemy_position[:] = 0.0
        self._last_seen_valid[:] = False
        if self._env is not None:
            self._env.close()
        sim_cfg = sim_cfg_with_map_bounds(self._base_sim_cfg, bounds)
        sim_cfg["randomize_map"] = True
        sim_cfg["cover_circles"] = [dict(marker) for marker in covers]
        sim_cfg["wall_segments"] = [dict(wall) for wall in walls]
        self._env = Phase4MappoEnv(
            sim_cfg,
            opponent_bot="snapshot",
            learner_team="A",
            reward_cfg=self._reward_cfg,
            opponent_policy=SnapshotPolicy(snapshot_path),
        )
        obs, info = self._env.reset(seed=seed, options=options)
        out = self._convert(obs)
        info = dict(info)
        info["map_bounds"] = dict(bounds)
        info["cover_markers"] = [dict(marker) for marker in covers]
        info["wall_segments"] = [dict(wall) for wall in walls]
        info["map_layout_hash"] = layout_hash
        info["snapshot_path"] = snapshot_path
        info["snapshot_group"] = snapshot.group
        info["snapshot_league"] = self._snapshot_league.summary
        return out, info

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        obs, reward, terminated, truncated, info = self._env.step(action)
        out = self._convert(obs)
        info = dict(info)
        if self._last_map_bounds is not None:
            info["map_bounds"] = dict(self._last_map_bounds)
            info["cover_markers"] = [dict(marker) for marker in self._last_cover_markers]
            info["wall_segments"] = [dict(wall) for wall in self._last_wall_segments]
            info["map_layout_hash"] = self._last_layout_hash
        if self._last_snapshot_path is not None:
            info["snapshot_path"] = self._last_snapshot_path
        if self._last_snapshot_group is not None:
            info["snapshot_group"] = self._last_snapshot_group
            info["snapshot_league"] = self._snapshot_league.summary
        return out, reward, terminated, truncated, info

    def build_critic_obs(self, out: np.ndarray) -> None:
        if self._env is None:
            raise RuntimeError("reset() must be called before build_critic_obs()")
        self._env.build_critic_obs(out)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _convert(self, obs: np.ndarray) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("reset() must be called before converting obs")
        critic = np.zeros((3, CRITIC_DIM), dtype=np.float32)
        self._env.build_critic_obs(critic[0])
        critic[1:] = critic[0]
        flat = np.asarray(obs, dtype=np.float32).reshape(3, -1)
        visible = self._enemy_visibility_matrix(flat, critic)
        for row in range(3):
            for enemy_idx in range(3):
                if visible[row, enemy_idx]:
                    self._last_seen_enemy_position[row, enemy_idx] = self._enemy_norm_position(
                        critic[row], enemy_idx
                    )
                    self._last_seen_valid[row, enemy_idx] = True
        return actor_obs_to_multi_enemy_entity_grid_obs(
            obs,
            critic_obs=critic,
            map_bounds=dict(self._last_map_bounds or {}),
            visible_radius=self._visible_radius,
            visible_override=visible,
            last_seen_enemy_position=self._last_seen_enemy_position,
            last_seen_valid=self._last_seen_valid,
        )

    def _enemy_visibility_matrix(self, flat_obs: np.ndarray, critic: np.ndarray) -> np.ndarray:
        if self._env is None or self._env._sim is None:
            raise RuntimeError("reset() must be called before converting obs")
        own_slots = self._env._own_slots
        enemy_slots = self._env._enemy_slots
        own_pos = flat_obs[:, actor_field_slice("own_position")]
        enemy_pos = np.zeros((3, 3, 2), dtype=np.float32)
        alive = np.zeros((3, 3), dtype=bool)
        for row in range(3):
            for enemy_idx in range(3):
                enemy_pos[row, enemy_idx] = self._enemy_norm_position(critic[row], enemy_idx)
                alive[row, enemy_idx] = (
                    critic[row, critic_field_slice(f"enemy{enemy_idx}/alive_flag")][0] > 0.5
                )
        radius = np.linalg.norm(enemy_pos - own_pos[:, None, :], axis=2) <= float(
            self._visible_radius
        )
        los = np.zeros((3, 3), dtype=bool)
        if self._team_shared:
            for enemy_idx, enemy_slot in enumerate(enemy_slots):
                union_los = any(
                    bool(_cpp.observable_enemy_slots(self._env._sim, ally)[enemy_slot])
                    for ally in own_slots
                )
                union_radius = bool(radius[:, enemy_idx].any())
                los[:, enemy_idx] = union_los
                radius[:, enemy_idx] = union_radius
        else:
            for row, own_slot in enumerate(own_slots):
                native = _cpp.observable_enemy_slots(self._env._sim, own_slot)
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[row, enemy_idx] = bool(native[enemy_slot])
        return alive & radius & los

    def _enemy_norm_position(self, critic: np.ndarray, enemy_idx: int) -> np.ndarray:
        return normalize_world_for_team(
            critic[critic_field_slice(f"enemy{enemy_idx}/world_position")],
            dict(self._last_map_bounds or {}),
            team_b_view=False,
        )
