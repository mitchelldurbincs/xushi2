"""Frozen MAPPO snapshot opponent utilities for Phase 9."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train.mappo import MappoActorCritic, MappoConfig
from xushi2 import xushi2_cpp as _cpp
from xushi2.entity_obs import actor_obs_to_entity_obs
from xushi2.grid_obs import actor_obs_to_entity_grid_obs
from xushi2.multi_enemy_obs import (
    actor_obs_to_multi_enemy_entity_grid_obs,
    map_bounds_from_sim_cfg,
    normalize_world_for_team,
)
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice, critic_field_slice
from xushi2.partial_obs import actor_obs_to_partial_entity_grid_obs


class SnapshotPolicy:
    """Frozen recurrent MAPPO policy used as an env-side opponent."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_path = str(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        self.phase = int(ckpt_config.get("phase", 4))
        self.env_cfg = dict(ckpt_config.get("env", {}))
        self.cfg = MappoConfig(**ckpt_config["mappo"])
        self.model = MappoActorCritic(self.cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.h = self.model.init_hidden(self.cfg.n_agents)

    def reset(self, batch_size: int | None = None) -> None:
        self.h = self.model.init_hidden(
            self.cfg.n_agents if batch_size is None else int(batch_size)
        )

    def act(
        self,
        sim: _cpp.Sim,
        slots: Sequence[int],
        *,
        map_bounds: dict[str, float] | None = None,
    ) -> np.ndarray:
        flat = np.zeros((len(slots), ACTOR_PHASE1_DIM), dtype=np.float32)
        for i, slot in enumerate(slots):
            _cpp.build_actor_obs(sim, int(slot), flat[i])
        obs = self._convert_obs(sim, flat, slots, map_bounds=map_bounds)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if self.h.shape[0] != len(slots):
            self.reset(batch_size=len(slots))
        with torch.no_grad():
            action, self.h = self.model.greedy_action(obs_t, self.h)
        return action.cpu().numpy().astype(np.float32)

    def _convert_obs(
        self,
        sim: _cpp.Sim,
        flat: np.ndarray,
        slots: Sequence[int],
        *,
        map_bounds: dict[str, float] | None = None,
    ) -> np.ndarray:
        if self.cfg.obs_encoder == "flat":
            return flat.astype(np.float32, copy=False)
        if self.cfg.obs_encoder == "entity_attention":
            return actor_obs_to_entity_obs(flat)
        if self.cfg.obs_encoder == "entity_attention_grid":
            if self.phase >= 7:
                if self.cfg.entity_token_count > 3:
                    return self._convert_multi_enemy_obs(
                        sim,
                        flat,
                        slots,
                        map_bounds=map_bounds,
                    )
                return actor_obs_to_partial_entity_grid_obs(
                    flat,
                    visible_radius=float(self.env_cfg.get("visible_radius", 0.65)),
                    team_shared=str(self.env_cfg.get("fog_mode", "team_shared")) == "team_shared",
                )
            return actor_obs_to_entity_grid_obs(flat)
        raise ValueError(f"unsupported snapshot obs_encoder {self.cfg.obs_encoder!r}")

    def _convert_multi_enemy_obs(
        self,
        sim: _cpp.Sim,
        flat: np.ndarray,
        slots: Sequence[int],
        *,
        map_bounds: dict[str, float] | None = None,
    ) -> np.ndarray:
        rows = len(slots)
        bounds = self._map_bounds(map_bounds)
        critic = np.zeros((rows, CRITIC_DIM), dtype=np.float32)
        team_b_view = np.asarray([int(slot) >= 3 for slot in slots], dtype=bool)
        if bool(team_b_view[0]):
            _cpp.build_critic_obs(sim, _cpp.Team.B, critic[0])
        else:
            _cpp.build_critic_obs(sim, _cpp.Team.A, critic[0])
        critic[1:] = critic[0]
        visible = self._visible_enemy_matrix(
            sim,
            flat,
            critic,
            slots,
            team_b_view,
            map_bounds=bounds,
        )
        return actor_obs_to_multi_enemy_entity_grid_obs(
            flat,
            critic_obs=critic,
            map_bounds=bounds,
            visible_radius=float(self.env_cfg.get("visible_radius", 0.65)),
            visible_override=visible,
            team_b_view=team_b_view,
        )

    def _visible_enemy_matrix(
        self,
        sim: _cpp.Sim,
        flat: np.ndarray,
        critic: np.ndarray,
        slots: Sequence[int],
        team_b_view: np.ndarray,
        *,
        map_bounds: dict[str, float],
    ) -> np.ndarray:
        rows = len(slots)
        own_pos = flat[:, actor_field_slice("own_position")]
        enemy_pos = np.zeros((rows, 3, 2), dtype=np.float32)
        alive = np.zeros((rows, 3), dtype=bool)
        for row in range(rows):
            for enemy_idx in range(3):
                enemy_pos[row, enemy_idx] = normalize_world_for_team(
                    critic[row, critic_field_slice(f"enemy{enemy_idx}/world_position")],
                    map_bounds,
                    team_b_view=bool(team_b_view[row]),
                )
                alive[row, enemy_idx] = (
                    critic[row, critic_field_slice(f"enemy{enemy_idx}/alive_flag")][0] > 0.5
                )
        radius = np.linalg.norm(enemy_pos - own_pos[:, None, :], axis=2) <= float(
            self.env_cfg.get("visible_radius", 0.65)
        )
        los = np.zeros((rows, 3), dtype=bool)
        team_shared = str(self.env_cfg.get("fog_mode", "team_shared")) == "team_shared"
        for row, slot in enumerate(slots):
            enemy_slots = range(0, 3) if int(slot) >= 3 else range(3, 6)
            if team_shared:
                ally_slots = range(3, 6) if int(slot) >= 3 else range(0, 3)
                team_rows = slice(0, rows)
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[row, enemy_idx] = any(
                        bool(_cpp.observable_enemy_slots(sim, ally)[enemy_slot])
                        for ally in ally_slots
                    )
                    radius[row, enemy_idx] = bool(radius[team_rows, enemy_idx].any())
            else:
                native = _cpp.observable_enemy_slots(sim, int(slot))
                for enemy_idx, enemy_slot in enumerate(enemy_slots):
                    los[row, enemy_idx] = bool(native[enemy_slot])
        return alive & radius & los

    def _map_bounds(self, map_bounds: dict[str, float] | None) -> dict[str, float]:
        if map_bounds is not None:
            return dict(map_bounds)
        return map_bounds_from_sim_cfg(dict(self.env_cfg.get("sim", {})))

    @staticmethod
    def _resolve_checkpoint_path(path: str | Path) -> Path:
        p = Path(path)
        if p.exists() or p.is_absolute():
            return p
        python_relative = Path("python") / p
        if python_relative.exists():
            return python_relative
        return p


@dataclass(frozen=True)
class SnapshotSample:
    path: str
    group: str


class SnapshotPool:
    """Deterministic uniform sampler over frozen snapshot paths."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("snapshot pool must contain at least one path")
        self.paths = tuple(str(Path(p)) for p in paths)

    def sample_path(self, seed: int) -> str:
        rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
        idx = int(rng.integers(0, len(self.paths)))
        return self.paths[idx]

    def sample(self, seed: int) -> SnapshotSample:
        return SnapshotSample(path=self.sample_path(seed), group="pool")


class SnapshotLeague:
    """Deterministic weighted sampler over named snapshot groups."""

    _GROUP_ORDER = ("latest", "historical", "anchor")

    def __init__(
        self,
        groups: dict[str, Sequence[str | Path]],
        weights: dict[str, float] | None = None,
    ) -> None:
        cleaned: dict[str, tuple[str, ...]] = {}
        for group in self._GROUP_ORDER:
            paths = tuple(str(Path(p)) for p in groups.get(group, ()))
            if paths:
                cleaned[group] = paths
        if not cleaned:
            raise ValueError("snapshot league must contain at least one path")
        raw_weights = dict(weights or {})
        weighted: list[tuple[str, float]] = []
        for group in self._GROUP_ORDER:
            if group not in cleaned:
                continue
            weight = float(raw_weights.get(group, 1.0))
            if weight < 0.0:
                raise ValueError("snapshot league weights must be non-negative")
            if weight > 0.0:
                weighted.append((group, weight))
        if not weighted:
            raise ValueError("snapshot league must have at least one positive weight")
        self.groups = cleaned
        self.weights = dict(weighted)
        self.summary = ",".join(
            f"{group}:{self.weights[group]:.3g}:{len(self.groups[group])}"
            for group in self._GROUP_ORDER
            if group in self.weights
        )

    @classmethod
    def from_config(
        cls,
        flat_paths: Sequence[str | Path],
        league_cfg: dict | None,
    ) -> SnapshotLeague:
        if league_cfg:
            groups = {group: tuple(league_cfg.get(group, ())) for group in cls._GROUP_ORDER}
            weights = dict(league_cfg.get("weights", {}))
            return cls(groups, weights)
        return cls({"latest": tuple(flat_paths)}, {"latest": 1.0})

    def sample(self, seed: int) -> SnapshotSample:
        rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)
        groups = [g for g in self._GROUP_ORDER if g in self.weights]
        weights = np.asarray([self.weights[g] for g in groups], dtype=np.float64)
        probs = weights / weights.sum()
        group = str(rng.choice(groups, p=probs))
        paths = self.groups[group]
        idx = int(rng.integers(0, len(paths)))
        return SnapshotSample(path=paths[idx], group=group)
