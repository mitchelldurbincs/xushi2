"""Frozen MAPPO snapshot opponent utilities for Phase 9."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train.mappo import MappoActorCritic, mappo_config_from_checkpoint
from xushi2 import xushi2_cpp as _cpp
from xushi2.entity_obs_native import snapshot_obs_config
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import ACTOR_PHASE1_DIM
from xushi2.partial_obs import actor_obs_to_entity_grid_obs, actor_obs_to_partial_entity_grid_obs


class SnapshotPolicy:
    """Frozen recurrent MAPPO policy used as an env-side opponent.

    ``stochastic=True`` samples the frozen policy's action distribution
    (deterministically seeded per episode) instead of playing greedy. A
    greedy frozen converter camped on the objective is functionally a
    turret and teaches opponents avoidance (selfplay_l1 post-mortem);
    sampled play reproduces the distribution the snapshot actually was.
    """

    def __init__(
        self, checkpoint_path: str | Path, *, stochastic: bool = False
    ) -> None:
        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_path = str(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        self.phase = int(ckpt_config.get("phase", 4))
        self.env_cfg = dict(ckpt_config.get("env", {}))
        self.cfg = mappo_config_from_checkpoint(ckpt_config["mappo"])
        self.model = MappoActorCritic(self.cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.h = self.model.init_hidden(self.cfg.n_agents)
        # Multi-enemy entity-grid checkpoints observe through a
        # per-checkpoint ObservationEngine configured with the checkpoint's
        # TRAINING-time semantics (see snapshot_obs_config) — serving skew
        # is impossible by construction. Flat and 3-token checkpoints keep
        # their legacy conversions (deliberately not migrated).
        self._obs_engine = None
        if (
            self.cfg.obs_encoder == "entity_attention_grid"
            and self.cfg.entity_token_count > 3
        ):
            self._obs_engine = _cpp.ObservationEngine(
                snapshot_obs_config(self.phase, self.env_cfg)
            )
        self.stochastic = bool(stochastic)
        self._episode_counter = 0
        self._env_seed = 0
        self._generator: torch.Generator | None = None
        if self.stochastic:
            self._generator = torch.Generator()
            self._seed_generator()

    def _seed_generator(self) -> None:
        if self._generator is not None:
            # Deterministic per (checkpoint, env seed, episode index).
            # Mixing in the env's reset seed keeps sampling streams
            # independent across campaign seeds and across vector slots that
            # share a checkpoint (2026-08-02 review finding); crc32 rather
            # than hash() because str hashing is salted per process.
            base = zlib.crc32(self.checkpoint_path.encode("utf-8")) & 0x7FFFFFFF
            mixed = (base ^ (self._env_seed * 2654435761)) & 0x7FFFFFFF
            self._generator.manual_seed(mixed + self._episode_counter)

    def reset(self, batch_size: int | None = None, *, seed: int | None = None) -> None:
        self.h = self.model.init_hidden(
            self.cfg.n_agents if batch_size is None else int(batch_size)
        )
        if self._obs_engine is not None:
            # Episode boundary: last-seen memory must not carry across
            # matches, exactly like the training env's reset.
            self._obs_engine.reset()
        if seed is not None:
            self._env_seed = int(seed)
        self._episode_counter += 1
        self._seed_generator()

    def act(self, sim: _cpp.Sim, slots: Sequence[int]) -> np.ndarray:
        if self._obs_engine is not None:
            obs = np.zeros(
                (len(slots), MULTI_ENEMY_ENTITY_GRID_OBS_DIM), dtype=np.float32
            )
            for i, slot in enumerate(slots):
                self._obs_engine.build_entity_obs(sim, int(slot), obs[i])
        else:
            flat = np.zeros((len(slots), ACTOR_PHASE1_DIM), dtype=np.float32)
            for i, slot in enumerate(slots):
                _cpp.build_actor_obs(sim, int(slot), flat[i])
            obs = self._convert_obs(flat)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if self.h.shape[0] != len(slots):
            self.reset(batch_size=len(slots))
        with torch.no_grad():
            if self.stochastic:
                action, _logprob, self.h = self.model.sample_action(
                    obs_t, self.h, generator=self._generator
                )
            else:
                action, self.h = self.model.greedy_action(obs_t, self.h)
        return action.cpu().numpy().astype(np.float32)

    def _convert_obs(self, flat: np.ndarray) -> np.ndarray:
        # Multi-enemy entity-grid checkpoints never reach here — act()
        # builds their obs natively via the ObservationEngine.
        if self.cfg.obs_encoder == "flat":
            return flat.astype(np.float32, copy=False)
        if self.cfg.obs_encoder == "entity_attention":
            raise ValueError("entity_attention snapshot observations are no longer supported")
        if self.cfg.obs_encoder == "entity_attention_grid":
            if self.phase >= 7:
                return actor_obs_to_partial_entity_grid_obs(
                    flat,
                    visible_radius=float(self.env_cfg.get("visible_radius", 0.65)),
                    team_shared=str(self.env_cfg.get("fog_mode", "team_shared")) == "team_shared",
                )
            return actor_obs_to_entity_grid_obs(flat)
        raise ValueError(f"unsupported snapshot obs_encoder {self.cfg.obs_encoder!r}")

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
