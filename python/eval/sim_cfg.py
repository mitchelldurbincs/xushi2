"""Shared eval simulation-config helpers."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import yaml

_REQUIRED_SIM_TOP_LEVEL_KEYS = ("seed", "round_length_seconds", "mechanics")
_REQUIRED_MECHANICS_KEYS = (
    "revolver_damage_centi_hp",
    "revolver_fire_cooldown_ticks",
    "revolver_hitbox_radius",
    "respawn_ticks",
)


def add_sim_cfg_args(parser) -> None:
    """Register simulation configuration arguments for eval CLIs."""
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument("--round-length-seconds", type=int, default=None)
    parser.add_argument("--revolver-damage-centi-hp", type=int, default=None)
    parser.add_argument("--revolver-fire-cooldown-ticks", type=int, default=None)
    parser.add_argument("--revolver-hitbox-radius", type=float, default=None)
    parser.add_argument("--respawn-ticks", type=int, default=None)


def _load_sim_cfg_from_file(config_path: Path | None) -> dict:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if not isinstance(loaded, dict):
        raise ValueError(f"config at '{config_path}' must be a mapping")

    sim_cfg = loaded.get("sim", loaded)
    if not isinstance(sim_cfg, dict):
        raise ValueError(f"sim config at '{config_path}' must be a mapping")

    return dict(sim_cfg)


def _merge_sim_cfg_with_overrides(base_sim_cfg: dict, args: Namespace) -> dict:
    sim_cfg = dict(base_sim_cfg)
    mechanics = dict(sim_cfg.get("mechanics", {}))

    if args.seed is not None:
        sim_cfg["seed"] = args.seed
    if args.round_length_seconds is not None:
        sim_cfg["round_length_seconds"] = args.round_length_seconds
    if args.revolver_damage_centi_hp is not None:
        mechanics["revolver_damage_centi_hp"] = args.revolver_damage_centi_hp
    if args.revolver_fire_cooldown_ticks is not None:
        mechanics["revolver_fire_cooldown_ticks"] = args.revolver_fire_cooldown_ticks
    if args.revolver_hitbox_radius is not None:
        mechanics["revolver_hitbox_radius"] = args.revolver_hitbox_radius
    if args.respawn_ticks is not None:
        mechanics["respawn_ticks"] = args.respawn_ticks

    sim_cfg["mechanics"] = mechanics
    sim_cfg.setdefault("seed", 0)
    sim_cfg.setdefault("fog_of_war_enabled", False)
    sim_cfg.setdefault("randomize_map", False)
    return sim_cfg


def validate_sim_cfg(sim_cfg: dict) -> dict:
    """Validate required simulation keys and return the normalized config."""
    missing_top_level = [key for key in _REQUIRED_SIM_TOP_LEVEL_KEYS if key not in sim_cfg]
    if missing_top_level:
        raise ValueError(f"sim config missing required keys: {missing_top_level}")

    mechanics = sim_cfg.get("mechanics")
    if not isinstance(mechanics, dict):
        raise ValueError("sim config 'mechanics' must be a mapping")

    missing_mechanics = [key for key in _REQUIRED_MECHANICS_KEYS if key not in mechanics]
    if missing_mechanics:
        raise ValueError(f"sim.mechanics missing required keys: {missing_mechanics}")

    return sim_cfg


def build_sim_cfg_from_args(args: Namespace) -> dict:
    """Build simulation config from YAML + CLI overrides."""
    sim_cfg = _load_sim_cfg_from_file(args.config)
    sim_cfg = _merge_sim_cfg_with_overrides(sim_cfg, args)
    return validate_sim_cfg(sim_cfg)
