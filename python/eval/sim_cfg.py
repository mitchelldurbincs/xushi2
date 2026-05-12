"""Shared eval simulation-config helpers."""

from __future__ import annotations

from argparse import Namespace


def add_mechanics_args(parser) -> None:
    """Register required mechanics arguments for eval CLIs."""
    parser.add_argument("--revolver-damage-centi-hp", type=int, required=True)
    parser.add_argument("--revolver-fire-cooldown-ticks", type=int, required=True)
    parser.add_argument("--revolver-hitbox-radius", type=float, required=True)
    parser.add_argument("--respawn-ticks", type=int, required=True)


def build_sim_cfg_from_args(args: Namespace) -> dict:
    """Build simulation config from CLI args."""
    return {
        "seed": args.seed,
        "round_length_seconds": args.round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "mechanics": {
            "revolver_damage_centi_hp": args.revolver_damage_centi_hp,
            "revolver_fire_cooldown_ticks": args.revolver_fire_cooldown_ticks,
            "revolver_hitbox_radius": args.revolver_hitbox_radius,
            "respawn_ticks": args.respawn_ticks,
        },
    }
