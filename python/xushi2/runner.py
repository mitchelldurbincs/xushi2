"""Thin Python wrapper over xushi2_cpp.run_scripted_episode.

Bot selection happens in C++. Python's job is just to translate a
config dict / YAML into a MatchConfig and call the binding.

Phase 1a introduces a required `mechanics:` block. Missing keys raise
KeyError — no silent defaults. Extra keys raise ValueError.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import xushi2_cpp as _cpp

_VALID_BOTS = frozenset({"walk_to_objective", "hold_and_shoot", "basic", "noop"})

_REQUIRED_MECHANICS_KEYS = frozenset({
    "revolver_damage_centi_hp",
    "revolver_fire_cooldown_ticks",
    "revolver_hitbox_radius",
    "respawn_ticks",
})


@dataclass(frozen=True)
class EpisodeResult:
    decision_hashes: list[int]
    final_tick: int
    team_a_kills: int = 0
    team_b_kills: int = 0
    winner: int = 0  # 0=Neutral/draw, 1=A, 2=B


def _build_mechanics(mech_cfg: dict) -> _cpp.Phase1MechanicsConfig:
    """Build a Phase1MechanicsConfig. Every required key must be present;
    missing keys raise KeyError; unknown keys raise ValueError."""
    missing = _REQUIRED_MECHANICS_KEYS - mech_cfg.keys()
    if missing:
        raise KeyError(
            f"sim.mechanics missing required keys: {sorted(missing)}. "
            "These values have no defaults — the sim will refuse to start "
            "if any is absent. See docs/game_design.md §6 and the plan."
        )
    unknown = mech_cfg.keys() - _REQUIRED_MECHANICS_KEYS
    if unknown:
        raise ValueError(f"sim.mechanics has unknown keys: {sorted(unknown)}")

    m = _cpp.Phase1MechanicsConfig()
    m.revolver_damage_centi_hp = int(mech_cfg["revolver_damage_centi_hp"])
    m.revolver_fire_cooldown_ticks = int(mech_cfg["revolver_fire_cooldown_ticks"])
    m.revolver_hitbox_radius = float(mech_cfg["revolver_hitbox_radius"])
    m.respawn_ticks = int(mech_cfg["respawn_ticks"])
    return m


def _build_config(sim_cfg: dict, seed_override: int | None = None) -> _cpp.MatchConfig:
    if "mechanics" not in sim_cfg:
        raise KeyError(
            "sim config is missing the `mechanics` block. Phase 1a requires "
            "explicit revolver_damage_centi_hp, revolver_fire_cooldown_ticks, "
            "revolver_hitbox_radius, and respawn_ticks — no silent defaults."
        )
    cfg = _cpp.MatchConfig()
    cfg.seed = int(sim_cfg["seed"] if seed_override is None else seed_override)
    cfg.round_length_seconds = int(sim_cfg.get("round_length_seconds", 180))
    cfg.fog_of_war_enabled = bool(sim_cfg.get("fog_of_war_enabled", True))
    cfg.randomize_map = bool(sim_cfg.get("randomize_map", False))
    if "action_repeat" in sim_cfg:
        cfg.action_repeat = int(sim_cfg["action_repeat"])
    cfg.mechanics = _build_mechanics(sim_cfg["mechanics"])
    return cfg


def run_episode(sim_cfg: dict, bot_a: str, bot_b: str,
                seed_override: int | None = None) -> EpisodeResult:
    """Run one scripted-vs-scripted episode and return the hash trajectory."""
    if bot_a not in _VALID_BOTS:
        raise ValueError(f"unknown bot_a {bot_a!r}; valid: {sorted(_VALID_BOTS)}")
    if bot_b not in _VALID_BOTS:
        raise ValueError(f"unknown bot_b {bot_b!r}; valid: {sorted(_VALID_BOTS)}")

    cfg = _build_config(sim_cfg, seed_override=seed_override)
    hashes, final_tick, a_kills, b_kills, winner = _cpp.run_scripted_episode(
        cfg, bot_a, bot_b)
    return EpisodeResult(
        decision_hashes=list(hashes),
        final_tick=int(final_tick),
        team_a_kills=int(a_kills),
        team_b_kills=int(b_kills),
        winner=int(winner),
    )
