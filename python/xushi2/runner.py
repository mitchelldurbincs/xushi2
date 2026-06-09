"""Thin Python wrapper over xushi2_cpp.run_scripted_episode.

Bot selection happens in C++. Python's job is to validate and
translate a config dict / YAML into a MatchConfig and call the binding.

Validation rules:
- semantic fields are explicit (missing required keys raise KeyError)
- unknown keys raise ValueError to catch typos early
- only optional top-level sections (`map`, `cover_circles`, `wall_segments`,
  `action_repeat`, `hero_kinds`, objective timing) may be omitted
"""

from __future__ import annotations

from dataclasses import dataclass

from . import xushi2_cpp as _cpp

_VALID_BOTS = frozenset(
    {
        "walk_to_objective",
        "hold_and_shoot",
        "basic",
        "weak_basic",
        "weak_basic_v2",
        "noop",
    }
)

_REQUIRED_MECHANICS_KEYS = frozenset(
    {
        "revolver_damage_centi_hp",
        "revolver_fire_cooldown_ticks",
        "revolver_hitbox_radius",
        "respawn_ticks",
    }
)

_HERO_KIND_BY_NAME = {
    "vanguard": _cpp.HeroKind.Vanguard,
    "ranger": _cpp.HeroKind.Ranger,
    "mender": _cpp.HeroKind.Mender,
}

_REQUIRED_SIM_KEYS = frozenset(
    {
        "seed",
        "round_length_seconds",
        "fog_of_war_enabled",
        "randomize_map",
        "mechanics",
    }
)
_OPTIONAL_SIM_KEYS = frozenset(
    {
        "map",
        "cover_circles",
        "wall_segments",
        "action_repeat",
        "hero_kinds",
        "objective_timing",
        "objective_unlock_ticks",
        "objective_unlock_seconds",
        "objective_capture_ticks",
        "objective_capture_seconds",
    }
)
_REQUIRED_MAP_KEYS = frozenset({"min_x", "min_y", "max_x", "max_y"})
_REQUIRED_COVER_KEYS = frozenset({"x", "y", "radius"})
_REQUIRED_WALL_KEYS = frozenset({"x1", "y1", "x2", "y2", "half_width"})


def _seconds_to_ticks(value: float) -> int:
    ticks = int(round(float(value) * float(_cpp.TICK_HZ)))
    if ticks <= 0:
        raise ValueError(f"objective timing seconds must resolve to >0 ticks, got {value!r}")
    return ticks


def _objective_timing_value(sim_cfg: dict, field: str, default_ticks: int) -> int:
    ticks_key = f"objective_{field}_ticks"
    seconds_key = f"objective_{field}_seconds"
    nested = dict(sim_cfg.get("objective_timing", {}))
    raw_ticks = sim_cfg.get(ticks_key, nested.get(f"{field}_ticks"))
    raw_seconds = sim_cfg.get(seconds_key, nested.get(f"{field}_seconds"))
    if raw_ticks is not None and raw_seconds is not None:
        raise ValueError(
            f"set only one of sim.{ticks_key} and sim.{seconds_key} (or objective_timing.{field}_*)"
        )
    if raw_ticks is not None:
        ticks = int(raw_ticks)
        if ticks <= 0:
            raise ValueError(f"sim.{ticks_key} must be >0, got {ticks}")
        return ticks
    if raw_seconds is not None:
        return _seconds_to_ticks(float(raw_seconds))
    return int(default_ticks)


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
    unknown_root = sim_cfg.keys() - _REQUIRED_SIM_KEYS - _OPTIONAL_SIM_KEYS
    if unknown_root:
        raise ValueError(f"sim config has unknown key(s): {sorted(unknown_root)}")
    missing_root = _REQUIRED_SIM_KEYS - sim_cfg.keys()
    if missing_root:
        raise KeyError(f"sim config is missing required key(s): {sorted(missing_root)}")

    cfg = _cpp.MatchConfig()
    cfg.seed = int(sim_cfg["seed"] if seed_override is None else seed_override)
    cfg.round_length_seconds = int(sim_cfg["round_length_seconds"])
    cfg.fog_of_war_enabled = bool(sim_cfg["fog_of_war_enabled"])
    cfg.randomize_map = bool(sim_cfg["randomize_map"])
    cfg.objective_unlock_ticks = _objective_timing_value(
        sim_cfg, "unlock", int(cfg.objective_unlock_ticks)
    )
    cfg.objective_capture_ticks = _objective_timing_value(
        sim_cfg, "capture", int(cfg.objective_capture_ticks)
    )
    if "map" in sim_cfg:
        map_cfg = sim_cfg["map"]
        unknown_map = map_cfg.keys() - _REQUIRED_MAP_KEYS
        if unknown_map:
            raise ValueError(f"sim.map has unknown key(s): {sorted(unknown_map)}")
        missing_map = _REQUIRED_MAP_KEYS - map_cfg.keys()
        if missing_map:
            raise KeyError(f"sim.map is missing required key(s): {sorted(missing_map)}")
        cfg.map.min_x = float(map_cfg["min_x"])
        cfg.map.min_y = float(map_cfg["min_y"])
        cfg.map.max_x = float(map_cfg["max_x"])
        cfg.map.max_y = float(map_cfg["max_y"])
    if "cover_circles" in sim_cfg:
        covers = []
        for raw in sim_cfg["cover_circles"]:
            missing_cover = _REQUIRED_COVER_KEYS - raw.keys()
            if missing_cover:
                raise KeyError(
                    f"sim.cover_circles entries are missing required key(s): {sorted(missing_cover)}"
                )
            unknown_cover = raw.keys() - _REQUIRED_COVER_KEYS
            if unknown_cover:
                raise ValueError(
                    f"sim.cover_circles entries have unknown key(s): {sorted(unknown_cover)}"
                )
            cover = _cpp.CoverCircle()
            center = _cpp.Vec2()
            center.x = float(raw["x"])
            center.y = float(raw["y"])
            cover.center = center
            cover.radius = float(raw["radius"])
            covers.append(cover)
        cfg.cover_circles = covers
    if "wall_segments" in sim_cfg:
        walls = []
        for raw in sim_cfg["wall_segments"]:
            missing_wall = _REQUIRED_WALL_KEYS - raw.keys()
            if missing_wall:
                raise KeyError(
                    f"sim.wall_segments entries are missing required key(s): {sorted(missing_wall)}"
                )
            unknown_wall = raw.keys() - _REQUIRED_WALL_KEYS
            if unknown_wall:
                raise ValueError(
                    f"sim.wall_segments entries have unknown key(s): {sorted(unknown_wall)}"
                )
            wall = _cpp.WallSegment()
            a = _cpp.Vec2()
            b = _cpp.Vec2()
            a.x = float(raw["x1"])
            a.y = float(raw["y1"])
            b.x = float(raw["x2"])
            b.y = float(raw["y2"])
            wall.a = a
            wall.b = b
            wall.half_width = float(raw["half_width"])
            walls.append(wall)
        cfg.wall_segments = walls
    if "action_repeat" in sim_cfg:
        cfg.action_repeat = int(sim_cfg["action_repeat"])
    if "hero_kinds" in sim_cfg:
        raw_kinds = list(sim_cfg["hero_kinds"])
        if len(raw_kinds) != 6:
            raise ValueError("sim.hero_kinds must list exactly 6 slot kinds")
        try:
            cfg.hero_kinds = [_HERO_KIND_BY_NAME[str(kind).lower()] for kind in raw_kinds]
        except KeyError as exc:
            raise ValueError("sim.hero_kinds entries must be Vanguard, Ranger, or Mender") from exc
    cfg.mechanics = _build_mechanics(sim_cfg["mechanics"])
    return cfg


def run_episode(
    sim_cfg: dict, bot_a: str, bot_b: str, seed_override: int | None = None
) -> EpisodeResult:
    """Run one scripted-vs-scripted episode and return the hash trajectory."""
    if bot_a not in _VALID_BOTS:
        raise ValueError(f"unknown bot_a {bot_a!r}; valid: {sorted(_VALID_BOTS)}")
    if bot_b not in _VALID_BOTS:
        raise ValueError(f"unknown bot_b {bot_b!r}; valid: {sorted(_VALID_BOTS)}")

    cfg = _build_config(sim_cfg, seed_override=seed_override)
    hashes, final_tick, a_kills, b_kills, winner = _cpp.run_scripted_episode(cfg, bot_a, bot_b)
    return EpisodeResult(
        decision_hashes=list(hashes),
        final_tick=int(final_tick),
        team_a_kills=int(a_kills),
        team_b_kills=int(b_kills),
        winner=int(winner),
    )
