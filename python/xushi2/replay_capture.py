"""Serialize resolved sim inputs using the existing viewer text format."""

from __future__ import annotations

import math

from xushi2 import xushi2_cpp as _cpp


def replay_header(cfg: _cpp.MatchConfig) -> str:
    # The text viewer has no randomize_map field. Refuse to silently change it.
    if cfg.randomize_map:
        raise ValueError("exact text replay capture requires randomize_map=False")
    fields = {
        "format": "xushi2-replay-v1",
        "seed": int(cfg.seed),
        "round_seconds": int(cfg.round_length_seconds),
        "action_repeat": int(cfg.action_repeat),
        "team_size": int(cfg.team_size),
        "target_slot": 1,
        "fog": int(cfg.fog_of_war_enabled),
        "obj_unlock_ticks": int(cfg.objective_unlock_ticks),
        "obj_capture_ticks": int(cfg.objective_capture_ticks),
        "mech_dmg": int(cfg.mechanics.revolver_damage_centi_hp),
        "mech_fcd": int(cfg.mechanics.revolver_fire_cooldown_ticks),
        "mech_hbr": float(cfg.mechanics.revolver_hitbox_radius),
        "mech_resp": int(cfg.mechanics.respawn_ticks),
        "map_min_x": float(cfg.map.min_x),
        "map_min_y": float(cfg.map.min_y),
        "map_max_x": float(cfg.map.max_x),
        "map_max_y": float(cfg.map.max_y),
        "heroes": ",".join(kind.name.lower() for kind in cfg.hero_kinds),
    }
    if cfg.cover_circles:
        fields["cover"] = ",".join(
            f"{c.center.x:.9g}:{c.center.y:.9g}:{c.radius:.9g}" for c in cfg.cover_circles
        )
    if cfg.wall_segments:
        fields["walls"] = ",".join(
            f"{w.a.x:.9g}:{w.a.y:.9g}:{w.b.x:.9g}:{w.b.y:.9g}:{w.half_width:.9g}"
            for w in cfg.wall_segments
        )
    return " ".join(f"{key}={value}" for key, value in fields.items())


def replay_decision(tick: int, actions: list[_cpp.Action]) -> str:
    if len(actions) != _cpp.AGENTS_PER_MATCH:
        raise ValueError("replay capture requires all six world-frame actions")
    fields = [str(int(tick))]
    for action in actions:
        values = (action.move_x, action.move_y, action.aim_delta)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite replay action")
        # Nine significant digits round-trip the actual float32 C++ fields.
        # Do not reconstruct from policy-space opponent_actions: snapshots and
        # scripted bots have different frames, and aim rescaling loses bits.
        fields.extend(f"{value:.9g}" for value in values)
        fields.extend(
            str(int(value))
            for value in (
                action.primary_fire,
                action.ability_1,
                action.ability_2,
                action.target_slot,
            )
        )
    return " ".join(fields)
