from __future__ import annotations

from pathlib import Path
from typing import Any

from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
)
from xushi2.snapshot_policy import SnapshotLeague
from xushi2.self_play_schedule import SelfPlaySchedule


def header_fields(ckpt_config: dict, *, seed: int) -> dict[str, Any]:
    sim_cfg = ckpt_config["env"].get("sim", {})
    phase = int(ckpt_config.get("phase", 3))
    if phase >= 8 and sim_cfg.get("randomize_map"):
        sim_cfg = dict(sim_cfg)
        sim_cfg["map"] = randomized_map_bounds(
            int(seed), ckpt_config["env"].get("map_randomization", {})
        )
    mech = sim_cfg.get("mechanics", {})
    fields: dict[str, Any] = {
        "format": "xushi2-replay-v1",
        "phase": int(ckpt_config.get("phase", 3)),
        "seed": int(seed),
        "round_seconds": int(sim_cfg.get("round_length_seconds", 30)),
        "action_repeat": int(sim_cfg.get("action_repeat", 3)),
        "mech_dmg": int(mech.get("revolver_damage_centi_hp", 7500)),
        "mech_fcd": int(mech.get("revolver_fire_cooldown_ticks", 15)),
        "mech_hbr": float(mech.get("revolver_hitbox_radius", 0.75)),
        "mech_resp": int(mech.get("respawn_ticks", 240)),
    }
    map_cfg = sim_cfg.get("map", {})
    if map_cfg:
        fields["map_min_x"] = float(map_cfg.get("min_x", 0.0))
        fields["map_min_y"] = float(map_cfg.get("min_y", 0.0))
        fields["map_max_x"] = float(map_cfg.get("max_x", 50.0))
        fields["map_max_y"] = float(map_cfg.get("max_y", 50.0))
    if phase >= 8 and sim_cfg.get("randomize_map"):
        covers = randomized_cover_markers(
            int(seed), ckpt_config["env"].get("map_randomization", {})
        )
        walls = randomized_wall_segments(int(seed), ckpt_config["env"].get("map_randomization", {}))
        fields["layout"] = map_layout_hash(sim_cfg["map"], covers, walls)
        if covers:
            fields["cover"] = ",".join(
                f"{marker['x']:.3f}:{marker['y']:.3f}:{marker.get('radius', 1.0):.3f}"
                for marker in covers
            )
        if walls:
            fields["walls"] = ",".join(
                (
                    f"{wall['x1']:.3f}:{wall['y1']:.3f}:"
                    f"{wall['x2']:.3f}:{wall['y2']:.3f}:"
                    f"{wall.get('half_width', 0.25):.3f}"
                )
                for wall in walls
            )
    if phase in (4, 5, 6, 7, 8, 9, 10, 11):
        fields["team_size"] = 3
        mappo_cfg = ckpt_config.get("mappo", {})
        loss_mask = mappo_cfg.get("agent_loss_mask")
        if loss_mask is None:
            loss_mask = [1.0] * int(mappo_cfg.get("n_agents", 3))
        fields["loss_mask"] = ",".join(f"{float(v):.0f}" for v in loss_mask)
    if sim_cfg.get("hero_kinds"):
        fields["heroes"] = ",".join(str(k).lower() for k in sim_cfg["hero_kinds"])
    if int(ckpt_config.get("mappo", {}).get("target_action_dim", 0)) > 0:
        fields["target_slot"] = 1
    if phase >= 9:
        env_cfg = ckpt_config.get("env", {})
        if env_cfg.get("self_play_schedule"):
            schedule = SelfPlaySchedule.from_config(
                dict(env_cfg.get("self_play_schedule", {})),
                dict(env_cfg.get("snapshot_league", {})),
            )
            fields["schedule"] = schedule.summary
            if phase == 11:
                sample = schedule.sample(int(seed))
                fields["match_type"] = sample.match_type
                fields["loss_mask"] = (
                    "1,1,1,1,1,1" if sample.match_type == "current" else "1,1,1,0,0,0"
                )
                if sample.anchor_bot:
                    fields["anchor_bot"] = sample.anchor_bot
                if sample.snapshot_path:
                    fields["snapshot_group"] = sample.group
                    fields["snapshot"] = Path(sample.snapshot_path).name
        snapshot_paths = tuple(str(p) for p in env_cfg.get("snapshot_paths", ()))
        if phase != 11 and (snapshot_paths or env_cfg.get("snapshot_league")):
            league = SnapshotLeague.from_config(
                snapshot_paths, dict(env_cfg.get("snapshot_league", {}))
            )
            sample = league.sample(int(seed))
            fields["league"] = league.summary
            fields["snapshot_group"] = sample.group
            fields["snapshot"] = Path(sample.path).name
    if phase >= 7:
        fields["fog"] = 1
        fields["last_seen"] = 1
        fields["fog_mode"] = str(ckpt_config.get("env", {}).get("fog_mode", "team_shared"))
    if ckpt_config.get("env", {}).get("match_type") and "match_type" not in fields:
        fields["match_type"] = str(ckpt_config["env"]["match_type"])
    return fields
