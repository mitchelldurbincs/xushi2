"""Analyze combat behavior from a text replay.

This diagnostic replays an existing ``.replay`` action stream through the C++
sim and reports per-slot fire commands, damage-producing hits, kill deltas, and
nearest visible target attribution at fire-command time. It intentionally uses
existing replay artifacts instead of creating another Phase 4 config.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xushi2 import xushi2_cpp as _cpp
from xushi2.obs_manifest import CRITIC_DIM, critic_field_slice

_HERO_KIND_BY_NAME = {
    "vanguard": _cpp.HeroKind.Vanguard,
    "ranger": _cpp.HeroKind.Ranger,
    "mender": _cpp.HeroKind.Mender,
}


@dataclass
class ReplayDecision:
    tick: int
    actions: list[_cpp.Action]


@dataclass
class SlotStats:
    fire_commands: int = 0
    visible_fire_commands: int = 0
    damage_hits: int = 0
    kill_deltas: int = 0
    damage_centi_hp: int = 0
    aim_error_sum: float = 0.0
    aim_error_count: int = 0
    target_counts: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        hit_rate = self.damage_hits / self.fire_commands if self.fire_commands else 0.0
        visible_rate = (
            self.visible_fire_commands / self.fire_commands if self.fire_commands else 0.0
        )
        mean_aim_error = self.aim_error_sum / self.aim_error_count if self.aim_error_count else None
        return {
            "fire_commands": self.fire_commands,
            "visible_fire_commands": self.visible_fire_commands,
            "visible_fire_command_rate": visible_rate,
            "damage_hits": self.damage_hits,
            "damage_hit_per_fire_command": hit_rate,
            "kill_deltas": self.kill_deltas,
            "damage_centi_hp": self.damage_centi_hp,
            "mean_nearest_visible_aim_error_rad": mean_aim_error,
            "aim_target_counts": {str(k): v for k, v in sorted(self.target_counts.items())},
        }


def _parse_header(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _parse_markers(raw: str | None, *, wall: bool) -> list[dict[str, float]]:
    if not raw:
        return []
    out: list[dict[str, float]] = []
    for item in raw.split(","):
        parts = item.split(":")
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if wall and len(vals) == 5:
            out.append(
                {
                    "x1": vals[0],
                    "y1": vals[1],
                    "x2": vals[2],
                    "y2": vals[3],
                    "half_width": vals[4],
                }
            )
        elif not wall and len(vals) in (2, 3):
            radius = vals[2] if len(vals) == 3 else 1.0
            out.append({"x": vals[0], "y": vals[1], "radius": radius})
    return out


def _config_from_header(header: dict[str, str]) -> _cpp.MatchConfig:
    cfg = _cpp.MatchConfig()
    cfg.seed = int(float(header.get("seed", 42)))
    cfg.round_length_seconds = int(float(header.get("round_seconds", 30)))
    cfg.fog_of_war_enabled = bool(int(float(header.get("fog", 0))))
    cfg.randomize_map = False
    cfg.action_repeat = int(float(header.get("action_repeat", 3)))
    cfg.team_size = int(float(header.get("team_size", 3)))
    cfg.map.min_x = float(header.get("map_min_x", cfg.map.min_x))
    cfg.map.min_y = float(header.get("map_min_y", cfg.map.min_y))
    cfg.map.max_x = float(header.get("map_max_x", cfg.map.max_x))
    cfg.map.max_y = float(header.get("map_max_y", cfg.map.max_y))

    mech = _cpp.Phase1MechanicsConfig()
    mech.revolver_damage_centi_hp = int(float(header.get("mech_dmg", 7500)))
    mech.revolver_fire_cooldown_ticks = int(float(header.get("mech_fcd", 15)))
    mech.revolver_hitbox_radius = float(header.get("mech_hbr", 0.75))
    mech.respawn_ticks = int(float(header.get("mech_resp", 240)))
    cfg.mechanics = mech

    covers = []
    for raw in _parse_markers(header.get("cover"), wall=False):
        cover = _cpp.CoverCircle()
        center = _cpp.Vec2()
        center.x = raw["x"]
        center.y = raw["y"]
        cover.center = center
        cover.radius = raw["radius"]
        covers.append(cover)
    if covers:
        cfg.cover_circles = covers

    walls = []
    for raw in _parse_markers(header.get("walls"), wall=True):
        wall = _cpp.WallSegment()
        a = _cpp.Vec2()
        b = _cpp.Vec2()
        a.x = raw["x1"]
        a.y = raw["y1"]
        b.x = raw["x2"]
        b.y = raw["y2"]
        wall.a = a
        wall.b = b
        wall.half_width = raw["half_width"]
        walls.append(wall)
    if walls:
        cfg.wall_segments = walls

    heroes = header.get("heroes")
    if heroes:
        kinds = [
            _HERO_KIND_BY_NAME.get(name.lower(), _cpp.HeroKind.Ranger)
            for name in heroes.split(",")
        ]
        if len(kinds) == _cpp.AGENTS_PER_MATCH:
            cfg.hero_kinds = kinds
    return cfg


def _make_action(values: list[float], offset: int, stride: int) -> _cpp.Action:
    action = _cpp.Action()
    action.move_x = values[offset]
    action.move_y = values[offset + 1]
    action.aim_delta = values[offset + 2]
    action.primary_fire = values[offset + 3] >= 0.5
    action.ability_1 = values[offset + 4] >= 0.5
    action.ability_2 = values[offset + 5] >= 0.5
    if stride >= 7:
        action.target_slot = int(max(0.0, min(255.0, values[offset + 6])))
    return action


def _load_replay(path: Path) -> tuple[dict[str, str], list[ReplayDecision]]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    header = _parse_header(lines[0])
    decisions: list[ReplayDecision] = []
    for raw in lines[1:]:
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        tick = int(fields[0])
        values = [float(v) for v in fields[1:]]
        if len(values) == _cpp.AGENTS_PER_MATCH * 6:
            stride = 6
        elif len(values) == _cpp.AGENTS_PER_MATCH * 7:
            stride = 7
        else:
            raise ValueError(f"unsupported replay action field count {len(values)} on line: {raw}")
        decisions.append(
            ReplayDecision(
                tick=tick,
                actions=[
                    _make_action(values, slot * stride, stride)
                    for slot in range(_cpp.AGENTS_PER_MATCH)
                ],
            )
        )
    return header, decisions


def _angle_wrap(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _read_critic(sim: _cpp.Sim) -> np.ndarray:
    out = np.zeros(CRITIC_DIM, dtype=np.float32)
    _cpp.build_critic_obs(sim, _cpp.Team.A, out)
    return out


def _slot_position(critic: np.ndarray, slot: int) -> np.ndarray:
    if slot < 3:
        return critic[critic_field_slice(f"slot{slot}/own_position")]
    return critic[critic_field_slice(f"enemy{slot - 3}/world_position")]


def _slot_aim_angle(critic: np.ndarray, slot: int) -> float:
    if slot < 3:
        unit = critic[critic_field_slice(f"slot{slot}/own_aim_unit")]
    else:
        unit = critic[critic_field_slice(f"enemy{slot - 3}/world_aim_unit")]
    return math.atan2(float(unit[0]), float(unit[1]))


def _slot_alive(critic: np.ndarray, slot: int) -> bool:
    if slot < 3:
        hp = float(critic[critic_field_slice(f"slot{slot}/own_hp")][0])
        return hp > 0.0
    return bool(float(critic[critic_field_slice(f"enemy{slot - 3}/alive_flag")][0]) > 0.5)


def _enemy_slots(slot: int) -> range:
    return range(3, 6) if slot < 3 else range(0, 3)


def _nearest_visible_target(
    sim: _cpp.Sim, critic: np.ndarray, slot: int
) -> tuple[int | None, float | None]:
    if not _slot_alive(critic, slot):
        return None, None
    try:
        visible = list(_cpp.observable_enemy_slots(sim, slot))
    except Exception:
        visible = [False] * _cpp.AGENTS_PER_MATCH
    own_pos = _slot_position(critic, slot)
    aim_angle = _slot_aim_angle(critic, slot)
    best_slot: int | None = None
    best_error: float | None = None
    for enemy in _enemy_slots(slot):
        if not visible[enemy] or not _slot_alive(critic, enemy):
            continue
        rel = _slot_position(critic, enemy) - own_pos
        target_angle = math.atan2(float(rel[1]), float(rel[0]))
        error = abs(_angle_wrap(aim_angle - target_angle))
        if best_error is None or error < best_error:
            best_error = error
            best_slot = enemy
    return best_slot, best_error


def _team_slots(team: str) -> range:
    return range(0, 3) if team == "A" else range(3, 6)


def analyze_replay(path: Path) -> dict:
    header, decisions = _load_replay(path)
    cfg = _config_from_header(header)
    sim = _cpp.Sim(cfg)
    stats = [SlotStats() for _ in range(_cpp.AGENTS_PER_MATCH)]
    base_seed = cfg.seed
    episodes_detected = 1 if decisions else 0
    last_tick: int | None = None

    previous_damage = np.asarray(sim.damage_dealt_by_slot, dtype=np.int64)
    previous_kills = np.asarray(sim.kills_by_slot, dtype=np.int64)

    for decision in decisions:
        if last_tick is not None and decision.tick < last_tick:
            cfg.seed = base_seed + episodes_detected
            sim = _cpp.Sim(cfg)
            previous_damage = np.asarray(sim.damage_dealt_by_slot, dtype=np.int64)
            previous_kills = np.asarray(sim.kills_by_slot, dtype=np.int64)
            episodes_detected += 1
        last_tick = decision.tick

        critic = _read_critic(sim)
        for slot, action in enumerate(decision.actions):
            if not action.primary_fire:
                continue
            slot_stats = stats[slot]
            slot_stats.fire_commands += 1
            target_slot, aim_error = _nearest_visible_target(sim, critic, slot)
            if target_slot is not None:
                slot_stats.visible_fire_commands += 1
                slot_stats.target_counts[target_slot] = (
                    slot_stats.target_counts.get(target_slot, 0) + 1
                )
            if aim_error is not None:
                slot_stats.aim_error_sum += aim_error
                slot_stats.aim_error_count += 1

        sim.step_decision(decision.actions)

        damage = np.asarray(sim.damage_dealt_by_slot, dtype=np.int64)
        kills = np.asarray(sim.kills_by_slot, dtype=np.int64)
        damage_delta = damage - previous_damage
        kill_delta = kills - previous_kills
        for slot in range(_cpp.AGENTS_PER_MATCH):
            if damage_delta[slot] > 0:
                stats[slot].damage_hits += 1
                stats[slot].damage_centi_hp += int(damage_delta[slot])
            if kill_delta[slot] > 0:
                stats[slot].kill_deltas += int(kill_delta[slot])
        previous_damage = damage
        previous_kills = kills

    team_summaries = {}
    for team in ("A", "B"):
        slots = list(_team_slots(team))
        fire_commands = sum(stats[s].fire_commands for s in slots)
        damage_hits = sum(stats[s].damage_hits for s in slots)
        visible_fire_commands = sum(stats[s].visible_fire_commands for s in slots)
        damage_centi_hp = sum(stats[s].damage_centi_hp for s in slots)
        kill_deltas = sum(stats[s].kill_deltas for s in slots)
        aim_error_sum = sum(stats[s].aim_error_sum for s in slots)
        aim_error_count = sum(stats[s].aim_error_count for s in slots)
        target_counts: dict[int, int] = {}
        for slot in slots:
            for target, count in stats[slot].target_counts.items():
                target_counts[target] = target_counts.get(target, 0) + count
        team_summaries[team] = {
            "fire_commands": fire_commands,
            "visible_fire_commands": visible_fire_commands,
            "visible_fire_command_rate": (
                visible_fire_commands / fire_commands if fire_commands else 0.0
            ),
            "damage_hits": damage_hits,
            "damage_hit_per_fire_command": (
                damage_hits / fire_commands if fire_commands else 0.0
            ),
            "kill_deltas": kill_deltas,
            "damage_centi_hp": damage_centi_hp,
            "mean_nearest_visible_aim_error_rad": (
                aim_error_sum / aim_error_count if aim_error_count else None
            ),
            "aim_target_counts": {str(k): v for k, v in sorted(target_counts.items())},
        }

    return {
        "replay": str(path),
        "header": header,
        "decisions": len(decisions),
        "episodes_detected": episodes_detected,
        "final": {
            "tick": int(sim.tick),
            "team_a_score": float(sim.team_a_score),
            "team_b_score": float(sim.team_b_score),
            "team_a_kills": int(sim.team_a_kills),
            "team_b_kills": int(sim.team_b_kills),
            "winner": str(sim.winner),
            "note": "final state is for the last detected episode; aggregates span all episodes",
        },
        "teams": team_summaries,
        "slots": {str(i): stats[i].as_dict() for i in range(_cpp.AGENTS_PER_MATCH)},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = analyze_replay(args.replay)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
