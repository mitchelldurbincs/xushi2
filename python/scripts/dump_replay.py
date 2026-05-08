"""Dump a greedy eval episode to a text replay file the viewer can replay.

Usage:
    python -m scripts.dump_replay \\
        --checkpoint runs/.../ckpt_0600.pt \\
        --output replays/phase3_v3_eval.replay \\
        --seed 0xD1CEDA7A

Replay format (ASCII, line-delimited):
    Line 1: header — space-separated ``key=value`` pairs. Required keys:
        format, seed, round_seconds, action_repeat,
        mech_dmg, mech_fcd, mech_hbr, mech_resp
    Phase 3 lines: one decision per line, 13 numeric fields:
        tick mx0 my0 ad0 pf0 a10 a20 mx3 my3 ad3 pf3 a13 a23
    where slot 0 is Team A's Ranger, slot 3 is Team B's Ranger. Booleans
    are 0/1 ints. ``aim_delta`` is in radians (already scaled to ±π/4).

    Phase 4-9 and Phase 11 lines: one decision per line, 37 numeric fields:
        tick, then six action slots of
        mx my aim_delta_rad primary_fire ability_1 ability_2.
    Phase 10+ lines append target_slot per action slot, for 43 fields total.
    Phase 4 replay dumping currently requires a noop scripted opponent so the
    enemy-team slots are exact zero actions.

The viewer reads the header to construct an identical ``MatchConfig`` and
then drives a fresh ``Sim`` with the per-decision actions; the replay
relies on Phase-0 determinism rather than dumping full state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval_phase3 import load_checkpoint as load_phase3_checkpoint
from train.mappo import MappoActorCritic, MappoConfig
from train.ppo_recurrent.orchestration import make_env_fn
from train.phases import resolve_phase
from xushi2.map_randomization import (
    map_layout_hash,
    randomized_cover_markers,
    randomized_map_bounds,
    randomized_wall_segments,
)
from xushi2.snapshot_policy import SnapshotLeague
from xushi2.self_play_schedule import SelfPlaySchedule


_LEARNER_SLOT = 0
_OPPONENT_SLOT = 3
_AIM_DELTA_LIMIT = float(3.141592653589793 / 4.0)


def _action_to_fields(action_arr, *, include_target: bool = False) -> list[float]:
    """Convert a raw policy action vector to replay action fields.

    Mirrors ``Phase3RangerEnv._action_to_dict`` but returns the *radians*
    aim_delta the sim actually sees (already scaled by π/4)."""
    import numpy as np
    arr = np.asarray(action_arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 6:
        raise ValueError(f"action must have at least 6 fields, got {arr.shape[0]}")
    mx = float(np.clip(arr[0], -1.0, 1.0))
    my = float(np.clip(arr[1], -1.0, 1.0))
    ad = float(np.clip(arr[2], -1.0, 1.0)) * _AIM_DELTA_LIMIT
    pf = int(np.clip(arr[3], 0.0, 1.0) >= 0.5)
    a1 = int(np.clip(arr[4], 0.0, 1.0) >= 0.5)
    a2 = int(np.clip(arr[5], 0.0, 1.0) >= 0.5)
    fields = [mx, my, ad, float(pf), float(a1), float(a2)]
    if include_target:
        target = 0
        if arr.shape[0] >= 7:
            target = int(np.rint(arr[6]).clip(0, 255))
        fields.append(float(target))
    return fields


def _format_decision(tick: int, slot0: list[float], slot3: list[float]) -> str:
    fields = [f"{tick}"]
    for v in slot0 + slot3:
        # Compact but lossless: 7 sig figs is plenty for replay.
        fields.append(f"{v:.7g}")
    return " ".join(fields)


def _format_decision_six(tick: int, slots: list[list[float]]) -> str:
    fields = [f"{tick}"]
    for slot in slots:
        for v in slot:
            fields.append(f"{v:.7g}")
    return " ".join(fields)


def _load_phase4_checkpoint(path: str | Path) -> tuple[MappoActorCritic, dict]:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint at {path} must be a dict, got {type(ckpt)!r}")
    ckpt_config = ckpt.get("config", {})
    cfg = MappoConfig(**ckpt_config["mappo"])
    model = MappoActorCritic(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_config


def _header_fields(ckpt_config: dict, *, seed: int) -> dict[str, Any]:
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
        walls = randomized_wall_segments(
            int(seed), ckpt_config["env"].get("map_randomization", {})
        )
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
                    "1,1,1,1,1,1"
                    if sample.match_type == "current"
                    else "1,1,1,0,0,0"
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


def _dump_phase3(model, ckpt_config: dict, *, seed: int, episodes: int,
                 max_decisions: int | None, output_path: Path) -> int:
    train_config = {
        "phase": int(ckpt_config.get("phase", 3)),
        "env": ckpt_config["env"],
    }
    env_fn, _env_meta, _seed_base = make_env_fn(train_config)
    header_fields = _header_fields(ckpt_config, seed=seed)

    n_decisions = 0
    with output_path.open("w", encoding="ascii") as f:
        f.write(" ".join(f"{k}={v}" for k, v in header_fields.items()) + "\n")

        for ep in range(int(episodes)):
            env = env_fn()
            try:
                obs, info = env.reset(seed=int(seed) + ep)
                h = model.init_hidden(batch_size=1)
                done = False
                tick = int(info.get("tick", 0))
                while not done:
                    if max_decisions is not None and n_decisions >= max_decisions:
                        return n_decisions
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
                    with torch.no_grad():
                        action_t, h = model.greedy_action(obs_t, h)
                    action = action_t.squeeze(0).cpu().numpy()
                    learner_fields = _action_to_fields(action)
                    obs, _r, term, trunc, info = env.step(action)
                    opp = info["opponent_action"]
                    opponent_fields = [
                        float(opp["move_x"]), float(opp["move_y"]),
                        float(opp["aim_delta"]),
                        float(opp["primary_fire"]),
                        float(opp["ability_1"]),
                        float(opp["ability_2"]),
                    ]
                    f.write(_format_decision(tick, learner_fields, opponent_fields))
                    f.write("\n")
                    n_decisions += 1
                    tick = int(info.get("tick", tick + int(header_fields["action_repeat"])))
                    done = bool(term or trunc)
            finally:
                env.close()
    return n_decisions


def _dump_mappo(model: MappoActorCritic, ckpt_config: dict, *, seed: int,
                episodes: int, max_decisions: int | None, output_path: Path,
                stochastic: bool = False) -> int:
    wanted_phase = int(ckpt_config.get("phase", 4))
    # Phase 4-8 with non-noop opponents are viewable as long as the env
    # exposes ``info["opponent_actions"]`` each step. Phase4MappoEnv (and
    # downstream Phase 5-8 envs that inherit its opponent loop) does this,
    # so we record opponent actions per decision instead of asserting noop.
    if wanted_phase != 11 and ckpt_config["env"].get("learner_team", "A") != "A":
        raise ValueError("MAPPO replay dumping currently supports learner_team='A'")

    phase, spec = resolve_phase({"phase": wanted_phase, "env": ckpt_config["env"]})
    if phase not in (4, 5, 6, 7, 8, 9, 10, 11):
        raise AssertionError("internal phase resolution error")
    env_fn, _env_meta, _seed_base = spec["env_bundle"](
        {"phase": phase, "env": ckpt_config["env"]}
    )
    header_fields = _header_fields(ckpt_config, seed=seed)
    include_target = int(model.cfg.target_action_dim) > 0
    zero_slot = [0.0] * (7 if include_target else 6)

    n_decisions = 0
    with output_path.open("w", encoding="ascii") as f:
        f.write(" ".join(f"{k}={v}" for k, v in header_fields.items()) + "\n")

        for ep in range(int(episodes)):
            env = env_fn()
            try:
                obs, info = env.reset(seed=int(seed) + ep)
                h = model.init_hidden(model.cfg.n_agents)
                done = False
                tick = int(info.get("tick", 0))
                while not done:
                    if max_decisions is not None and n_decisions >= max_decisions:
                        return n_decisions
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)
                    with torch.no_grad():
                        if stochastic:
                            # sample_action returns (action, logprob, h_next).
                            action_t, _logprob, h = model.sample_action(
                                obs_t, h
                            )
                        else:
                            action_t, h = model.greedy_action(obs_t, h)
                    action = action_t.cpu().numpy()
                    policy_slots = [
                        _action_to_fields(action[i], include_target=include_target)
                        for i in range(model.cfg.n_agents)
                    ]
                    if phase == 11:
                        if len(policy_slots) != 6:
                            raise ValueError(
                                "phase11 replay dump requires six policy action slots"
                            )
                        obs, _reward, term, trunc, info = env.step(action)
                        if str(info.get("match_type", "current")) == "current":
                            slots = policy_slots
                        else:
                            opponent_actions = np.asarray(
                                info.get("opponent_actions"), dtype=np.float32
                            )
                            if opponent_actions.shape != (3, 6):
                                raise ValueError(
                                    "phase11 league replay dump requires "
                                    "opponent_actions info"
                                )
                            opponent_slots = [
                                _action_to_fields(opponent_actions[i])
                                for i in range(3)
                            ]
                            slots = policy_slots[:3] + opponent_slots
                    elif phase >= 4:
                        obs, _reward, term, trunc, info = env.step(action)
                        opponent_actions_raw = info.get("opponent_actions")
                        if opponent_actions_raw is None:
                            # Old code path for envs that don't expose
                            # opponent_actions: assume zero (noop) opponent.
                            slots = policy_slots[:3] + [
                                zero_slot, zero_slot, zero_slot,
                            ]
                        else:
                            opponent_actions = np.asarray(
                                opponent_actions_raw, dtype=np.float32
                            )
                            if opponent_actions.shape != (3, 6):
                                raise ValueError(
                                    "phase>=4 replay dump expects "
                                    "opponent_actions of shape (3, 6)"
                                )
                            opponent_slots = [
                                _action_to_fields(
                                    opponent_actions[i],
                                    include_target=include_target,
                                )
                                for i in range(3)
                            ]
                            slots = policy_slots[:3] + opponent_slots
                    else:
                        slots = policy_slots[:3] + [zero_slot, zero_slot, zero_slot]
                    f.write(_format_decision_six(tick, slots))
                    f.write("\n")
                    if phase < 4:
                        obs, _reward, term, trunc, info = env.step(action)
                    n_decisions += 1
                    tick = int(info.get("tick", tick + int(header_fields["action_repeat"])))
                    done = bool(term or trunc)
            finally:
                env.close()
    return n_decisions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump a greedy eval episode for the viewer to replay"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of consecutive episodes to dump")
    parser.add_argument("--max-decisions", type=int, default=None,
                        help="Optional cap for quick smoke dumps")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from the policy distribution "
                        "instead of greedy. Reflects training-time behavior.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    phase = int(raw_ckpt.get("config", {}).get("phase", 3))
    if phase in (4, 5, 6, 7, 8, 9, 10, 11):
        model, ckpt_config = _load_phase4_checkpoint(args.checkpoint)
        n_decisions = _dump_mappo(
            model,
            ckpt_config,
            seed=int(args.seed),
            episodes=int(args.episodes),
            max_decisions=args.max_decisions,
            output_path=output_path,
            stochastic=bool(args.stochastic),
        )
    else:
        model, ckpt_config = load_phase3_checkpoint(args.checkpoint)
        n_decisions = _dump_phase3(
            model,
            ckpt_config,
            seed=int(args.seed),
            episodes=int(args.episodes),
            max_decisions=args.max_decisions,
            output_path=output_path,
        )

    print(f"[dump_replay] wrote {n_decisions} decisions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
