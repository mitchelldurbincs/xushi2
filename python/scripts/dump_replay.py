"""Dump a Phase-3 greedy eval episode to a replay file the viewer can replay.

Usage:
    python -m scripts.dump_replay \\
        --checkpoint runs/.../ckpt_0600.pt \\
        --output replays/phase3_v3_eval.replay \\
        --seed 0xD1CEDA7A

Replay format (ASCII, line-delimited):
    Line 1: header — space-separated ``key=value`` pairs. Required keys:
        format, seed, round_seconds, action_repeat,
        mech_dmg, mech_fcd, mech_hbr, mech_resp
    Lines 2..N: one decision per line, 13 numeric fields:
        tick mx0 my0 ad0 pf0 a10 a20 mx3 my3 ad3 pf3 a13 a23
    where slot 0 is Team A's Ranger, slot 3 is Team B's Ranger. Booleans
    are 0/1 ints. ``aim_delta`` is in radians (already scaled to ±π/4).

The viewer reads the header to construct an identical ``MatchConfig`` and
then drives a fresh ``Sim`` with the per-decision actions; the replay
relies on Phase-0 determinism rather than dumping full state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from eval.eval_phase3 import load_checkpoint
from train.ppo_recurrent.orchestration import make_env_fn


_LEARNER_SLOT = 0
_OPPONENT_SLOT = 3
_AIM_DELTA_LIMIT = float(3.141592653589793 / 4.0)


def _action_to_fields(action_arr) -> list[float]:
    """Convert a raw policy action vector to (mx, my, aim_delta_rad, pf, a1, a2).

    Mirrors ``Phase3RangerEnv._action_to_dict`` but returns the *radians*
    aim_delta the sim actually sees (already scaled by π/4)."""
    import numpy as np
    arr = np.asarray(action_arr, dtype=np.float32).reshape(6)
    mx = float(np.clip(arr[0], -1.0, 1.0))
    my = float(np.clip(arr[1], -1.0, 1.0))
    ad = float(np.clip(arr[2], -1.0, 1.0)) * _AIM_DELTA_LIMIT
    pf = int(np.clip(arr[3], 0.0, 1.0) >= 0.5)
    a1 = int(np.clip(arr[4], 0.0, 1.0) >= 0.5)
    a2 = int(np.clip(arr[5], 0.0, 1.0) >= 0.5)
    return [mx, my, ad, float(pf), float(a1), float(a2)]


def _format_decision(tick: int, slot0: list[float], slot3: list[float]) -> str:
    fields = [f"{tick}"]
    for v in slot0 + slot3:
        # Compact but lossless: 7 sig figs is plenty for replay.
        fields.append(f"{v:.7g}")
    return " ".join(fields)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump a Phase-3 greedy eval episode for the viewer to replay"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of consecutive episodes to dump")
    args = parser.parse_args()

    model, ckpt_config = load_checkpoint(args.checkpoint)
    train_config = {
        "phase": int(ckpt_config.get("phase", 3)),
        "env": ckpt_config["env"],
    }
    env_fn, _env_meta, _seed_base = make_env_fn(train_config)

    sim_cfg = ckpt_config["env"].get("sim", {})
    mech = sim_cfg.get("mechanics", {})
    header_fields: dict[str, Any] = {
        "format": "xushi2-replay-v1",
        "seed": int(args.seed),
        "round_seconds": int(sim_cfg.get("round_length_seconds", 30)),
        "action_repeat": int(sim_cfg.get("action_repeat", 3)),
        "mech_dmg": int(mech.get("revolver_damage_centi_hp", 7500)),
        "mech_fcd": int(mech.get("revolver_fire_cooldown_ticks", 15)),
        "mech_hbr": float(mech.get("revolver_hitbox_radius", 0.75)),
        "mech_resp": int(mech.get("respawn_ticks", 240)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_decisions = 0
    with output_path.open("w", encoding="ascii") as f:
        # Header.
        header_str = " ".join(f"{k}={v}" for k, v in header_fields.items())
        f.write(header_str + "\n")

        for ep in range(int(args.episodes)):
            env = env_fn()
            try:
                obs, info = env.reset(seed=int(args.seed) + ep)
                h = model.init_hidden(batch_size=1)
                done = False
                tick = int(info.get("tick", 0))
                while not done:
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

    print(f"[dump_replay] wrote {n_decisions} decisions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
