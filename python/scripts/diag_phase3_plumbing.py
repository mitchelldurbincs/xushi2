"""Phase-3 plumbing sanity probe.

Drives ``XushiEnv`` with fixed, hand-written actions against a noop opponent
and prints per-event diagnostics. Verifies that the objective state machine
(lock window, capture, score) is wired correctly end-to-end by driving the
agent with an optimal "sit on cap" policy.

Usage (from repo root, with Python 3.13):

    py -3.13 python/scripts/diag_phase3_plumbing.py
    py -3.13 python/scripts/diag_phase3_plumbing.py --round-length-seconds 240

The ``--round-length-seconds`` flag overrides the YAML default without
editing it. Default (the YAML's 30s) exposes the "win is mathematically
impossible" pathology; 240s exposes the intended design.

Scenarios, in order:

1. ``sit_on_cap`` — ignore the enemy, walk to normalized origin (0, 0)
                    where the cap lives, sit there. This is the optimal
                    strategy for a stationary opponent.
2. ``homing``    — head toward the enemy while alive (fire-and-kill),
                   head back to the cap while enemy is respawning.
3. ``forward``   — move_x=+1, fire=1, aim_delta=0. Sanity: pure motion.
4. ``still``     — no movement, no fire. Null baseline.

Event log format (one line per transition):
    [tick T] EVENT: field X=A -> X=B  (context)

So if plumbing is correct, ``sit_on_cap`` should print:
  - on_pt:   0 -> 1 somewhere in ticks 100-150
  - unlocked: 0 -> 1 at tick 449 (kObjectiveLockTicks - 1)
  - cap progress crossing 0.25, 0.50, 0.75, 1.0 between ticks 450 and ~690
  - ownership change: Neutral -> Us at ~tick 690
  - score first > 0 at ~tick 691
If cap_progress never rises in a 240s round, plumbing IS broken.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "python"))

from xushi2.env import XushiEnv
from xushi2.obs_manifest import actor_field_slice


_CONFIG_PATH = _REPO_ROOT / "experiments" / "configs" / "phase3_ranger_noop_probe.yaml"
_CAP_ARRIVE_THRESHOLD = 0.05  # normalized distance — well inside 0.12 radius


def load_cfg(round_length_override: int | None) -> tuple[dict, dict, str, str]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    env_cfg = cfg["env"]
    sim_cfg = dict(env_cfg["sim"])
    if round_length_override is not None:
        sim_cfg["round_length_seconds"] = int(round_length_override)
    return (
        sim_cfg,
        env_cfg.get("reward", {}),
        env_cfg["opponent_bot"],
        env_cfg.get("learner_team", "A"),
    )


def obs_field(obs: np.ndarray, name: str) -> np.ndarray:
    return obs[actor_field_slice(name)]


def make_action(
    *,
    move_x: float = 0.0,
    move_y: float = 0.0,
    aim_delta: float = 0.0,
    primary_fire: int = 0,
    ability_1: int = 0,
    ability_2: int = 0,
) -> dict[str, Any]:
    return {
        "move_x": float(move_x),
        "move_y": float(move_y),
        "aim_delta": float(aim_delta),
        "primary_fire": int(primary_fire),
        "ability_1": int(ability_1),
        "ability_2": int(ability_2),
    }


def _move_toward(delta_xy: np.ndarray, stop_inside: float) -> tuple[float, float]:
    """Unit step toward (self + delta_xy). Returns (move_x, move_y).

    If we're already within ``stop_inside`` of the target (i.e. ``|delta| < stop_inside``),
    hold still.
    """
    mag = float(np.hypot(delta_xy[0], delta_xy[1]))
    if mag < stop_inside:
        return 0.0, 0.0
    return (
        float(np.clip(delta_xy[0] / mag, -1.0, 1.0)),
        float(np.clip(delta_xy[1] / mag, -1.0, 1.0)),
    )


def _sit_on_cap_action(obs: np.ndarray) -> dict[str, Any]:
    """Walk to normalized origin (the cap) and sit. Ignore enemy."""
    own_pos = obs_field(obs, "own_position")
    # Target is origin; delta from self = -own_pos.
    move_x, move_y = _move_toward(-own_pos, _CAP_ARRIVE_THRESHOLD)
    return make_action(move_x=move_x, move_y=move_y, primary_fire=0)


def _homing_action(obs: np.ndarray) -> dict[str, Any]:
    """Chase enemy when alive, otherwise return to cap. Fire while alive."""
    rel = obs_field(obs, "enemy_relative_position")
    own_pos = obs_field(obs, "own_position")
    aim = obs_field(obs, "own_aim_unit")  # (sin θ, cos θ)
    enemy_alive = float(obs_field(obs, "enemy_alive")[0]) > 0.5

    if enemy_alive:
        delta = rel
    else:
        # Enemy respawning — head back to the cap instead of drifting.
        delta = -own_pos

    move_x, move_y = _move_toward(delta, _CAP_ARRIVE_THRESHOLD)

    aim_delta = 0.0
    if enemy_alive:
        rel_mag = float(np.hypot(rel[0], rel[1]))
        if rel_mag > 1e-6:
            theta_aim = math.atan2(float(aim[0]), float(aim[1]))
            theta_target = math.atan2(float(rel[1]), float(rel[0]))
            err = theta_target - theta_aim
            while err > math.pi:
                err -= 2 * math.pi
            while err < -math.pi:
                err += 2 * math.pi
            aim_delta = float(np.clip(err, -math.pi / 4, math.pi / 4))

    return make_action(
        move_x=move_x,
        move_y=move_y,
        aim_delta=aim_delta,
        primary_fire=1 if enemy_alive else 0,
    )


def _forward_action(_obs: np.ndarray) -> dict[str, Any]:
    return make_action(move_x=1.0, move_y=0.0, primary_fire=1)


def _still_action(_obs: np.ndarray) -> dict[str, Any]:
    return make_action()


SCENARIOS: dict[str, Callable[[np.ndarray], dict[str, Any]]] = {
    "sit_on_cap": _sit_on_cap_action,
    "homing":     _homing_action,
    "forward":    _forward_action,
    "still":      _still_action,
}


def _obs_on_pt(obs: np.ndarray) -> int:
    return int(float(obs_field(obs, "self_on_point")[0]) > 0.5)


def _obs_unlocked(obs: np.ndarray) -> int:
    return int(float(obs_field(obs, "objective_unlocked")[0]) > 0.5)


def _obs_cap_progress(obs: np.ndarray) -> float:
    return float(obs_field(obs, "cap_progress")[0])


def _obs_owner_onehot(obs: np.ndarray) -> np.ndarray:
    return obs_field(obs, "objective_owner_onehot").copy()


def _owner_label(onehot: np.ndarray) -> str:
    idx = int(np.argmax(onehot))
    return ("Neutral", "Us", "Them")[idx]


def run_scenario(name: str, policy, seed: int, decisions: int) -> None:
    sim_cfg, reward_cfg, opponent_bot, learner_team = load_cfg(ROUND_LENGTH_OVERRIDE)
    env = XushiEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )

    obs, info = env.reset(seed=seed)
    header = (
        f"\n=== scenario: {name}  seed=0x{seed:x}  "
        f"decisions={decisions}  round_length_s={sim_cfg['round_length_seconds']} ==="
    )
    print(header)
    print(f"  opponent={opponent_bot}  learner_team={learner_team}")

    own_pos = obs_field(obs, "own_position")
    print(
        f"  reset: tick={info['tick']} "
        f"ownpos=({own_pos[0]:+.2f},{own_pos[1]:+.2f}) "
        f"on_pt={_obs_on_pt(obs)} unlocked={_obs_unlocked(obs)} "
        f"cap={_obs_cap_progress(obs):.2f} owner={_owner_label(_obs_owner_onehot(obs))}"
    )

    # Transition-tracking state.
    prev_on_pt = _obs_on_pt(obs)
    prev_unlocked = _obs_unlocked(obs)
    prev_cap_bucket = int(_obs_cap_progress(obs) * 4)  # log crossings of 0.25/0.50/0.75/1.0
    prev_owner = _owner_label(_obs_owner_onehot(obs))
    prev_kills_a = int(info["team_a_kills"])
    prev_score_a = float(info["team_a_score"])
    first_score_logged = False
    total_reward = 0.0

    for step_idx in range(decisions):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        tick = int(info["tick"])
        own_pos = obs_field(obs, "own_position")
        on_pt = _obs_on_pt(obs)
        unlocked = _obs_unlocked(obs)
        cap = _obs_cap_progress(obs)
        owner = _owner_label(_obs_owner_onehot(obs))
        kills_a = int(info["team_a_kills"])
        score_a = float(info["team_a_score"])
        score_b = float(info["team_b_score"])

        if on_pt != prev_on_pt:
            print(
                f"  [tick {tick:4d}] on_pt {prev_on_pt}->{on_pt}  "
                f"(ownpos=({own_pos[0]:+.2f},{own_pos[1]:+.2f}))"
            )
            prev_on_pt = on_pt

        if unlocked != prev_unlocked:
            print(f"  [tick {tick:4d}] objective_unlocked {prev_unlocked}->{unlocked}")
            prev_unlocked = unlocked

        cap_bucket = int(cap * 4)
        if cap_bucket != prev_cap_bucket:
            print(
                f"  [tick {tick:4d}] cap_progress crossed {prev_cap_bucket*0.25:.2f} "
                f"-> now {cap:.2f}  (on_pt={on_pt}, unlocked={unlocked})"
            )
            prev_cap_bucket = cap_bucket

        if owner != prev_owner:
            print(f"  [tick {tick:4d}] owner {prev_owner} -> {owner}")
            prev_owner = owner

        if kills_a != prev_kills_a:
            print(f"  [tick {tick:4d}] kill #{kills_a} (team A)")
            prev_kills_a = kills_a

        if not first_score_logged and score_a > 0.0:
            print(
                f"  [tick {tick:4d}] score_a first > 0: {score_a:.2f} "
                f"(owner={owner})"
            )
            first_score_logged = True
        prev_score_a = score_a

        if terminated or truncated:
            break

    # Final snapshot.
    print(
        f"  final: tick={info['tick']} "
        f"terminated={terminated} truncated={truncated} winner={info['winner']} "
        f"score=A{score_a:.2f}/B{score_b:.2f} "
        f"kills=A{kills_a}/B{info['team_b_kills']} "
        f"total_reward={total_reward:+.3f}"
    )
    env.close()


ROUND_LENGTH_OVERRIDE: int | None = None


def main() -> None:
    global ROUND_LENGTH_OVERRIDE
    parser = argparse.ArgumentParser(description="Phase-3 plumbing sanity probe")
    parser.add_argument(
        "--round-length-seconds",
        type=int,
        default=None,
        help=(
            "Override round_length_seconds from the YAML (defaults to YAML value). "
            "Try 240 to confirm the objective state machine works end-to-end."
        ),
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the named scenario (one of: sit_on_cap, homing, forward, still).",
    )
    args = parser.parse_args()
    ROUND_LENGTH_OVERRIDE = args.round_length_seconds

    # action_repeat=3 sim ticks per decision at 30 Hz ⇒ 10 decisions per second.
    # Pick enough decisions to always reach the end-of-round truncation.
    seconds = args.round_length_seconds if args.round_length_seconds is not None else 30
    decisions = int(seconds * 30 / 3) + 20  # +20 safety margin past truncation

    seed = 0xD1CEDA7A
    scenarios = (
        {args.only: SCENARIOS[args.only]} if args.only else SCENARIOS
    )
    for name, policy in scenarios.items():
        run_scenario(name, policy, seed=seed, decisions=decisions)


if __name__ == "__main__":
    main()
