"""Per-tick state dump for a cap_duel checkpoint, single episode.

Stage A of GOAL_INSTRUCTIONS.md. Reuses the same checkpoint/env construction
path as ``scripts/replay_dump/rollout.py`` so that the rollout matches what
training and eval see. Emits a JSON file with one entry per decision plus a
top-level summary that includes ``kill_then_hold_ratio`` and
``accidental_ratio`` — the metrics the user inspects to answer the
subjective gate question.

Run greedy and stochastic once each:

    py -3.13 scripts/inspect_cap_duel_rollout.py --mode greedy \\
      --output runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_greedy.json
    py -3.13 scripts/inspect_cap_duel_rollout.py --mode stochastic \\
      --output runs/phase4_mappo_cap_duel_selfplay_v1/mappo/diagnostics/inspect_stochastic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_dump.rollout import load_mappo_checkpoint  # noqa: E402
from train.checkpoint_runtime import checkpoint_runtime  # noqa: E402

_DEFAULT_CHECKPOINT = Path(
    "runs/phase4_mappo_cap_duel_selfplay_v1/mappo/ckpt_final.pt"
)
_DEFAULT_CONFIG = Path(
    "../experiments/configs/phase4/probe/phase4_mappo_cap_duel_selfplay_v1.yaml"
)
_DEFAULT_SEED = 3519994490


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Source YAML (informational; embedded checkpoint config is authoritative).",
    )
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--mode", choices=("greedy", "stochastic"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--episodes",
        type=int,
        default=10,
        help=(
            "Number of episodes to run; per-episode seed is seed+i (matches the "
            "eval loop pattern in scripts/replay_dump/rollout.py)."
        ),
    )
    return p.parse_args()


def _select_action(model, obs_t: torch.Tensor, hidden, *, mode: str):
    with torch.no_grad():
        if mode == "stochastic":
            action_t, _logprob, hidden = model.sample_action(obs_t, hidden)
        else:
            action_t, hidden = model.greedy_action(obs_t, hidden)
    return action_t.cpu().numpy(), hidden


def _attribute_score_events(
    per_tick: list[dict[str, Any]], recontest_delay: int
) -> dict[str, Any]:
    score_event_indices = [
        i for i, e in enumerate(per_tick) if e["score_event_this_step"]
    ]
    kill_then_hold = 0
    displace_then_hold = 0
    accidental = 0
    for i in score_event_indices:
        window_start = max(0, i - recontest_delay)
        window = per_tick[window_start:i]
        any_kill_in_window = any(e["kill_this_step"] for e in window)
        any_hit_in_window = any(e["hit_this_step"] for e in window)
        enemy_dead_at_score = not per_tick[i]["enemy_alive"]
        if enemy_dead_at_score or any_kill_in_window:
            kill_then_hold += 1
        elif any_hit_in_window:
            displace_then_hold += 1
        else:
            accidental += 1
    total = len(score_event_indices)
    return {
        "total_score_events": total,
        "kill_then_hold": kill_then_hold,
        "displace_then_hold": displace_then_hold,
        "accidental": accidental,
        "kill_then_hold_ratio": (float(kill_then_hold) / total) if total else 0.0,
        "displace_then_hold_ratio": (
            float(displace_then_hold) / total if total else 0.0
        ),
        "accidental_ratio": (float(accidental) / total) if total else 0.0,
    }


def _first_kills_with_followup_scores(
    per_tick: list[dict[str, Any]], recontest_delay: int, max_show: int = 3
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(per_tick):
        if not entry["kill_this_step"]:
            continue
        followups = [
            j
            for j in range(i + 1, min(len(per_tick), i + 1 + 4 * recontest_delay))
            if per_tick[j]["score_event_this_step"]
        ]
        out.append(
            {
                "kill_step_idx": i,
                "kill_tick": entry["tick"],
                "score_step_indices_in_window": followups,
            }
        )
        if len(out) >= max_show:
            break
    return out


def _run_episode(
    model, env, *, seed: int, mode: str, recontest_delay: int
) -> dict[str, Any]:
    obs, info = env.reset(seed=int(seed))
    hidden = model.init_hidden(model.cfg.n_agents)
    prev_kills = int(info.get("cap_duel_kills", 0))
    prev_enemy_kills = int(info.get("team_b_kills", 0))
    prev_hits = int(info.get("cap_duel_hits", 0))
    prev_score_events = int(info.get("cap_duel_score_events", 0))

    per_tick: list[dict[str, Any]] = []
    done = False
    step_idx = 0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        action_np, hidden = _select_action(model, obs_t, hidden, mode=mode)
        obs, _reward, term, trunc, info = env.step(action_np)

        kills = int(info.get("cap_duel_kills", prev_kills))
        enemy_kills = int(info.get("team_b_kills", prev_enemy_kills))
        hits = int(info.get("cap_duel_hits", prev_hits))
        score_events = int(info.get("cap_duel_score_events", prev_score_events))
        enemy_alive = bool(info.get("cap_duel_enemy_alive", True))

        entry = {
            "step_idx": step_idx,
            "tick": int(info.get("tick", step_idx + 1)),
            "match_type": str(info.get("match_type", "anchor")),
            "self_pos": list(info.get("cap_duel_self_pos", [0.0, 0.0])),
            "enemy_pos": list(info.get("cap_duel_enemy_pos", [0.0, 0.0])),
            "self_hp": int(info.get("cap_duel_self_hp", 0)),
            "enemy_hp": int(info.get("cap_duel_enemy_hp", 0)),
            "self_on_point": bool(info.get("cap_duel_self_on_point", False)),
            "enemy_on_point": bool(info.get("cap_duel_enemy_on_point", False)),
            "enemy_alive": enemy_alive,
            "enemy_off_point_decisions": int(
                info.get("cap_duel_enemy_off_point_decisions", 0)
            ),
            "self_score_ready_after_step": bool(
                info.get("cap_duel_self_score_ready", False)
            ),
            "score_event_this_step": (score_events - prev_score_events) >= 1,
            "kill_this_step": (kills - prev_kills) >= 1,
            "enemy_killed_self_this_step": (enemy_kills - prev_enemy_kills) >= 1,
            "hit_this_step": (hits - prev_hits) >= 1,
            "cap_duel_score_events_total": score_events,
            "cap_duel_score_ticks_total": int(
                info.get("cap_duel_score_ticks", 0)
            ),
            "cap_duel_enemy_score_ticks_total": int(
                info.get("cap_duel_enemy_score_ticks", 0)
            ),
            "team_b_score_total": float(info.get("team_b_score", 0.0)),
            "team_b_kills_total": int(info.get("team_b_kills", 0)),
            "cap_duel_kills_total": kills,
            "cap_duel_hits_total": hits,
            "cap_duel_fires_total": int(info.get("cap_duel_fires", 0)),
            "cap_duel_misses_total": int(info.get("cap_duel_misses", 0)),
            "action": [float(x) for x in action_np[0].tolist()],
        }
        per_tick.append(entry)

        prev_kills = kills
        prev_enemy_kills = enemy_kills
        prev_hits = hits
        prev_score_events = score_events
        step_idx += 1
        done = bool(term or trunc)

    n_steps = len(per_tick)
    last = per_tick[-1] if per_tick else {}
    reported_total_score_events = int(last.get("cap_duel_score_events_total", 0))
    summed_score_events = sum(1 for e in per_tick if e["score_event_this_step"])
    reported_total_kills = int(last.get("cap_duel_kills_total", 0))
    summed_kills = sum(1 for e in per_tick if e["kill_this_step"])
    if summed_score_events != reported_total_score_events:
        raise RuntimeError(
            f"seed={seed} score event reconciliation failed: "
            f"summed={summed_score_events} reported={reported_total_score_events}"
        )
    if summed_kills != reported_total_kills:
        raise RuntimeError(
            f"seed={seed} kill reconciliation failed: "
            f"summed={summed_kills} reported={reported_total_kills}"
        )

    forbidden_score_ticks = sum(
        1
        for e in per_tick
        if e["score_event_this_step"]
        and e["self_on_point"]
        and e["enemy_on_point"]
        and e["enemy_alive"]
    )

    attribution = _attribute_score_events(per_tick, recontest_delay)
    first_kills = _first_kills_with_followup_scores(per_tick, recontest_delay)

    self_on_point_steps = sum(1 for e in per_tick if e["self_on_point"])
    enemy_alive_steps = sum(1 for e in per_tick if e["enemy_alive"])
    fire_steps = sum(1 for e in per_tick if e["action"][3] >= 0.5)
    team_a_score_ticks = int(last.get("cap_duel_score_ticks_total", 0))
    team_b_score_ticks = int(last.get("cap_duel_enemy_score_ticks_total", 0))
    score_ticks_to_clear = 0  # filled in by caller from config; placeholder for episode summary
    winner = (
        "A"
        if team_a_score_ticks >= team_b_score_ticks and team_a_score_ticks > 0
        else "B"
        if team_b_score_ticks > team_a_score_ticks
        else "Neutral"
    )

    episode_summary = {
        "seed": int(seed),
        "steps": n_steps,
        "match_type": str(last.get("match_type", "")),
        "winner": winner,
        "team_a_score_ticks": team_a_score_ticks,
        "team_b_score_ticks": team_b_score_ticks,
        "team_a_kills": reported_total_kills,
        "team_b_kills": int(last.get("team_b_kills_total", 0)),
        "team_a_hits": int(last.get("cap_duel_hits_total", 0)),
        "team_a_fires": int(last.get("cap_duel_fires_total", 0)),
        "team_a_misses": int(last.get("cap_duel_misses_total", 0)),
        "self_on_point_step_fraction": (
            float(self_on_point_steps) / n_steps if n_steps else 0.0
        ),
        "fire_step_fraction": (
            float(fire_steps) / n_steps if n_steps else 0.0
        ),
        "enemy_alive_step_fraction": (
            float(enemy_alive_steps) / n_steps if n_steps else 0.0
        ),
        "score_ticks_with_self_on_point_and_enemy_alive_on_point": forbidden_score_ticks,
        "first_kills_with_followups": first_kills,
        **attribution,
    }
    return {"summary": episode_summary, "per_tick": per_tick}


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model, ckpt_config = load_mappo_checkpoint(ckpt_path)
    ckpt_runtime = checkpoint_runtime(ckpt_config)
    if ckpt_runtime.env_cfg.get("mini_game") != "cap_duel":
        raise ValueError(
            "inspector requires a cap_duel checkpoint; "
            f"got mini_game={ckpt_runtime.env_cfg.get('mini_game')!r}"
        )
    runtime = ckpt_runtime.runtime
    if runtime.env_fn is None:
        raise ValueError("runtime is missing env_fn")
    env = runtime.env_fn()

    mini_cfg = dict(ckpt_runtime.env_cfg.get("mini_game_config", {}))
    recontest_delay = int(mini_cfg.get("enemy_recontest_delay", 12))
    score_ticks_to_clear = int(mini_cfg.get("score_ticks_to_clear", 12))
    episode_decisions = int(mini_cfg.get("episode_decisions", 96))

    episodes: list[dict[str, Any]] = []
    for i in range(int(args.episodes)):
        episode = _run_episode(
            model,
            env,
            seed=int(args.seed) + i,
            mode=args.mode,
            recontest_delay=recontest_delay,
        )
        episodes.append(episode)

    # Aggregate across episodes.
    total_score_events = sum(e["summary"]["total_score_events"] for e in episodes)
    total_kill_then_hold = sum(e["summary"]["kill_then_hold"] for e in episodes)
    total_displace_then_hold = sum(
        e["summary"]["displace_then_hold"] for e in episodes
    )
    total_accidental = sum(e["summary"]["accidental"] for e in episodes)
    total_kills = sum(e["summary"]["team_a_kills"] for e in episodes)
    total_hits = sum(e["summary"]["team_a_hits"] for e in episodes)
    total_fires = sum(e["summary"]["team_a_fires"] for e in episodes)
    wins_a = sum(1 for e in episodes if e["summary"]["winner"] == "A")
    wins_b = sum(1 for e in episodes if e["summary"]["winner"] == "B")
    draws = sum(1 for e in episodes if e["summary"]["winner"] == "Neutral")
    mean_score_a = (
        sum(e["summary"]["team_a_score_ticks"] for e in episodes) / len(episodes)
        if episodes
        else 0.0
    )
    mean_score_b = (
        sum(e["summary"]["team_b_score_ticks"] for e in episodes) / len(episodes)
        if episodes
        else 0.0
    )

    summary = {
        "mode": args.mode,
        "checkpoint": str(ckpt_path),
        "config": str(args.config),
        "base_seed": int(args.seed),
        "episodes": int(args.episodes),
        "episode_decisions_configured": episode_decisions,
        "score_ticks_to_clear": score_ticks_to_clear,
        "enemy_recontest_delay": recontest_delay,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "mean_team_a_score_ticks": mean_score_a,
        "mean_team_b_score_ticks": mean_score_b,
        "total_kills": total_kills,
        "total_hits": total_hits,
        "total_fires": total_fires,
        "total_score_events": total_score_events,
        "kill_then_hold": total_kill_then_hold,
        "displace_then_hold": total_displace_then_hold,
        "accidental": total_accidental,
        "kill_then_hold_ratio": (
            float(total_kill_then_hold) / total_score_events
            if total_score_events
            else 0.0
        ),
        "displace_then_hold_ratio": (
            float(total_displace_then_hold) / total_score_events
            if total_score_events
            else 0.0
        ),
        "accidental_ratio": (
            float(total_accidental) / total_score_events
            if total_score_events
            else 0.0
        ),
    }
    return {"summary": summary, "episodes": episodes}


def main() -> None:
    args = _parse_args()
    result = inspect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    s = result["summary"]
    print(
        f"mode={s['mode']} episodes={s['episodes']} "
        f"wins_a={s['wins_a']} wins_b={s['wins_b']} draws={s['draws']}"
    )
    print(
        f"  mean_score_a={s['mean_team_a_score_ticks']:.2f} "
        f"mean_score_b={s['mean_team_b_score_ticks']:.2f}"
    )
    print(
        f"  score_events={s['total_score_events']} "
        f"kill_then_hold={s['kill_then_hold']} "
        f"displace_then_hold={s['displace_then_hold']} "
        f"accidental={s['accidental']}"
    )
    print(
        f"  kill_then_hold_ratio={s['kill_then_hold_ratio']:.3f} "
        f"displace_then_hold_ratio={s['displace_then_hold_ratio']:.3f} "
        f"accidental_ratio={s['accidental_ratio']:.3f}"
    )
    print(
        f"  total_kills={s['total_kills']} total_hits={s['total_hits']} "
        f"total_fires={s['total_fires']}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
