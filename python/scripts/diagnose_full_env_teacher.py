"""Run the full-env scripted rehearsal teacher directly in Phase 4.

This diagnostic answers a narrow question before another supervised/PPO run:
can the teacher's own action stream hit, hold point, and avoid losing in the
same full-env distribution where the neural policy failed?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from train.full_env_rehearsal import scripted_objective_focus_fire_targets
from train.mappo_rollout_trainer import make_mappo_config
from train.runtime_specs import resolve_runtime_spec
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import ENTITY_TOKEN_DIM, MULTI_ENEMY_TOKEN_COUNT

_AIM_DELTA_LIMIT = float(np.pi / 4.0)
_SELF_TOKEN = 0
_FIRST_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 4
_POSITION = slice(8, 10)
_AIM = slice(12, 14)
_AUX = 17


def _actor_obs_teacher_action(obs: np.ndarray, cfg) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    cont, binary = scripted_objective_focus_fire_targets(obs_t, cfg)
    pieces = [cont, binary]
    if int(cfg.target_action_dim) > 0:
        pieces.append(torch.zeros(obs_t.shape[0], 1, dtype=obs_t.dtype))
    return torch.cat(pieces, dim=-1).cpu().numpy().astype(np.float32, copy=False)


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def _multi_enemy_obs_teacher_action(obs: np.ndarray, cfg) -> np.ndarray:
    if cfg.obs_encoder != "entity_attention_grid":
        raise ValueError("multi_enemy_visible teacher requires entity_attention_grid obs")
    flat = np.asarray(obs, dtype=np.float32).reshape(-1, cfg.obs_dim)
    token_width = MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = flat[:, :token_width].reshape(-1, MULTI_ENEMY_TOKEN_COUNT, ENTITY_TOKEN_DIM)
    mask = flat[:, token_width : token_width + MULTI_ENEMY_TOKEN_COUNT] > 0.5

    action = np.zeros((flat.shape[0], 6), dtype=np.float32)
    objective_rel = tokens[:, _OBJECTIVE_TOKEN, _POSITION]
    self_on_point = tokens[:, _SELF_TOKEN, _AUX] > 0.5
    obj_norm = np.linalg.norm(objective_rel, axis=1, keepdims=True)
    move = np.divide(
        objective_rel,
        np.maximum(obj_norm, 1.0e-6),
        out=np.zeros_like(objective_rel),
        where=obj_norm > 1.0e-6,
    )
    action[:, :2] = np.where(self_on_point[:, None], 0.0, move).clip(-1.0, 1.0)

    enemy_tokens = tokens[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN, :]
    enemy_mask = mask[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN]
    enemy_rel = enemy_tokens[:, :, _POSITION]
    enemy_dist = np.linalg.norm(enemy_rel, axis=2)
    masked_dist = np.where(enemy_mask, enemy_dist, np.inf)
    target_idx = np.argmin(masked_dist, axis=1)
    has_target = np.isfinite(masked_dist[np.arange(flat.shape[0]), target_idx])
    target_rel = enemy_rel[np.arange(flat.shape[0]), target_idx]
    # The direct diagnostic emits action aim_delta, whose simulator convention
    # matches the C++ bots: atan2(dy, dx). This is separate from supervised
    # label helpers that target actor-observation conventions.
    target_angle = np.arctan2(target_rel[:, 1], target_rel[:, 0])
    aim_unit = tokens[:, _SELF_TOKEN, _AIM]
    current_angle = np.arctan2(aim_unit[:, 0], aim_unit[:, 1])
    delta = np.clip(
        _wrap_angle_np(target_angle - current_angle),
        -_AIM_DELTA_LIMIT,
        _AIM_DELTA_LIMIT,
    )
    action[:, 2] = np.where(has_target, delta / _AIM_DELTA_LIMIT, 0.0)
    action[:, 3] = has_target.astype(np.float32)
    return action


def _cpp_bot_teacher_action(env, *, bot_name: str, learner_team: str) -> np.ndarray:
    if getattr(env, "_sim", None) is None:
        raise RuntimeError("env must be reset before cpp bot teacher action")
    slots = (0, 1, 2) if learner_team == "A" else (3, 4, 5)
    action = np.zeros((3, 6), dtype=np.float32)
    for row, slot in enumerate(slots):
        scripted = _cpp.scripted_bot_action(env._sim, slot, bot_name)
        move_sign = -1.0 if slot >= 3 else 1.0
        action[row] = np.array(
            [
                move_sign * scripted.move_x,
                move_sign * scripted.move_y,
                scripted.aim_delta / _AIM_DELTA_LIMIT,
                float(scripted.primary_fire),
                float(scripted.ability_1),
                float(scripted.ability_2),
            ],
            dtype=np.float32,
        )
    return action


def _empty_totals() -> dict[str, float]:
    return {
        "fire_commands": 0.0,
        "visible_fire_commands": 0.0,
        "damage_hits": 0.0,
        "damage_centi_hp": 0.0,
        "majority_on_point_seconds_a": 0.0,
        "uncontested_on_point_seconds_a": 0.0,
        "final_seconds": 0.0,
        "score_a": 0.0,
        "score_b": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "draws": 0.0,
    }


def run_teacher_diagnostic(
    config: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    teacher: str = "actor_obs_scripted",
    max_decisions: int | None = None,
) -> dict[str, Any]:
    cfg = make_mappo_config(config)
    runtime = resolve_runtime_spec(config)
    if runtime.env_fn is None:
        raise ValueError("config does not resolve to a runnable env_fn")
    if teacher == "actor_obs_scripted" and cfg.obs_encoder != "flat":
        raise ValueError("actor_obs_scripted diagnostic supports flat obs only")
    if teacher == "multi_enemy_visible" and cfg.obs_encoder != "entity_attention_grid":
        raise ValueError("multi_enemy_visible diagnostic supports entity_attention_grid obs only")

    totals = _empty_totals()
    episode_rows: list[dict[str, float | int | str]] = []
    for episode in range(int(episodes)):
        env = runtime.env_fn()
        try:
            obs, _info = env.reset(seed=int(seed) + episode)
            decisions = 0
            final_info: dict[str, Any] = {}
            episode_totals = _empty_totals()
            while True:
                if teacher == "actor_obs_scripted":
                    action = _actor_obs_teacher_action(obs, cfg)
                elif teacher == "multi_enemy_visible":
                    action = _multi_enemy_obs_teacher_action(obs, cfg)
                elif teacher.startswith("cpp_"):
                    action = _cpp_bot_teacher_action(
                        env,
                        bot_name=teacher.removeprefix("cpp_"),
                        learner_team=runtime.env.learner_team,
                    )
                else:
                    raise ValueError(f"unknown teacher {teacher!r}")
                obs, _reward, term, trunc, info = env.step(action)
                decisions += 1
                final_info = info
                combat = dict(info.get("combat_metrics", {})).get("A", {})
                objective = dict(info.get("objective_metrics", {}))
                for key in (
                    "fire_commands",
                    "visible_fire_commands",
                    "damage_hits",
                    "damage_centi_hp",
                ):
                    episode_totals[key] += float(combat.get(key, 0.0))
                for key in (
                    "majority_on_point_seconds_a",
                    "uncontested_on_point_seconds_a",
                ):
                    episode_totals[key] += float(objective.get(key, 0.0))
                if term or trunc or (max_decisions is not None and decisions >= int(max_decisions)):
                    break

            final_tick = float(final_info.get("tick", 0.0))
            final_seconds = final_tick / 30.0
            score_a = float(final_info.get("team_a_score", 0.0))
            score_b = float(final_info.get("team_b_score", 0.0))
            winner = str(final_info.get("winner", "Neutral"))
            episode_totals["final_seconds"] = final_seconds
            episode_totals["score_a"] = score_a
            episode_totals["score_b"] = score_b
            episode_totals["wins"] = 1.0 if winner == "A" else 0.0
            episode_totals["losses"] = 1.0 if winner == "B" else 0.0
            episode_totals["draws"] = 1.0 if winner == "Neutral" else 0.0
            for key, value in episode_totals.items():
                totals[key] += float(value)
            episode_rows.append(
                {
                    "episode": episode,
                    "seed": int(seed) + episode,
                    "decisions": decisions,
                    "winner": winner,
                    **episode_totals,
                }
            )
        finally:
            env.close()

    episodes_f = max(1.0, float(episodes))
    fire_commands = max(1.0, totals["fire_commands"])
    final_seconds = max(1.0, totals["final_seconds"])
    summary = {
        "episodes": int(episodes),
        "seed": int(seed),
        "teacher": str(teacher),
        "opponent_bot": runtime.env.opponent_kind,
        "metrics": {
            "team_a_hit_fire": totals["damage_hits"] / fire_commands,
            "team_a_visible_fire_rate": totals["visible_fire_commands"] / fire_commands,
            "team_a_damage_centi_hp_mean": totals["damage_centi_hp"] / episodes_f,
            "objective_on_point": totals["majority_on_point_seconds_a"] / final_seconds,
            "uncontested_on_point": totals["uncontested_on_point_seconds_a"] / final_seconds,
            "wins": totals["wins"],
            "losses": totals["losses"],
            "draws": totals["draws"],
            "mean_score_a": totals["score_a"] / episodes_f,
            "mean_score_b": totals["score_b"] / episodes_f,
        },
        "totals": totals,
        "episodes_detail": episode_rows,
    }
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--teacher",
        default="actor_obs_scripted",
        help="actor_obs_scripted, multi_enemy_visible, or cpp_<bot_name> such as cpp_basic",
    )
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    env_cfg = dict(config.get("env", {}))
    seed = int(args.seed if args.seed is not None else env_cfg.get("seed_base", 0))
    summary = run_teacher_diagnostic(
        config,
        episodes=int(args.episodes),
        seed=seed,
        teacher=str(args.teacher),
        max_decisions=args.max_decisions,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
