from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from train.mappo_model import (
    MappoActorCritic,
    MappoEvalStats,
    _eval_outcome_counts,
    mode_aux_targets,
    target_selection_policy_metrics,
)
from xushi2.obs_manifest import actor_field_slice
from xushi2.vector_env import make_xushi_vector_env


def _empty_combat_totals() -> dict[str, Any]:
    return {
        "fire_commands": 0,
        "visible_fire_commands": 0,
        "damage_hits": 0,
        "damage_centi_hp": 0,
        "aim_error_sum": 0.0,
        "aim_error_count": 0,
        "target_counts": {},
        "contested_majority_fire_commands": 0,
        "contested_majority_damage_hits": 0,
        "contested_majority_damage_centi_hp": 0,
    }


def _merge_combat_metrics(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst["fire_commands"] += int(src.get("fire_commands", 0))
    dst["visible_fire_commands"] += int(src.get("visible_fire_commands", 0))
    dst["damage_hits"] += int(src.get("damage_hits", 0))
    dst["damage_centi_hp"] += int(src.get("damage_centi_hp", 0))
    dst["aim_error_sum"] += float(src.get("aim_error_sum", 0.0))
    dst["aim_error_count"] += int(src.get("aim_error_count", 0))
    dst["contested_majority_fire_commands"] += int(src.get("contested_majority_fire_commands", 0))
    dst["contested_majority_damage_hits"] += int(src.get("contested_majority_damage_hits", 0))
    dst["contested_majority_damage_centi_hp"] += int(
        src.get("contested_majority_damage_centi_hp", 0)
    )
    counts = dst["target_counts"]
    for raw_target, raw_count in dict(src.get("target_counts", {})).items():
        target = int(raw_target)
        counts[target] = int(counts.get(target, 0)) + int(raw_count)


def _combat_summary(totals: dict[str, Any]) -> dict[str, float]:
    fire_commands = int(totals["fire_commands"])
    visible_fire_commands = int(totals["visible_fire_commands"])
    aim_error_count = int(totals["aim_error_count"])
    target_counts = dict(totals["target_counts"])
    target_total = sum(int(v) for v in target_counts.values())
    entropy = 0.0
    if target_total > 0:
        for count in target_counts.values():
            p = float(count) / float(target_total)
            if p > 0.0:
                entropy -= p * math.log(p)
    return {
        "hit_fire": (
            float(totals["damage_hits"]) / float(fire_commands) if fire_commands else 0.0
        ),
        "visible_fire_rate": (
            float(visible_fire_commands) / float(fire_commands) if fire_commands else 0.0
        ),
        "aim_error_rad": (
            float(totals["aim_error_sum"]) / float(aim_error_count)
            if aim_error_count
            else 0.0
        ),
        "target_entropy": entropy,
        "damage_per_fire": (
            float(totals["damage_centi_hp"]) / float(fire_commands) if fire_commands else 0.0
        ),
    }


_OBJECTIVE_METRIC_KEYS = (
    "uncontested_on_point_seconds_a",
    "uncontested_on_point_seconds_b",
    "majority_on_point_seconds_a",
    "majority_on_point_seconds_b",
    "alive_edge_no_score_seconds_a",
    "alive_edge_no_score_seconds_b",
    "cap_progress_gain_ticks",
    "cap_progress_loss_ticks",
    "contested_majority_flag_a",
    "contested_majority_flag_b",
    "on_point_nearest_enemy_distance_sum_a",
    "on_point_nearest_enemy_distance_count_a",
    "on_point_enemy_los_count_a",
    "on_point_total_count_a",
    "on_point_nearest_enemy_distance_sum_b",
    "on_point_nearest_enemy_distance_count_b",
    "on_point_enemy_los_count_b",
    "on_point_total_count_b",
)


def _empty_objective_totals() -> dict[str, float]:
    out = {key: 0.0 for key in _OBJECTIVE_METRIC_KEYS}
    out["first_team_a_alive_edge_to_score_seconds"] = -1.0
    out["majority_to_uncontested_within_n_num_a"] = 0.0
    out["majority_to_uncontested_within_n_den_a"] = 0.0
    out["majority_to_uncontested_within_n_num_b"] = 0.0
    out["majority_to_uncontested_within_n_den_b"] = 0.0
    out["contested_majority_windows_a"] = 0.0
    out["contested_majority_windows_b"] = 0.0
    return out


def _merge_objective_metrics(dst: dict[str, float], src: dict[str, Any]) -> None:
    for key in _OBJECTIVE_METRIC_KEYS:
        dst[key] = float(dst.get(key, 0.0)) + float(src.get(key, 0.0))
    edge_to_score = float(src.get("first_team_a_alive_edge_to_score_seconds", -1.0))
    if edge_to_score >= 0.0:
        dst["first_team_a_alive_edge_to_score_seconds"] = edge_to_score


def evaluate_mappo(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
    *,
    num_envs: int | None = None,
    backend: str = "sync",
    objective_timing_seconds: tuple[float, float] | None = None,
    respawn_ticks: int | None = None,
    stochastic: bool = False,
) -> MappoEvalStats:
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")
    cfg = model.cfg
    if num_envs is None:
        num_envs = min(episodes, max(1, os.cpu_count() or 1))
    num_envs = max(1, min(int(num_envs), episodes))

    was_training = model.training
    model.eval()

    rewards: list[float] = []
    final_ticks: list[int] = []
    team_a_scores: list[float] = []
    team_b_scores: list[float] = []
    team_a_kills: list[int] = []
    team_b_kills: list[int] = []
    wins = losses = draws = terminated_count = truncated_count = 0
    combat_totals = {"A": _empty_combat_totals(), "B": _empty_combat_totals()}
    focus_totals = {
        "A": {"entropy_sum": 0.0, "same_sum": 0.0, "count": 0},
        "B": {"entropy_sum": 0.0, "same_sum": 0.0, "count": 0},
    }
    mode_total = 0
    mode_correct = 0
    p_combat_sum = 0.0
    intentional_fire_count = 0
    objective_focus_count = 0
    self_on_point_slice = actor_field_slice("self_on_point")
    completed_objective_totals: list[dict[str, float]] = []

    vec_env = make_xushi_vector_env(
        [env_fn for _ in range(num_envs)],
        critic_obs_dim=(
            cfg.critic_obs_dim * cfg.n_agents if cfg.value_per_agent else cfg.critic_obs_dim
        ),
        seed_base=int(seed),
        auto_reset=True,
        backend=backend,
    )
    try:
        device = next(model.parameters()).device
        vec_env.set_majority_on_point_alpha(0.0)
        vec_env.set_uncontested_on_point_alpha(0.0)
        if objective_timing_seconds is not None:
            vec_env.set_objective_timing_seconds(
                float(objective_timing_seconds[0]),
                float(objective_timing_seconds[1]),
            )
        if respawn_ticks is not None:
            # Applied on the reset below: the respawn setter is reset-time
            # only (no live-sim setter).
            vec_env.set_respawn_ticks(int(respawn_ticks))
        obs_np, _critic_obs, _infos = vec_env.reset(seed=int(seed))
        if objective_timing_seconds is not None:
            eval_unlock_seconds = float(objective_timing_seconds[0])
            eval_capture_seconds = float(objective_timing_seconds[1])
        elif _infos:
            eval_unlock_seconds = float(_infos[0].get("objective_unlock_seconds", 0.0))
            eval_capture_seconds = float(_infos[0].get("objective_capture_seconds", 0.0))
        else:
            eval_unlock_seconds = 0.0
            eval_capture_seconds = 0.0
        if respawn_ticks is not None:
            eval_respawn_ticks = int(respawn_ticks)
        elif _infos:
            eval_respawn_ticks = int(_infos[0].get("respawn_ticks", 0))
        else:
            eval_respawn_ticks = 0
        # Greedy on a fixed map is one deterministic trajectory regardless of
        # seed; stochastic sampling (seeded here for reproducibility) is what
        # makes the `episodes` count carry independent evidence.
        action_generator: torch.Generator | None = None
        if stochastic:
            action_generator = torch.Generator(device=device)
            action_generator.manual_seed(int(seed))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        h = model.init_hidden(num_envs * cfg.n_agents).view(
            num_envs, cfg.n_agents, cfg.gru_hidden
        )
        ep_rewards = np.zeros(num_envs, dtype=np.float32)
        ep_objective_totals = [_empty_objective_totals() for _ in range(num_envs)]
        window_n_ticks = int(5 * 16)
        contested_windows = {
            "A": [[] for _ in range(num_envs)],
            "B": [[] for _ in range(num_envs)],
        }
        majority_run_ticks = {"A": [0] * num_envs, "B": [0] * num_envs}
        post_majority_ticks = {"A": [-1] * num_envs, "B": [-1] * num_envs}

        while len(rewards) < episodes:
            flat_obs = obs.reshape(num_envs * cfg.n_agents, cfg.obs_dim)
            flat_h = h.reshape(num_envs * cfg.n_agents, cfg.gru_hidden)
            with torch.no_grad():
                if cfg.target_selection_dim > 0:
                    features, _ = model.actor_head_features(flat_obs, flat_h)
                    _mean, _logits, target_selection_logits = model.policy_heads_from_features(
                        flat_obs, features
                    )
                    for env_i in range(num_envs):
                        metrics = target_selection_policy_metrics(
                            target_selection_logits[
                                env_i * cfg.n_agents : (env_i + 1) * cfg.n_agents
                            ],
                            flat_obs[env_i * cfg.n_agents : (env_i + 1) * cfg.n_agents],
                            cfg,
                        )
                        team = "A"
                        if env_i < len(_infos):
                            team = str(_infos[env_i].get("learner_team", "A"))
                        if team in focus_totals:
                            focus_totals[team]["entropy_sum"] += float(
                                metrics["target_selection_policy_entropy"].item()
                            )
                            focus_totals[team]["same_sum"] += float(
                                metrics[
                                    "target_selection_policy_same_target_fraction"
                                ].item()
                            )
                            focus_totals[team]["count"] += 1
                if cfg.mode_gated_combat:
                    features, _ = model.actor_head_features(flat_obs, flat_h)
                    mode_logits = model.mode_logits_from_features(features)
                    p_combat = model.combat_probability(mode_logits)
                    if p_combat is not None:
                        labels, valid = mode_aux_targets(flat_obs, cfg)
                        valid_count = int(valid.sum().item())
                        if valid_count > 0:
                            mode_total += valid_count
                            mode_correct += int(
                                (
                                    mode_logits[valid].argmax(dim=-1) == labels[valid]
                                ).sum().item()
                            )
                            p_combat_sum += float(p_combat[valid].sum().item())
                if stochastic:
                    action, _logprob, h_next = model.sample_action(
                        flat_obs, flat_h, generator=action_generator
                    )
                else:
                    action, h_next = model.greedy_action(flat_obs, flat_h)
                if cfg.mode_gated_combat:
                    mode_logits = model.mode_logits_from_features(
                        model.actor_head_features(flat_obs, flat_h)[0]
                    )
                    p_combat = model.combat_probability(mode_logits)
                    if p_combat is not None:
                        fire_start = cfg.continuous_action_dim
                        fire_taken = action[:, fire_start] > 0.5
                        objective_focus = p_combat < 0.5
                        if cfg.obs_encoder == "flat":
                            on_point = flat_obs[:, self_on_point_slice].squeeze(-1) > 0.5
                        else:
                            on_point = torch.zeros_like(objective_focus)
                        intentional_fire_count += int(((p_combat > 0.5) & fire_taken).sum().item())
                        objective_focus_count += int((objective_focus & on_point).sum().item())
            action_3d = action.view(num_envs, cfg.n_agents, cfg.action_dim)
            action_np = action_3d.cpu().numpy()
            next_obs_np, reward_np, term, trunc, _critic, infos = vec_env.step(action_np)
            for env_i, info_i in enumerate(infos):
                combat_metrics = info_i.get("combat_metrics")
                if not isinstance(combat_metrics, dict):
                    combat_metrics = {}
                for team in ("A", "B"):
                    team_metrics = combat_metrics.get(team)
                    if isinstance(team_metrics, dict):
                        _merge_combat_metrics(combat_totals[team], team_metrics)
                objective_metrics = info_i.get("objective_metrics")
                if isinstance(objective_metrics, dict):
                    _merge_objective_metrics(ep_objective_totals[env_i], objective_metrics)
                    for team in ("A", "B"):
                        flag = float(objective_metrics.get(f"contested_majority_flag_{team.lower()}", 0.0))
                        if flag > 0.5:
                            majority_run_ticks[team][env_i] += 1
                        else:
                            if majority_run_ticks[team][env_i] > 0:
                                secs = float(majority_run_ticks[team][env_i]) / 16.0
                                contested_windows[team][env_i].append(secs)
                                ep_objective_totals[env_i][f"contested_majority_windows_{team.lower()}"] += 1.0
                                post_majority_ticks[team][env_i] = 0
                                majority_run_ticks[team][env_i] = 0
                            elif post_majority_ticks[team][env_i] >= 0:
                                post_majority_ticks[team][env_i] += 1
                                uncontested = float(
                                    objective_metrics.get(
                                        f"uncontested_on_point_seconds_{team.lower()}", 0.0
                                    )
                                ) > 0.0
                                if uncontested:
                                    ep_objective_totals[env_i][
                                        f"majority_to_uncontested_within_n_num_{team.lower()}"
                                    ] += 1.0
                                    ep_objective_totals[env_i][
                                        f"majority_to_uncontested_within_n_den_{team.lower()}"
                                    ] += 1.0
                                    post_majority_ticks[team][env_i] = -1
                                elif post_majority_ticks[team][env_i] > window_n_ticks:
                                    ep_objective_totals[env_i][
                                        f"majority_to_uncontested_within_n_den_{team.lower()}"
                                    ] += 1.0
                                    post_majority_ticks[team][env_i] = -1

            ep_rewards += reward_np.mean(axis=1)
            h = h_next.view(num_envs, cfg.n_agents, cfg.gru_hidden)
            done_np = np.logical_or(term, trunc)
            for i in range(num_envs):
                if not done_np[i]:
                    continue
                if len(rewards) >= episodes:
                    break
                info_i = infos[i]
                final_info = info_i.get("final_info", info_i)
                won, lost, drew = _eval_outcome_counts(
                    winner=str(final_info.get("winner", "")),
                    learner_team=str(final_info.get("learner_team", "")),
                    truncated=bool(trunc[i]),
                )
                wins += won
                losses += lost
                draws += drew
                terminated_count += int(bool(term[i]))
                truncated_count += int(bool(trunc[i]))
                final_ticks.append(int(final_info.get("tick", 0)))
                team_a_scores.append(float(final_info.get("team_a_score", 0.0)))
                team_b_scores.append(float(final_info.get("team_b_score", 0.0)))
                team_a_kills.append(int(final_info.get("team_a_kills", 0)))
                team_b_kills.append(int(final_info.get("team_b_kills", 0)))
                completed_objective_totals.append(dict(ep_objective_totals[i]))
                for team in ("A", "B"):
                    if majority_run_ticks[team][i] > 0:
                        contested_windows[team][i].append(float(majority_run_ticks[team][i]) / 16.0)
                        ep_objective_totals[i][f"contested_majority_windows_{team.lower()}"] += 1.0
                        majority_run_ticks[team][i] = 0
                ep_objective_totals[i] = _empty_objective_totals()
                rewards.append(float(ep_rewards[i]))
                ep_rewards[i] = 0.0
                h[i] = 0.0
            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    finally:
        vec_env.close()
        if was_training:
            model.train()

    combat_a = _combat_summary(combat_totals["A"])
    combat_b = _combat_summary(combat_totals["B"])
    focus_a_count = max(1, int(focus_totals["A"]["count"]))
    focus_b_count = max(1, int(focus_totals["B"]["count"]))
    mode_den = max(1, mode_total)
    objective_count = max(1, len(completed_objective_totals))

    def _objective_mean(key: str) -> float:
        return float(sum(row.get(key, 0.0) for row in completed_objective_totals)) / float(
            objective_count
        )

    edge_to_score_values = [
        float(row.get("first_team_a_alive_edge_to_score_seconds", -1.0))
        for row in completed_objective_totals
        if float(row.get("first_team_a_alive_edge_to_score_seconds", -1.0)) >= 0.0
    ]
    all_windows_a = [w for env_w in contested_windows["A"] for w in env_w]
    all_windows_b = [w for env_w in contested_windows["B"] for w in env_w]
    cm_a_fire = float(combat_totals["A"].get("contested_majority_fire_commands", 0))
    cm_b_fire = float(combat_totals["B"].get("contested_majority_fire_commands", 0))
    cm_a_hits = float(combat_totals["A"].get("contested_majority_damage_hits", 0))
    cm_b_hits = float(combat_totals["B"].get("contested_majority_damage_hits", 0))
    cm_a_dmg = float(combat_totals["A"].get("contested_majority_damage_centi_hp", 0))
    cm_b_dmg = float(combat_totals["B"].get("contested_majority_damage_centi_hp", 0))
    return MappoEvalStats(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        episodes=len(rewards),
        wins=wins,
        losses=losses,
        draws=draws,
        terminated=terminated_count,
        truncated=truncated_count,
        mean_final_tick=float(np.mean(final_ticks)) if final_ticks else 0.0,
        mean_team_a_score=float(np.mean(team_a_scores)) if team_a_scores else 0.0,
        mean_team_b_score=float(np.mean(team_b_scores)) if team_b_scores else 0.0,
        std_team_a_score=float(np.std(team_a_scores)) if team_a_scores else 0.0,
        std_team_b_score=float(np.std(team_b_scores)) if team_b_scores else 0.0,
        respawn_ticks=eval_respawn_ticks,
        mean_team_a_kills=float(np.mean(team_a_kills)) if team_a_kills else 0.0,
        mean_team_b_kills=float(np.mean(team_b_kills)) if team_b_kills else 0.0,
        team_a_hit_fire=float(combat_a["hit_fire"]),
        team_b_hit_fire=float(combat_b["hit_fire"]),
        team_a_visible_fire_rate=float(combat_a["visible_fire_rate"]),
        team_b_visible_fire_rate=float(combat_b["visible_fire_rate"]),
        team_a_aim_error_rad=float(combat_a["aim_error_rad"]),
        team_b_aim_error_rad=float(combat_b["aim_error_rad"]),
        team_a_target_entropy=float(combat_a["target_entropy"]),
        team_b_target_entropy=float(combat_b["target_entropy"]),
        team_a_same_target_fraction=float(focus_totals["A"]["same_sum"]) / focus_a_count,
        team_b_same_target_fraction=float(focus_totals["B"]["same_sum"]) / focus_b_count,
        team_a_target_selection_entropy=float(focus_totals["A"]["entropy_sum"])
        / focus_a_count,
        team_b_target_selection_entropy=float(focus_totals["B"]["entropy_sum"])
        / focus_b_count,
        team_a_damage_per_fire=float(combat_a["damage_per_fire"]),
        team_b_damage_per_fire=float(combat_b["damage_per_fire"]),
        mean_p_combat=float(p_combat_sum) / float(mode_den),
        mode_accuracy=float(mode_correct) / float(mode_den),
        intentional_fire_fraction=float(intentional_fire_count) / float(mode_den),
        objective_focus_fraction=float(objective_focus_count) / float(mode_den),
        mean_uncontested_on_point_seconds_a=_objective_mean(
            "uncontested_on_point_seconds_a"
        ),
        mean_uncontested_on_point_seconds_b=_objective_mean(
            "uncontested_on_point_seconds_b"
        ),
        mean_majority_on_point_seconds_a=_objective_mean("majority_on_point_seconds_a"),
        mean_majority_on_point_seconds_b=_objective_mean("majority_on_point_seconds_b"),
        mean_alive_edge_no_score_seconds_a=_objective_mean(
            "alive_edge_no_score_seconds_a"
        ),
        mean_alive_edge_no_score_seconds_b=_objective_mean(
            "alive_edge_no_score_seconds_b"
        ),
        mean_cap_progress_gain_ticks=_objective_mean("cap_progress_gain_ticks"),
        mean_cap_progress_loss_ticks=_objective_mean("cap_progress_loss_ticks"),
        mean_first_team_a_alive_edge_to_score_seconds=(
            float(np.mean(edge_to_score_values)) if edge_to_score_values else -1.0
        ),
        majority_to_uncontested_within_n_fraction_a=(
            _objective_mean("majority_to_uncontested_within_n_num_a")
            / max(1e-9, _objective_mean("majority_to_uncontested_within_n_den_a"))
        ),
        majority_to_uncontested_within_n_fraction_b=(
            _objective_mean("majority_to_uncontested_within_n_num_b")
            / max(1e-9, _objective_mean("majority_to_uncontested_within_n_den_b"))
        ),
        contested_majority_windows_per_episode_a=_objective_mean("contested_majority_windows_a"),
        contested_majority_windows_per_episode_b=_objective_mean("contested_majority_windows_b"),
        contested_majority_window_mean_seconds_a=float(np.mean(all_windows_a)) if all_windows_a else 0.0,
        contested_majority_window_mean_seconds_b=float(np.mean(all_windows_b)) if all_windows_b else 0.0,
        contested_majority_window_p50_seconds_a=float(np.percentile(all_windows_a, 50.0)) if all_windows_a else 0.0,
        contested_majority_window_p50_seconds_b=float(np.percentile(all_windows_b, 50.0)) if all_windows_b else 0.0,
        contested_majority_window_p90_seconds_a=float(np.percentile(all_windows_a, 90.0)) if all_windows_a else 0.0,
        contested_majority_window_p90_seconds_b=float(np.percentile(all_windows_b, 90.0)) if all_windows_b else 0.0,
        on_point_nearest_enemy_distance_mean_a=(
            _objective_mean("on_point_nearest_enemy_distance_sum_a")
            / max(1e-9, _objective_mean("on_point_nearest_enemy_distance_count_a"))
        ),
        on_point_nearest_enemy_distance_mean_b=(
            _objective_mean("on_point_nearest_enemy_distance_sum_b")
            / max(1e-9, _objective_mean("on_point_nearest_enemy_distance_count_b"))
        ),
        on_point_enemy_los_fraction_a=(
            _objective_mean("on_point_enemy_los_count_a")
            / max(1e-9, _objective_mean("on_point_total_count_a"))
        ),
        on_point_enemy_los_fraction_b=(
            _objective_mean("on_point_enemy_los_count_b")
            / max(1e-9, _objective_mean("on_point_total_count_b"))
        ),
        contested_majority_hit_fire_a=(cm_a_hits / cm_a_fire) if cm_a_fire > 0 else 0.0,
        contested_majority_hit_fire_b=(cm_b_hits / cm_b_fire) if cm_b_fire > 0 else 0.0,
        contested_majority_damage_per_fire_a=(cm_a_dmg / cm_a_fire) if cm_a_fire > 0 else 0.0,
        contested_majority_damage_per_fire_b=(cm_b_dmg / cm_b_fire) if cm_b_fire > 0 else 0.0,
        objective_unlock_seconds=float(eval_unlock_seconds),
        objective_capture_seconds=float(eval_capture_seconds),
    )


def eval_stats_dict(stats: MappoEvalStats) -> dict[str, float | int]:
    episodes = max(1, int(stats.episodes))
    return {
        "episodes": int(stats.episodes),
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "win_rate": float(stats.wins) / float(episodes),
        "loss_rate": float(stats.losses) / float(episodes),
        "draw_rate": float(stats.draws) / float(episodes),
        "mean_reward": float(stats.mean_reward),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
        "mean_kills_a": float(stats.mean_team_a_kills),
        "mean_kills_b": float(stats.mean_team_b_kills),
        "mean_final_tick": float(stats.mean_final_tick),
        "terminated": int(stats.terminated),
        "truncated": int(stats.truncated),
        "team_a_hit_fire": float(stats.team_a_hit_fire),
        "team_b_hit_fire": float(stats.team_b_hit_fire),
        "team_a_visible_fire_rate": float(stats.team_a_visible_fire_rate),
        "team_b_visible_fire_rate": float(stats.team_b_visible_fire_rate),
        "team_a_aim_error_rad": float(stats.team_a_aim_error_rad),
        "team_b_aim_error_rad": float(stats.team_b_aim_error_rad),
        "team_a_target_entropy": float(stats.team_a_target_entropy),
        "team_b_target_entropy": float(stats.team_b_target_entropy),
        "team_a_same_target_fraction": float(stats.team_a_same_target_fraction),
        "team_b_same_target_fraction": float(stats.team_b_same_target_fraction),
        "team_a_target_selection_entropy": float(stats.team_a_target_selection_entropy),
        "team_b_target_selection_entropy": float(stats.team_b_target_selection_entropy),
        "team_a_damage_per_fire": float(stats.team_a_damage_per_fire),
        "team_b_damage_per_fire": float(stats.team_b_damage_per_fire),
        "mean_p_combat": float(stats.mean_p_combat),
        "mode_accuracy": float(stats.mode_accuracy),
        "intentional_fire_fraction": float(stats.intentional_fire_fraction),
        "objective_focus_fraction": float(stats.objective_focus_fraction),
        "mean_uncontested_on_point_seconds_a": float(
            stats.mean_uncontested_on_point_seconds_a
        ),
        "mean_uncontested_on_point_seconds_b": float(
            stats.mean_uncontested_on_point_seconds_b
        ),
        "mean_majority_on_point_seconds_a": float(stats.mean_majority_on_point_seconds_a),
        "mean_majority_on_point_seconds_b": float(stats.mean_majority_on_point_seconds_b),
        "mean_alive_edge_no_score_seconds_a": float(
            stats.mean_alive_edge_no_score_seconds_a
        ),
        "mean_alive_edge_no_score_seconds_b": float(
            stats.mean_alive_edge_no_score_seconds_b
        ),
        "mean_cap_progress_gain_ticks": float(stats.mean_cap_progress_gain_ticks),
        "mean_cap_progress_loss_ticks": float(stats.mean_cap_progress_loss_ticks),
        "mean_first_team_a_alive_edge_to_score_seconds": float(
            stats.mean_first_team_a_alive_edge_to_score_seconds
        ),
        "majority_to_uncontested_within_n_fraction_a": float(
            stats.majority_to_uncontested_within_n_fraction_a
        ),
        "majority_to_uncontested_within_n_fraction_b": float(
            stats.majority_to_uncontested_within_n_fraction_b
        ),
        "contested_majority_windows_per_episode_a": float(
            stats.contested_majority_windows_per_episode_a
        ),
        "contested_majority_windows_per_episode_b": float(
            stats.contested_majority_windows_per_episode_b
        ),
        "contested_majority_window_mean_seconds_a": float(
            stats.contested_majority_window_mean_seconds_a
        ),
        "contested_majority_window_mean_seconds_b": float(
            stats.contested_majority_window_mean_seconds_b
        ),
        "contested_majority_window_p50_seconds_a": float(
            stats.contested_majority_window_p50_seconds_a
        ),
        "contested_majority_window_p50_seconds_b": float(
            stats.contested_majority_window_p50_seconds_b
        ),
        "contested_majority_window_p90_seconds_a": float(
            stats.contested_majority_window_p90_seconds_a
        ),
        "contested_majority_window_p90_seconds_b": float(
            stats.contested_majority_window_p90_seconds_b
        ),
        "on_point_nearest_enemy_distance_mean_a": float(
            stats.on_point_nearest_enemy_distance_mean_a
        ),
        "on_point_nearest_enemy_distance_mean_b": float(
            stats.on_point_nearest_enemy_distance_mean_b
        ),
        "on_point_enemy_los_fraction_a": float(stats.on_point_enemy_los_fraction_a),
        "on_point_enemy_los_fraction_b": float(stats.on_point_enemy_los_fraction_b),
        "contested_majority_hit_fire_a": float(stats.contested_majority_hit_fire_a),
        "contested_majority_hit_fire_b": float(stats.contested_majority_hit_fire_b),
        "contested_majority_damage_per_fire_a": float(
            stats.contested_majority_damage_per_fire_a
        ),
        "contested_majority_damage_per_fire_b": float(
            stats.contested_majority_damage_per_fire_b
        ),
        "objective_unlock_seconds": float(stats.objective_unlock_seconds),
        "objective_capture_seconds": float(stats.objective_capture_seconds),
        "respawn_ticks": int(stats.respawn_ticks),
        "std_score_a": float(stats.std_team_a_score),
        "std_score_b": float(stats.std_team_b_score),
    }
