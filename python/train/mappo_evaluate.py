from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch

from train.mappo_model import MappoActorCritic, MappoEvalStats, _eval_outcome_counts


def evaluate_mappo(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
) -> MappoEvalStats:
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    final_ticks: list[int] = []
    team_a_scores: list[float] = []
    team_b_scores: list[float] = []
    team_a_kills: list[int] = []
    team_b_kills: list[int] = []
    wins = losses = draws = terminated_count = truncated_count = 0
    for i in range(int(episodes)):
        env = env_fn()
        try:
            obs, _info = env.reset(seed=int(seed) + i)
            h = model.init_hidden(model.cfg.n_agents)
            done = term = trunc = False
            ep_reward = 0.0
            info = {}
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, h = model.greedy_action(obs_t, h)
                obs, reward, term, trunc, info = env.step(action.cpu().numpy())
                ep_reward += float(np.mean(reward))
                done = bool(term or trunc)
            rewards.append(ep_reward)
            won, lost, drew = _eval_outcome_counts(
                winner=str(info.get("winner", "")),
                learner_team=str(info.get("learner_team", "")),
                truncated=bool(trunc),
            )
            wins += won
            losses += lost
            draws += drew
            terminated_count += int(bool(term))
            truncated_count += int(bool(trunc))
            final_ticks.append(int(info.get("tick", 0)))
            team_a_scores.append(float(info.get("team_a_score", 0.0)))
            team_b_scores.append(float(info.get("team_b_score", 0.0)))
            team_a_kills.append(int(info.get("team_a_kills", 0)))
            team_b_kills.append(int(info.get("team_b_kills", 0)))
        finally:
            env.close()
    if was_training:
        model.train()
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
        mean_team_a_kills=float(np.mean(team_a_kills)) if team_a_kills else 0.0,
        mean_team_b_kills=float(np.mean(team_b_kills)) if team_b_kills else 0.0,
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
    }
