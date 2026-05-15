from __future__ import annotations

import os
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch

from train.mappo_model import MappoActorCritic, MappoEvalStats, _eval_outcome_counts
from xushi2.vector_env import make_xushi_vector_env


def evaluate_mappo(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
    *,
    num_envs: int | None = None,
    backend: str = "sync",
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

    vec_env = make_xushi_vector_env(
        [env_fn for _ in range(num_envs)],
        critic_obs_dim=cfg.critic_obs_dim,
        seed_base=int(seed),
        auto_reset=True,
        backend=backend,
    )
    try:
        device = next(model.parameters()).device
        obs_np, _critic_obs, _infos = vec_env.reset(seed=int(seed))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        h = model.init_hidden(num_envs * cfg.n_agents).view(
            num_envs, cfg.n_agents, cfg.gru_hidden
        )
        ep_rewards = np.zeros(num_envs, dtype=np.float32)

        while len(rewards) < episodes:
            flat_obs = obs.reshape(num_envs * cfg.n_agents, cfg.obs_dim)
            flat_h = h.reshape(num_envs * cfg.n_agents, cfg.gru_hidden)
            with torch.no_grad():
                action, h_next = model.greedy_action(flat_obs, flat_h)
            action_3d = action.view(num_envs, cfg.n_agents, cfg.action_dim)
            action_np = action_3d.cpu().numpy()
            next_obs_np, reward_np, term, trunc, _critic, infos = vec_env.step(action_np)

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
                rewards.append(float(ep_rewards[i]))
                ep_rewards[i] = 0.0
                h[i] = 0.0
            obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    finally:
        vec_env.close()
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
