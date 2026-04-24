"""Greedy evaluation helper for trained recurrent policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from train.models import ActorCritic


@dataclass(frozen=True)
class EvaluationStats:
    mean_reward: float
    episodes: int
    wins: int
    losses: int
    draws: int
    terminated: int
    truncated: int
    mean_final_tick: float
    mean_team_a_score: float
    mean_team_b_score: float
    mean_team_a_kills: float
    mean_team_b_kills: float


def evaluate_policy(
    model: ActorCritic,
    env_fn: Callable[[], gym.Env],
    num_episodes: int,
    seed: int,
) -> float:
    """Mean episodic reward over greedy rollouts."""
    return evaluate_policy_stats(model, env_fn, num_episodes, seed).mean_reward


def evaluate_policy_stats(
    model: ActorCritic,
    env_fn: Callable[[], gym.Env],
    num_episodes: int,
    seed: int,
) -> EvaluationStats:
    """Mean episodic reward over ``num_episodes`` greedy rollouts.

    Greedy = ``tanh(action_mean)`` without sampling. We carry hidden state
    across ticks within an episode and re-zero at each episode boundary.
    """
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    final_ticks: list[int] = []
    team_a_scores: list[float] = []
    team_b_scores: list[float] = []
    team_a_kills: list[int] = []
    team_b_kills: list[int] = []
    wins = 0
    losses = 0
    draws = 0
    terminated_count = 0
    truncated_count = 0
    for i in range(int(num_episodes)):
        env = env_fn()
        try:
            obs, info = env.reset(seed=int(seed) + i)
            h = model.init_hidden(batch_size=1)
            ep_reward = 0.0
            done = False
            term = False
            trunc = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
                with torch.no_grad():
                    action_t, h = model.greedy_action(obs_t, h)
                action = action_t.squeeze(0).cpu().numpy()
                obs, r, term, trunc, info = env.step(action)
                ep_reward += float(r)
                done = bool(term or trunc)
            rewards.append(ep_reward)

            winner = str(info.get("winner", ""))
            learner_team = str(info.get("learner_team", ""))
            if winner in ("A", "B") and learner_team in ("A", "B"):
                if winner == learner_team:
                    wins += 1
                else:
                    losses += 1
            elif winner == "Neutral" or trunc:
                draws += 1

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
    episodes = len(rewards)
    return EvaluationStats(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        episodes=episodes,
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
