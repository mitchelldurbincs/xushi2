"""Greedy evaluation helper for trained recurrent policies."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from train.models import ActorCritic


def evaluate_policy(
    model: ActorCritic,
    env_fn: Callable[[], gym.Env],
    num_episodes: int,
    seed: int,
) -> float:
    """Mean episodic reward over ``num_episodes`` greedy rollouts.

    Greedy = ``tanh(action_mean)`` without sampling. We carry hidden state
    across ticks within an episode and re-zero at each episode boundary.
    """
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    for i in range(int(num_episodes)):
        env = env_fn()
        obs, _ = env.reset(seed=int(seed) + i)
        h = model.init_hidden(batch_size=1)
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                action_t, h = model.greedy_action(obs_t, h)
            action = action_t.squeeze(0).cpu().numpy()
            obs, r, term, trunc, _info = env.step(action)
            ep_reward += float(r)
            done = bool(term or trunc)
        rewards.append(ep_reward)
        env.close()
    if was_training:
        model.train()
    return float(np.mean(rewards)) if rewards else 0.0
