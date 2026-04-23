"""Hyperparameter dataclass for the recurrent PPO trainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Hyperparameters for :class:`PPOTrainer`."""

    num_envs: int
    rollout_len: int
    obs_dim: int
    action_dim: int
    embed_dim: int
    gru_hidden: int
    head_hidden: int
    action_log_std_init: float
    use_recurrence: bool
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_clip_ratio: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    num_epochs: int
    minibatch_size: int
    lr_schedule: str = "constant"
    lr_final_ratio: float = 1.0
    warmup_updates: int = 0
    value_normalization: bool = True
