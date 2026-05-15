from __future__ import annotations

import pytest
import torch

from train.mappo import MappoConfig, MappoRollout
from train.mappo_metrics import rollout_metrics
from train.mappo_rollout import step_loss_mask
from train.mappo_update import action_logprob_entropy


def _cfg(*, target_action_dim: int = 0) -> MappoConfig:
    return MappoConfig(
        num_envs=2,
        n_agents=3,
        rollout_len=4,
        obs_dim=16,
        critic_obs_dim=5,
        action_dim=6 + (1 if target_action_dim > 0 else 0),
        continuous_action_dim=2,
        binary_action_dim=4,
        target_action_dim=target_action_dim,
        embed_dim=8,
        gru_hidden=8,
        head_hidden=8,
        action_log_std_init=-1.0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        learning_rate=1.0e-4,
        num_epochs=1,
        minibatch_size=1,
        agent_loss_mask=(1.0, 1.0, 0.0),
    )


def test_step_loss_mask_shape_and_fallback() -> None:
    cfg = _cfg()
    infos = [{}, {"loss_mask": [1.0, 0.0, 1.0]}]
    mask = step_loss_mask(cfg, infos)
    assert mask.shape == (cfg.num_envs, cfg.n_agents)
    assert torch.allclose(mask[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.allclose(mask[1], torch.tensor([1.0, 0.0, 0.0]))


def test_step_loss_mask_rejects_all_zero() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="at least one active agent"):
        step_loss_mask(cfg, [{"loss_mask": [0.0, 0.0, 0.0]}, {}])


def test_rollout_metrics_returns_expected_keys() -> None:
    cfg = _cfg()
    rollout = MappoRollout(cfg)
    rollout.reward.fill_(1.0)
    rollout.advantages.fill_(0.5)
    rollout.returns.fill_(1.5)
    metrics = rollout_metrics(cfg, rollout)
    assert "rollout_reward_mean" in metrics
    assert "action_binary_mean" in metrics
    assert isinstance(metrics["rollout_reward_mean"], float)


def test_action_logprob_entropy_shapes_without_target() -> None:
    cfg = _cfg(target_action_dim=0)
    batch = 5
    mean = torch.zeros(batch, cfg.continuous_action_dim)
    log_std = torch.zeros(batch, cfg.continuous_action_dim)
    binary_logits = torch.zeros(batch, cfg.binary_action_dim)
    action = torch.zeros(batch, cfg.action_dim)
    logp, ent = action_logprob_entropy(cfg, mean, log_std, binary_logits, None, action)
    assert logp.shape == (batch,)
    assert ent.shape == (batch,)
