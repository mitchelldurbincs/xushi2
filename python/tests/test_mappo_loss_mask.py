from __future__ import annotations

import pytest
import torch
import yaml

from train.mappo import MappoConfig, MappoRollout, make_mappo_config
from train.mappo_advantage import compute_gae
from _paths import config_path


def _cfg(
    agent_loss_mask: tuple[float, ...] = (),
    *,
    n_agents: int = 3,
    value_per_agent: bool = False,
) -> MappoConfig:
    return MappoConfig(
        num_envs=1,
        n_agents=n_agents,
        rollout_len=2,
        obs_dim=4,
        critic_obs_dim=5,
        action_dim=6,
        continuous_action_dim=3,
        binary_action_dim=3,
        embed_dim=8,
        gru_hidden=8,
        head_hidden=8,
        action_log_std_init=-1.0,
        gamma=0.0,
        gae_lambda=0.0,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.001,
        max_grad_norm=0.5,
        learning_rate=1.0e-4,
        num_epochs=1,
        minibatch_size=1,
        agent_loss_mask=agent_loss_mask,
        value_per_agent=value_per_agent,
    )


def test_mappo_rollout_gae_defaults_to_all_agents() -> None:
    cfg = _cfg()
    rollout = MappoRollout(cfg)
    rewards = torch.tensor([1.0, 10.0, 100.0])
    rollout.reward[0, :, 0] = rewards
    rollout.reward[0, :, 1] = rewards

    compute_gae(rollout, cfg)

    assert rollout.advantages[0].tolist() == pytest.approx([37.0, 37.0])


def test_mappo_rollout_gae_uses_agent_loss_mask_for_reward_average() -> None:
    cfg = _cfg(agent_loss_mask=(1.0, 0.0, 0.0))
    rollout = MappoRollout(cfg)
    rewards = torch.tensor([1.0, 10.0, 100.0])
    rollout.reward[0, :, 0] = rewards
    rollout.reward[0, :, 1] = rewards

    compute_gae(rollout, cfg)

    assert rollout.advantages[0].tolist() == pytest.approx([1.0, 1.0])


def test_mappo_rollout_gae_accepts_dynamic_env_loss_masks() -> None:
    cfg = _cfg()
    rollout = MappoRollout(cfg)
    rewards = torch.tensor([1.0, 10.0, 100.0])
    rollout.reward[0, :, 0] = rewards
    rollout.reward[0, :, 1] = rewards
    rollout.agent_loss_mask[0, :, 0] = torch.tensor([1.0, 0.0, 0.0])
    rollout.agent_loss_mask[0, :, 1] = torch.tensor([0.0, 1.0, 0.0])

    compute_gae(rollout, cfg)

    assert rollout.advantages[0].tolist() == pytest.approx([1.0, 10.0])


def test_mappo_rollout_per_agent_gae_keeps_opposing_team_rewards_separate() -> None:
    cfg = _cfg(n_agents=6, value_per_agent=True)
    rollout = MappoRollout(cfg)
    rewards = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    rollout.reward[0, :, 0] = rewards
    rollout.reward[0, :, 1] = rewards

    compute_gae(rollout, cfg)

    assert rollout.advantages.shape == (1, 6, 2)
    assert torch.allclose(rollout.advantages[0, :3, :], torch.ones(3, 2))
    assert torch.allclose(rollout.advantages[0, 3:, :], -torch.ones(3, 2))


def test_make_mappo_config_validates_agent_loss_mask_shape() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"), encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    config["ppo"] = dict(config["ppo"])
    config["ppo"]["agent_loss_mask"] = [1.0, 0.0]

    with pytest.raises(ValueError, match="agent_loss_mask length"):
        make_mappo_config(config)


def test_make_mappo_config_requires_one_active_agent() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"), encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    config["ppo"] = dict(config["ppo"])
    config["ppo"]["agent_loss_mask"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="at least one active agent"):
        make_mappo_config(config)


def test_mappo_rollout_gae_rejects_negative_dynamic_env_loss_mask() -> None:
    cfg = _cfg()
    rollout = MappoRollout(cfg)
    rollout.agent_loss_mask[0, 0, 0] = -1.0

    with pytest.raises(AssertionError, match="non-negative"):
        compute_gae(rollout, cfg)


def test_mappo_rollout_gae_requires_one_active_agent_per_env_step() -> None:
    cfg = _cfg()
    rollout = MappoRollout(cfg)
    rollout.agent_loss_mask[0, :, 0] = 0.0

    with pytest.raises(AssertionError, match="at least one active agent"):
        compute_gae(rollout, cfg)
