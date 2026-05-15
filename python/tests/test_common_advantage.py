from __future__ import annotations

import torch

from train.common_advantage import compute_gae_core


def test_common_gae_value_per_agent_layout() -> None:
    rewards = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    values = torch.zeros_like(rewards)
    dones = torch.zeros(1, 2)
    last_value = torch.zeros(1, 2)
    last_done = torch.zeros(1)

    advantages, returns = compute_gae_core(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        last_done=last_done,
        gamma=0.0,
        gae_lambda=0.0,
    )

    expected = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_common_gae_centralized_mode_uses_agent_mask_mean_reduction() -> None:
    rewards = torch.tensor([[[1.0, 4.0], [10.0, 40.0], [100.0, 400.0]]])
    values = torch.zeros(1, 2)
    dones = torch.zeros(1, 2)
    last_value = torch.zeros(1)
    last_done = torch.zeros(1)
    mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])

    advantages, _ = compute_gae_core(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        last_done=last_done,
        gamma=0.0,
        gae_lambda=0.0,
        agent_mask=mask,
        reduce_agents="mean",
    )

    torch.testing.assert_close(advantages, torch.tensor([[1.0, 40.0]]))


def test_common_gae_terminal_vs_truncated_boundary_semantics() -> None:
    rewards = torch.tensor([[0.0, 0.0]])
    values = torch.tensor([[0.0, 0.0]])
    dones = torch.tensor([[1.0, 0.0]])

    terminal_adv, _ = compute_gae_core(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=torch.tensor([5.0]),
        last_done=torch.tensor([0.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    truncated_adv, _ = compute_gae_core(
        rewards=rewards,
        values=values,
        dones=torch.tensor([[0.0, 0.0]]),
        last_value=torch.tensor([5.0]),
        last_done=torch.tensor([0.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    # done at t=0 prevents bootstrapping/recursion from t=1 back into t=0.
    torch.testing.assert_close(terminal_adv, torch.tensor([[0.0, 5.0]]))
    torch.testing.assert_close(truncated_adv, torch.tensor([[5.0, 5.0]]))
