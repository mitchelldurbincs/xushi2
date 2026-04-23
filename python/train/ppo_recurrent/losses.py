"""Loss math helpers for the recurrent PPO trainer.

Kept as free functions (not methods on ``PPOTrainer``) so they can be
unit-tested in isolation and reused by evaluation/analysis code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import Normal

_ATANH_EPS = 1e-6
_LOG2 = math.log(2.0)


def _tanh_squashed_logprob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob of a tanh-squashed Gaussian evaluated at ``action``.

    Inverts ``action = tanh(u)`` via ``u = atanh(action)`` after clamping
    ``action`` into ``[-1 + eps, 1 - eps]`` to keep the inverse finite.
    Returns ``(logprob, pre_tanh_entropy)`` where ``pre_tanh_entropy`` is
    the summed entropy of the underlying Normal (we use this as a
    standard proxy for the full squashed-dist entropy).
    """
    action = action.clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
    # atanh(x) = 0.5 * (log1p(x) - log1p(-x))
    u = 0.5 * (torch.log1p(action) - torch.log1p(-action))
    std = log_std.exp()
    dist = Normal(mean, std)
    # Same tanh-correction formulation as models.sample_action.
    correction = 2.0 * (_LOG2 - u - F.softplus(-2.0 * u))
    logprob = dist.log_prob(u).sum(-1) - correction.sum(-1)
    entropy = dist.entropy().sum(-1)
    return logprob, entropy


def action_logprob_and_entropy(
    continuous_mean: torch.Tensor,
    continuous_log_std: torch.Tensor,
    binary_logits: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob / entropy for a mixed continuous+binary action tensor."""
    cont_dim = int(continuous_mean.shape[-1])
    binary_dim = int(binary_logits.shape[-1])

    logprob = action.new_zeros(action.shape[0])
    entropy = action.new_zeros(action.shape[0])

    if cont_dim > 0:
        cont_action = action[:, :cont_dim]
        cont_logprob, cont_entropy = _tanh_squashed_logprob(
            continuous_mean,
            continuous_log_std,
            cont_action,
        )
        logprob = logprob + cont_logprob
        entropy = entropy + cont_entropy

    if binary_dim > 0:
        binary_action = action[:, cont_dim : cont_dim + binary_dim]
        binary_dist = Bernoulli(logits=binary_logits)
        logprob = logprob + binary_dist.log_prob(binary_action).sum(-1)
        entropy = entropy + binary_dist.entropy().sum(-1)

    return logprob, entropy


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum ``values * mask`` and divide by ``mask.sum()`` (>=1 guard).

    Produces zero when the mask is entirely zero, which is the right
    behavior for empty padded batches.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom
