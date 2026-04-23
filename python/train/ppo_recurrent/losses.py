"""Loss math helpers for the recurrent PPO trainer.

Kept as free functions (not methods on ``PPOTrainer``) so they can be
unit-tested in isolation and reused by evaluation/analysis code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum ``values * mask`` and divide by ``mask.sum()`` (>=1 guard).

    Produces zero when the mask is entirely zero, which is the right
    behavior for empty padded batches.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom
