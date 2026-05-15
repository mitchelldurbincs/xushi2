from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_LOSSES_PATH = Path(__file__).resolve().parents[1] / "train" / "ppo_recurrent" / "losses.py"
_SPEC = importlib.util.spec_from_file_location("ppo_losses", _LOSSES_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LOSSES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOSSES)
_masked_mean = _LOSSES._masked_mean
compute_ppo_loss = _LOSSES.compute_ppo_loss


def _old_loss_reference(
    *,
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantage: torch.Tensor,
    value: torch.Tensor,
    old_value: torch.Tensor,
    return_: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_ratio: float,
    value_clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
    entropy: torch.Tensor,
    return_mean: float,
    return_std: float,
    value_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    adv_mean = _masked_mean(advantage, valid_mask)
    adv_var = _masked_mean((advantage - adv_mean) ** 2, valid_mask)
    norm_adv = (advantage - adv_mean) / adv_var.clamp(min=1e-8).sqrt()

    ratio = (new_logprob - old_logprob).exp()
    pg1 = ratio * norm_adv
    pg2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * norm_adv
    policy_loss = _masked_mean(-torch.min(pg1, pg2), valid_mask)

    value_n = (value - return_mean) / return_std
    old_value_n = (old_value - return_mean) / return_std
    return_n = (return_ - return_mean) / return_std
    value_clipped_n = old_value_n + torch.clamp(value_n - old_value_n, -value_clip_ratio, value_clip_ratio)
    vl_unclipped = (value_n - return_n) ** 2
    vl_clipped = (value_clipped_n - return_n) ** 2
    vm = valid_mask if value_mask is None else value_mask
    value_loss = _masked_mean(0.5 * torch.max(vl_unclipped, vl_clipped), vm)

    entropy_mean = _masked_mean(entropy, valid_mask)
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    approx_kl = _masked_mean(old_logprob - new_logprob, valid_mask)
    clip_fraction = _masked_mean(((ratio - 1.0).abs() > clip_ratio).float(), valid_mask)
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "total_loss": total_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def test_compute_ppo_loss_matches_reference_single_agent_mask() -> None:
    shape = (2, 4)
    new_logprob = torch.tensor([[0.1, -0.2, 0.3, 0.0], [0.2, -0.1, 0.15, -0.25]])
    old_logprob = torch.tensor([[0.0, -0.1, 0.2, -0.05], [0.1, -0.2, 0.25, -0.2]])
    advantage = torch.tensor([[1.0, -0.5, 0.3, -0.1], [0.7, 0.2, -0.4, 0.6]])
    value = torch.tensor([[0.3, 0.1, 0.2, 0.4], [0.0, -0.2, 0.5, 0.1]])
    old_value = torch.tensor([[0.2, 0.2, 0.1, 0.3], [0.1, -0.1, 0.6, 0.2]])
    return_ = torch.tensor([[0.4, -0.1, 0.0, 0.5], [0.1, -0.3, 0.4, 0.2]])
    entropy = torch.tensor([[0.8, 0.7, 0.75, 0.72], [0.9, 0.65, 0.7, 0.68]])
    valid_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0]])
    assert new_logprob.shape == shape

    kwargs = dict(
        new_logprob=new_logprob,
        old_logprob=old_logprob,
        advantage=advantage,
        value=value,
        old_value=old_value,
        return_=return_,
        valid_mask=valid_mask,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        entropy=entropy,
        return_mean=0.1,
        return_std=0.7,
    )
    ref = _old_loss_reference(**kwargs)
    out = compute_ppo_loss(**kwargs)

    for k, v in ref.items():
        torch.testing.assert_close(getattr(out, k), v)


def test_compute_ppo_loss_matches_reference_multi_agent_masks() -> None:
    new_logprob = torch.tensor([[[0.1, 0.0], [0.2, -0.1]], [[-0.2, 0.3], [0.05, 0.1]]])
    old_logprob = torch.tensor([[[0.0, -0.1], [0.1, -0.2]], [[-0.1, 0.1], [0.0, 0.2]]])
    advantage = torch.tensor([[[1.0, 0.5], [0.0, -0.4]], [[-0.3, 0.7], [0.2, -0.1]]])
    value = torch.tensor([[[0.2, 0.1], [0.3, -0.1]], [[0.0, 0.4], [0.5, 0.2]]])
    old_value = torch.tensor([[[0.1, 0.0], [0.4, -0.2]], [[-0.1, 0.3], [0.6, 0.3]]])
    return_ = torch.tensor([[[0.3, 0.0], [0.1, -0.2]], [[-0.2, 0.5], [0.4, 0.1]]])
    entropy = torch.tensor([[[0.8, 0.75], [0.7, 0.65]], [[0.9, 0.85], [0.72, 0.69]]])
    valid_mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
    value_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    kwargs = dict(
        new_logprob=new_logprob,
        old_logprob=old_logprob,
        advantage=advantage,
        value=value,
        old_value=old_value,
        return_=return_,
        valid_mask=valid_mask,
        clip_ratio=0.15,
        value_clip_ratio=0.1,
        value_coef=1.0,
        entropy_coef=0.02,
        entropy=entropy,
        return_mean=0.0,
        return_std=1.0,
        value_mask=value_mask,
    )
    ref = _old_loss_reference(**kwargs)
    out = compute_ppo_loss(**kwargs)

    for k, v in ref.items():
        torch.testing.assert_close(getattr(out, k), v)
