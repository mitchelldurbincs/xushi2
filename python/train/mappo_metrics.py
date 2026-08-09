from __future__ import annotations

import torch

from train.mappo_model import _OWN_POSITION_SLICE
from train.losses import _masked_mean
from xushi2.multi_enemy_obs import entity_obs_self_position
from xushi2.obs_manifest import actor_field_slice


def rollout_metrics(cfg, rollout, *, model=None) -> dict[str, float]:
    """Summarize a rollout into scalar metrics.

    ``model`` is optional and only needed for ``fire_valid_fraction``, which
    requires the policy's invalid-fire mask. Callers without a model (unit
    tests, offline analysis) simply get that key omitted.
    """
    reward = rollout.reward
    advantages = rollout.advantages
    returns = rollout.returns
    action = rollout.action
    agent_mask = rollout.agent_loss_mask.expand_as(reward)
    move_mag = torch.linalg.vector_norm(action[:, :, :, 0:2], dim=-1)
    cont = action[:, :, :, : cfg.continuous_action_dim]
    binary_start = cfg.continuous_action_dim
    binary_end = binary_start + cfg.binary_action_dim
    binary = action[:, :, :, binary_start:binary_end]
    target = action[:, :, :, binary_end:] if cfg.target_action_dim > 0 else None

    self_on_point_slice = actor_field_slice("self_on_point")
    if cfg.obs_encoder in ("entity_attention", "entity_attention_grid"):
        obs_np = rollout.actor_obs.detach().cpu().numpy()
        own_pos_np = entity_obs_self_position(obs_np)
        own_pos = torch.as_tensor(
            own_pos_np, dtype=rollout.actor_obs.dtype, device=rollout.actor_obs.device
        )
        self_on_point = torch.zeros_like(own_pos[..., :1])
    else:
        own_pos = rollout.actor_obs[:, :, :, _OWN_POSITION_SLICE]
        self_on_point = rollout.actor_obs[:, :, :, self_on_point_slice]
    distance_to_objective = torch.linalg.vector_norm(own_pos, dim=-1)

    cont_mask = agent_mask.unsqueeze(-1).expand_as(cont)
    binary_mask = agent_mask.unsqueeze(-1).expand_as(binary) if binary.numel() > 0 else None
    target_mask = (
        agent_mask.unsqueeze(-1).expand_as(target)
        if (target is not None and target.numel() > 0)
        else None
    )
    self_on_point_mask = agent_mask.unsqueeze(-1).expand_as(self_on_point)

    with torch.no_grad():
        reward_mean = _masked_mean(reward, agent_mask)
        reward_active = reward[agent_mask > 0.0]
        cont_mean = _masked_mean(cont, cont_mask)
        tensor_metrics: dict[str, torch.Tensor] = {
            "active_agent_fraction": agent_mask.mean(),
            "rollout_reward_mean": reward_mean,
            "rollout_reward_std": _masked_mean((reward - reward_mean) ** 2, agent_mask).sqrt(),
            "rollout_reward_min": reward_active.min(),
            "rollout_reward_max": reward_active.max(),
            "advantage_mean": advantages.mean(),
            "advantage_std": advantages.std(unbiased=False),
            "advantage_min": advantages.min(),
            "advantage_max": advantages.max(),
            "return_mean": returns.mean(),
            "return_std": returns.std(unbiased=False),
            "action_move_mag_mean": _masked_mean(move_mag, agent_mask),
            "action_cont_mean": cont_mean,
            "action_cont_std": _masked_mean((cont - cont_mean) ** 2, cont_mask).sqrt(),
            "mean_distance_to_objective": _masked_mean(distance_to_objective, agent_mask),
            "self_on_point_fraction": _masked_mean(self_on_point, self_on_point_mask),
        }
        if binary_mask is not None:
            tensor_metrics["action_binary_mean"] = _masked_mean(binary, binary_mask)
        if target_mask is not None:
            tensor_metrics["action_target_slot_mean"] = _masked_mean(target, target_mask)
        if cfg.mask_fire_when_no_visible_enemy and model is not None:
            valid = model.fire_valid_mask(rollout.actor_obs.reshape(-1, cfg.obs_dim))
            if valid is not None:
                tensor_metrics["fire_valid_fraction"] = _masked_mean(
                    valid.to(agent_mask.dtype), agent_mask.reshape(-1)
                )

    # Stack + single .item() so we incur one host sync instead of N.
    keys = list(tensor_metrics.keys())
    stacked = torch.stack([tensor_metrics[k] for k in keys]).cpu().tolist()
    out: dict[str, float] = {k: float(v) for k, v in zip(keys, stacked, strict=False)}
    if "action_binary_mean" not in out:
        out["action_binary_mean"] = 0.0

    # Env-reported info metrics are already host-side floats accumulated over
    # the rollout; average them by the number of samples that contributed.
    samples = float(rollout.info_metrics.get("info_metric_samples", 0.0))
    if samples > 0.0:
        for key, value in rollout.info_metrics.items():
            if key == "info_metric_samples":
                continue
            out[f"rollout_{key}_mean"] = float(value) / samples
    return out
