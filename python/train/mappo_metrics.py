from __future__ import annotations

import torch

from train.mappo_model import _OWN_POSITION_SLICE
from train.ppo_recurrent.losses import _masked_mean
from xushi2.entity_obs import entity_obs_self_position
from xushi2.obs_manifest import actor_field_slice


def rollout_metrics(cfg, rollout) -> dict[str, float]:
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
        own_pos = torch.as_tensor(own_pos_np, dtype=rollout.actor_obs.dtype, device=rollout.actor_obs.device)
        self_on_point = torch.zeros_like(own_pos[..., :1])
    else:
        own_pos = rollout.actor_obs[:, :, :, _OWN_POSITION_SLICE]
        self_on_point = rollout.actor_obs[:, :, :, self_on_point_slice]
    distance_to_objective = torch.linalg.vector_norm(own_pos, dim=-1)

    out = {
        "active_agent_fraction": float(agent_mask.mean().item()),
        "rollout_reward_mean": float(_masked_mean(reward, agent_mask).item()),
        "rollout_reward_std": float(_masked_mean((reward - _masked_mean(reward, agent_mask)) ** 2, agent_mask).sqrt().item()),
        "rollout_reward_min": float(reward[agent_mask > 0.0].min().item()),
        "rollout_reward_max": float(reward[agent_mask > 0.0].max().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "advantage_min": float(advantages.min().item()),
        "advantage_max": float(advantages.max().item()),
        "return_mean": float(returns.mean().item()),
        "return_std": float(returns.std(unbiased=False).item()),
        "action_move_mag_mean": float(_masked_mean(move_mag, agent_mask).item()),
        "action_cont_mean": float(_masked_mean(cont, agent_mask.unsqueeze(-1).expand_as(cont)).item()),
        "action_cont_std": float(
            _masked_mean(
                (cont - _masked_mean(cont, agent_mask.unsqueeze(-1).expand_as(cont))) ** 2,
                agent_mask.unsqueeze(-1).expand_as(cont),
            )
            .sqrt()
            .item()
        ),
        "mean_distance_to_objective": float(_masked_mean(distance_to_objective, agent_mask).item()),
        "self_on_point_fraction": float(
            _masked_mean(self_on_point, agent_mask.unsqueeze(-1).expand_as(self_on_point)).item()
        ),
    }
    out["action_binary_mean"] = (
        float(_masked_mean(binary, agent_mask.unsqueeze(-1).expand_as(binary)).item()) if binary.numel() > 0 else 0.0
    )
    if target is not None and target.numel() > 0:
        out["action_target_slot_mean"] = float(
            _masked_mean(target, agent_mask.unsqueeze(-1).expand_as(target)).item()
        )
    return out
