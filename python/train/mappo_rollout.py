from __future__ import annotations

import numpy as np
import torch

from train.device import resolve_device
from train.mappo_model import MappoConfig


def step_loss_mask(
    cfg: MappoConfig,
    infos: list[dict],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    dev = resolve_device(cfg.device if device is None else device)
    static = torch.as_tensor(cfg.agent_loss_mask, dtype=torch.float32, device=dev)
    masks = torch.zeros(cfg.num_envs, cfg.n_agents, dtype=torch.float32, device=dev)
    for env_idx, info in enumerate(infos):
        raw = info.get("loss_mask")
        if raw is None:
            final_info = info.get("final_info")
            if isinstance(final_info, dict):
                raw = final_info.get("loss_mask")
        if raw is None:
            masks[env_idx] = static
            continue
        mask = torch.as_tensor(raw, dtype=torch.float32, device=dev).reshape(-1)
        if mask.numel() != cfg.n_agents:
            raise ValueError(f"env loss_mask length must be {cfg.n_agents}, got {mask.numel()}")
        mask = torch.clamp(mask, min=0.0) * static
        if float(mask.sum().item()) <= 0.0:
            raise ValueError("env loss_mask must leave at least one active agent")
        masks[env_idx] = mask
    return masks


def collect_rollout(trainer) -> "MappoRollout":
    cfg = trainer.cfg
    device = trainer.device
    rollout = trainer.rollout_cls(cfg, device=device)
    obs = trainer.last_obs
    h = trainer.h
    critic_obs = trainer.last_critic_obs
    for t in range(cfg.rollout_len):
        flat_obs = obs.reshape(cfg.num_envs * cfg.n_agents, cfg.obs_dim)
        flat_h = h.reshape(cfg.num_envs * cfg.n_agents, cfg.gru_hidden)
        with torch.no_grad():
            prev_rng = torch.get_rng_state()
            torch.set_rng_state(trainer._sampling_rng_state)
            try:
                action, logprob, h_next = trainer.model.sample_action(flat_obs, flat_h)
                trainer._sampling_rng_state = torch.get_rng_state()
            finally:
                torch.set_rng_state(prev_rng)
            if cfg.value_per_agent:
                value = trainer.model.value(
                    critic_obs.reshape(cfg.num_envs * cfg.n_agents, cfg.critic_obs_dim)
                ).view(cfg.num_envs, cfg.n_agents)
            else:
                value = trainer.model.value(critic_obs)
        action_3d = action.view(cfg.num_envs, cfg.n_agents, cfg.action_dim)
        next_obs_np, reward_np, terminated, truncated, next_critic_obs_np, infos = trainer.vec_env.step(
            action_3d.detach().cpu().numpy()
        )
        done_np = np.logical_or(terminated, truncated)
        rollout.actor_obs[:, :, t] = obs
        if cfg.value_per_agent:
            rollout.critic_obs[:, :, t] = critic_obs
        else:
            rollout.critic_obs[:, t] = critic_obs
        rollout.action[:, :, t] = action_3d
        rollout.logprob[:, :, t] = logprob.view(cfg.num_envs, cfg.n_agents)
        rollout.reward[:, :, t] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        rollout.agent_loss_mask[:, :, t] = step_loss_mask(cfg, infos, device=device)
        if cfg.value_per_agent:
            rollout.value[:, :, t] = value
        else:
            rollout.value[:, t] = value
        rollout.done[:, t] = torch.as_tensor(done_np, dtype=torch.float32, device=device)
        h = h_next.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
        rollout.h_init[:, :, t] = flat_h.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
        for e, done in enumerate(done_np):
            if bool(done):
                h[e] = 0.0
        obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
        critic_obs = trainer._critic_obs_from_np(next_critic_obs_np)
    with torch.no_grad():
        if cfg.value_per_agent:
            rollout.last_value = trainer.model.value(
                critic_obs.reshape(cfg.num_envs * cfg.n_agents, cfg.critic_obs_dim)
            ).view(cfg.num_envs, cfg.n_agents)
        else:
            rollout.last_value = trainer.model.value(critic_obs)
    rollout.last_done = rollout.done[:, -1].clone()
    trainer.last_obs = obs
    trainer.last_critic_obs = critic_obs
    trainer.h = h
    return rollout
