from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from train.device import resolve_device
from train.mappo_model import MappoConfig

if TYPE_CHECKING:
    from train.mappo_rollout_trainer import MappoRollout

_INFO_METRIC_KEYS = (
    "majority_on_point_alpha",
    "majority_on_point_reward_a",
    "majority_on_point_reward_b",
    "majority_on_point_advantage_a",
    "majority_on_point_advantage_b",
    "majority_on_point_count_a",
    "majority_on_point_count_b",
    "uncontested_on_point_alpha",
    "uncontested_on_point_reward_a",
    "uncontested_on_point_reward_b",
    "uncontested_on_point_count_a",
    "uncontested_on_point_count_b",
    "objective_unlock_seconds",
    "objective_capture_seconds",
)

_OBJECTIVE_METRIC_KEYS = (
    "uncontested_on_point_seconds_a",
    "uncontested_on_point_seconds_b",
    "majority_on_point_seconds_a",
    "majority_on_point_seconds_b",
    "alive_edge_no_score_seconds_a",
    "alive_edge_no_score_seconds_b",
    "cap_progress_gain_ticks",
    "cap_progress_loss_ticks",
    "team_a_score_delta_ticks",
    "team_b_score_delta_ticks",
)


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


def collect_rollout(trainer) -> MappoRollout:
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
            action, logprob, h_next = trainer.model.sample_action(
                flat_obs,
                flat_h,
                generator=trainer.policy_sampling_generator,
            )
            if cfg.value_per_agent:
                value = trainer.model.value(
                    critic_obs.reshape(cfg.num_envs * cfg.n_agents, cfg.critic_obs_dim)
                ).view(cfg.num_envs, cfg.n_agents)
            else:
                value = trainer.model.value(critic_obs)
        action_3d = action.view(cfg.num_envs, cfg.n_agents, cfg.action_dim)
        step_result = trainer.vec_env.step(action_3d.detach().cpu().numpy())
        next_obs_np, reward_np, terminated, truncated, next_critic_obs_np, infos = step_result
        _accumulate_info_metrics(rollout.info_metrics, infos)
        done_np = np.logical_or(terminated, truncated)
        truncated_np = np.logical_and(truncated, np.logical_not(terminated))
        if truncated_np.any():
            # A truncated step's `next_critic_obs` belongs to the freshly reset
            # episode, so the value the critic should bootstrap from has to
            # come from the pre-reset observation the env stashed for us.
            _fill_truncated_values(trainer, rollout, infos, truncated_np, t)
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
        rollout.terminated[:, t] = torch.as_tensor(
            terminated, dtype=torch.float32, device=device
        )
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
    rollout.last_terminated = rollout.terminated[:, -1].clone()
    trainer.last_obs = obs
    trainer.last_critic_obs = critic_obs
    trainer.h = h
    return rollout


def _fill_truncated_values(trainer, rollout, infos, truncated_np, t) -> None:
    """Store V(final_critic_observation) for each truncated env at step t.

    Envs that do not publish `final_critic_observation` leave the entry at 0
    and the step falls back to the reset episode's value, i.e. the previous
    behavior, rather than silently producing a wrong bootstrap target.
    """
    cfg = trainer.cfg
    for env_idx, was_truncated in enumerate(truncated_np):
        if not bool(was_truncated):
            continue
        info = infos[env_idx]
        final_critic = info.get("final_critic_observation")
        if final_critic is None:
            continue
        obs = torch.as_tensor(
            np.asarray(final_critic, dtype=np.float32), device=trainer.device
        )
        with torch.no_grad():
            if cfg.value_per_agent:
                value = trainer.model.value(
                    obs.reshape(cfg.n_agents, cfg.critic_obs_dim)
                ).view(cfg.n_agents)
                rollout.truncated_value[env_idx, :, t] = value
            else:
                value = trainer.model.value(obs.reshape(1, cfg.critic_obs_dim)).view(())
                rollout.truncated_value[env_idx, t] = value


def _accumulate_info_metrics(dst: dict[str, float], infos: list[dict]) -> None:
    for info in infos:
        dst["info_metric_samples"] = float(dst.get("info_metric_samples", 0.0)) + 1.0
        for key in _INFO_METRIC_KEYS:
            value = info.get(key)
            if isinstance(value, (int, float, np.floating)):
                dst[key] = float(dst.get(key, 0.0)) + float(value)
        objective_metrics = info.get("objective_metrics")
        if not isinstance(objective_metrics, dict):
            continue
        for key in _OBJECTIVE_METRIC_KEYS:
            value = objective_metrics.get(key)
            if isinstance(value, (int, float, np.floating)):
                dst[key] = float(dst.get(key, 0.0)) + float(value)
