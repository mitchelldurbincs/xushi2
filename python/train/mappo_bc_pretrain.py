from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.mappo_model import _OWN_POSITION_SLICE, MappoActorCritic, MappoConfig
from xushi2.entity_obs import entity_obs_self_position


def _walk_to_objective_targets(obs: torch.Tensor, cfg: MappoConfig) -> torch.Tensor:
    if cfg.obs_encoder in ("entity_attention", "entity_attention_grid"):
        own_pos_np = entity_obs_self_position(obs.detach().cpu().numpy())
        own_pos = torch.as_tensor(own_pos_np, dtype=obs.dtype, device=obs.device)
    else:
        own_pos = obs[:, _OWN_POSITION_SLICE]
    move = -own_pos
    norm = torch.linalg.vector_norm(move, dim=-1, keepdim=True).clamp(min=1e-6)
    move = torch.where(norm > 0.02, move / norm, torch.zeros_like(move))
    target = torch.zeros(obs.shape[0], cfg.action_dim, dtype=obs.dtype, device=obs.device)
    target[:, :2] = move
    return target


def _walk_and_shoot_to_objective_targets(obs: torch.Tensor, cfg: MappoConfig) -> torch.Tensor:
    """Walk to cap, aim roughly toward nearest visible enemy, fire when enemy alive."""
    # Start with walk-to-objective base
    target = _walk_to_objective_targets(obs, cfg)

    # For flat observations, check enemy alive flag and relative position
    if cfg.obs_encoder not in ("entity_attention", "entity_attention_grid"):
        # Enemy alive at slice(10, 11), enemy_relative_position at slice(12, 14)
        enemy_alive = obs[:, 10:11]  # (B, 1)
        enemy_rel_pos = obs[:, 12:14]  # (B, 2)

        # Aim toward enemy: compute angle, set aim_delta roughly in that direction
        # aim_delta is index 2 in continuous actions
        # Simple heuristic: if enemy visible, aim roughly toward them; else aim toward cap
        enemy_angle = torch.atan2(enemy_rel_pos[:, 1:2], enemy_rel_pos[:, 0:1])
        # Clamp to reasonable range for tanh output (-1, 1) ~ (-45°, 45°)
        aim = torch.clamp(enemy_angle / (3.14159 / 4), -0.9, 0.9)
        # Only apply aim when enemy is alive
        target[:, 2:3] = torch.where(enemy_alive > 0.5, aim, target[:, 2:3])

        # Fire (primary_fire = index 3) when enemy is alive
        # Set primary_fire = 1.0 when enemy_alive > 0.5
        if cfg.binary_action_dim > 0:
            target[:, cfg.continuous_action_dim] = torch.where(
                enemy_alive.squeeze(-1) > 0.5,
                torch.ones_like(target[:, cfg.continuous_action_dim]),
                target[:, cfg.continuous_action_dim],
            )
    else:
        # Entity observation: enemy token index 1, position at _POSITION slice
        # For simplicity, just fire always in entity mode (less common)
        if cfg.binary_action_dim > 0:
            target[:, cfg.continuous_action_dim] = torch.ones_like(target[:, cfg.continuous_action_dim])

    return target


def _collect_walk_bc_sequence(
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    batch_size: int,
    seed: int,
    target_fn: Callable[[torch.Tensor, MappoConfig], torch.Tensor] = _walk_to_objective_targets,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    max_decisions = max(1, int(np.ceil(float(batch_size) / float(cfg.n_agents))))
    env = env_fn()
    try:
        obs, _info = env.reset(seed=seed)
        for _ in range(max_decisions):
            obs_parts.append(obs.astype(np.float32, copy=True))
            target = target_fn(torch.as_tensor(obs, dtype=torch.float32), cfg)
            target_parts.append(target.numpy().astype(np.float32, copy=True))
            obs, _reward, term, trunc, _info = env.step(target.numpy())
            if term or trunc:
                obs, _info = env.reset(seed=seed + len(obs_parts))
    finally:
        env.close()
    obs_seq = torch.as_tensor(np.stack(obs_parts, axis=0), dtype=torch.float32)
    target_seq = torch.as_tensor(np.stack(target_parts, axis=0), dtype=torch.float32)
    return obs_seq, target_seq


def bc_pretrain_walk_to_objective(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_label: str = "phase4",
) -> None:
    if steps <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    for step in range(1, int(steps) + 1):
        obs_seq, target_seq = _collect_walk_bc_sequence(
            env_fn, cfg, batch_size=int(batch_size), seed=int(seed) + step
        )
        h = model.init_hidden(cfg.n_agents)
        cont_losses = []
        binary_losses = []
        for t in range(obs_seq.shape[0]):
            mean, _log_std, logits, _target_logits, h = model.policy_outputs(obs_seq[t], h)
            pred_cont = torch.tanh(mean)
            target = target_seq[t]
            cont_losses.append(
                torch.nn.functional.mse_loss(pred_cont, target[:, : cfg.continuous_action_dim])
            )
            binary_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    target[
                        :,
                        cfg.continuous_action_dim : cfg.continuous_action_dim
                        + cfg.binary_action_dim,
                    ],
                )
            )
        cont_loss = torch.stack(cont_losses).mean()
        binary_loss = torch.stack(binary_losses).mean()
        loss = cont_loss + 0.1 * binary_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(
                f"[{log_label}/mappo] bc_pretrain step={step}/{steps} "
                f"loss={float(loss.item()):.4f} "
                f"cont_loss={float(cont_loss.item()):.4f} "
                f"binary_loss={float(binary_loss.item()):.4f}",
                flush=True,
            )


def bc_pretrain_walk_and_shoot_to_objective(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_label: str = "phase4",
) -> None:
    """BC pretrain that walks to cap, aims at enemies, and fires when visible."""
    if steps <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    for step in range(1, int(steps) + 1):
        obs_seq, target_seq = _collect_walk_bc_sequence(
            env_fn, cfg, batch_size=int(batch_size), seed=int(seed) + step,
            target_fn=_walk_and_shoot_to_objective_targets,
        )
        h = model.init_hidden(cfg.n_agents)
        cont_losses = []
        binary_losses = []
        for t in range(obs_seq.shape[0]):
            mean, _log_std, logits, _target_logits, h = model.policy_outputs(obs_seq[t], h)
            pred_cont = torch.tanh(mean)
            target = target_seq[t]
            cont_losses.append(
                torch.nn.functional.mse_loss(pred_cont, target[:, : cfg.continuous_action_dim])
            )
            binary_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    target[
                        :,
                        cfg.continuous_action_dim : cfg.continuous_action_dim
                        + cfg.binary_action_dim,
                    ],
                )
            )
        cont_loss = torch.stack(cont_losses).mean()
        binary_loss = torch.stack(binary_losses).mean()
        loss = cont_loss + 0.1 * binary_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(
                f"[{log_label}/mappo] bc_pretrain_walk_and_shoot step={step}/{steps} "
                f"loss={float(loss.item()):.4f} "
                f"cont_loss={float(cont_loss.item()):.4f} "
                f"binary_loss={float(binary_loss.item()):.4f}",
                flush=True,
            )
