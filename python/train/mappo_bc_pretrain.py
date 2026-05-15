from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.mappo_model import (
    _OWN_POSITION_SLICE,
    MappoActorCritic,
    MappoConfig,
    aim_aux_loss_and_rmse,
)
from xushi2.entity_obs import entity_obs_self_position

_AIM_ACTION_INDEX = 2


@contextmanager
def _freeze_actor_aim_for_bc(
    model: MappoActorCritic,
    cfg: MappoConfig,
    *,
    enabled: bool,
) -> Iterator[None]:
    """Protect a learned flat-observation aim mapping during BC.

    Phase 4's current actor shares its representation between movement, aim,
    and fire. To keep BC from overwriting the aim-only checkpoint mapping, we
    freeze the shared actor trunk during BC and mask gradients for the aim row
    of the continuous action head. Movement and binary output heads can still
    adapt to the full-env BC target.
    """
    if not enabled:
        yield
        return
    if cfg.obs_encoder != "flat":
        raise ValueError("freeze_actor_aim during BC currently supports only flat observations")
    if cfg.continuous_action_dim <= _AIM_ACTION_INDEX:
        raise ValueError("freeze_actor_aim requires an aim continuous action dimension")

    frozen_modules: list[nn.Module | None] = [
        model.actor_embed,
        model.actor_gru,
        model.actor_body,
        model.actor_aim_aux_head,
    ]
    previous: list[tuple[nn.Parameter, bool]] = []
    for module in frozen_modules:
        if module is None:
            continue
        for param in module.parameters():
            previous.append((param, param.requires_grad))
            param.requires_grad_(False)

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _mask_aim_row(grad: torch.Tensor) -> torch.Tensor:
        masked = grad.clone()
        masked[_AIM_ACTION_INDEX].zero_()
        return masked

    handles.append(model.actor_mean_head.weight.register_hook(_mask_aim_row))
    if model.actor_mean_head.bias is not None:
        handles.append(model.actor_mean_head.bias.register_hook(_mask_aim_row))
    if model.log_std.requires_grad and model.log_std.numel() > _AIM_ACTION_INDEX:
        handles.append(model.log_std.register_hook(_mask_aim_row))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for param, requires_grad in previous:
            param.requires_grad_(requires_grad)


def _bc_trainable_parameters(model: MappoActorCritic) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


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
            target[:, cfg.continuous_action_dim] = torch.ones_like(
                target[:, cfg.continuous_action_dim]
            )

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
    freeze_actor_aim: bool = False,
) -> None:
    if steps <= 0:
        return
    with _freeze_actor_aim_for_bc(model, cfg, enabled=freeze_actor_aim):
        opt = torch.optim.Adam(_bc_trainable_parameters(model), lr=float(learning_rate))
        if freeze_actor_aim:
            print(
                f"[{log_label}/mappo] bc_pretrain freeze_actor_aim=true",
                flush=True,
            )
        for step in range(1, int(steps) + 1):
            obs_seq, target_seq = _collect_walk_bc_sequence(
                env_fn, cfg, batch_size=int(batch_size), seed=int(seed) + step
            )
            h = model.init_hidden(cfg.n_agents)
            cont_losses = []
            binary_losses = []
            aim_aux_losses = []
            aim_aux_rmses = []
            aim_aux_counts = []
            for t in range(obs_seq.shape[0]):
                features, h = model.actor_head_features(obs_seq[t], h)
                mean = model.actor_mean_head(features)
                logits = model.actor_binary_head(features)
                pred_cont = torch.tanh(mean)
                target = target_seq[t]
                cont_losses.append(
                    torch.nn.functional.mse_loss(
                        pred_cont, target[:, : cfg.continuous_action_dim]
                    )
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
                aim_pred = model.aim_aux_prediction_from_features(features)
                aim_loss, aim_rmse, aim_count = aim_aux_loss_and_rmse(
                    aim_pred, obs_seq[t], cfg
                )
                aim_aux_losses.append(aim_loss)
                aim_aux_rmses.append(aim_rmse)
                aim_aux_counts.append(aim_count)
            cont_loss = torch.stack(cont_losses).mean()
            binary_loss = torch.stack(binary_losses).mean()
            aim_aux_loss = torch.stack(aim_aux_losses).mean()
            aim_aux_count = torch.stack(aim_aux_counts).sum()
            aim_aux_rmse = (
                torch.stack(aim_aux_rmses).mean()
                if float(aim_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            loss = cont_loss + 0.1 * binary_loss + cfg.aim_aux_coef * aim_aux_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(_bc_trainable_parameters(model), cfg.max_grad_norm)
            opt.step()
            if step == 1 or step == steps or step % max(1, steps // 5) == 0:
                print(
                    f"[{log_label}/mappo] bc_pretrain step={step}/{steps} "
                    f"loss={float(loss.item()):.4f} "
                    f"cont_loss={float(cont_loss.item()):.4f} "
                    f"binary_loss={float(binary_loss.item()):.4f} "
                    f"aim_aux_rmse={float(aim_aux_rmse.item()):.4f}",
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
    freeze_actor_aim: bool = False,
) -> None:
    """BC pretrain that walks to cap, aims at enemies, and fires when visible."""
    if steps <= 0:
        return
    with _freeze_actor_aim_for_bc(model, cfg, enabled=freeze_actor_aim):
        opt = torch.optim.Adam(_bc_trainable_parameters(model), lr=float(learning_rate))
        if freeze_actor_aim:
            print(
                f"[{log_label}/mappo] bc_pretrain_walk_and_shoot freeze_actor_aim=true",
                flush=True,
            )
        for step in range(1, int(steps) + 1):
            obs_seq, target_seq = _collect_walk_bc_sequence(
                env_fn,
                cfg,
                batch_size=int(batch_size),
                seed=int(seed) + step,
                target_fn=_walk_and_shoot_to_objective_targets,
            )
            h = model.init_hidden(cfg.n_agents)
            cont_losses = []
            binary_losses = []
            aim_aux_losses = []
            aim_aux_rmses = []
            aim_aux_counts = []
            for t in range(obs_seq.shape[0]):
                features, h = model.actor_head_features(obs_seq[t], h)
                mean = model.actor_mean_head(features)
                logits = model.actor_binary_head(features)
                pred_cont = torch.tanh(mean)
                target = target_seq[t]
                cont_losses.append(
                    torch.nn.functional.mse_loss(
                        pred_cont, target[:, : cfg.continuous_action_dim]
                    )
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
                aim_pred = model.aim_aux_prediction_from_features(features)
                aim_loss, aim_rmse, aim_count = aim_aux_loss_and_rmse(
                    aim_pred, obs_seq[t], cfg
                )
                aim_aux_losses.append(aim_loss)
                aim_aux_rmses.append(aim_rmse)
                aim_aux_counts.append(aim_count)
            cont_loss = torch.stack(cont_losses).mean()
            binary_loss = torch.stack(binary_losses).mean()
            aim_aux_loss = torch.stack(aim_aux_losses).mean()
            aim_aux_count = torch.stack(aim_aux_counts).sum()
            aim_aux_rmse = (
                torch.stack(aim_aux_rmses).mean()
                if float(aim_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            loss = cont_loss + 0.1 * binary_loss + cfg.aim_aux_coef * aim_aux_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(_bc_trainable_parameters(model), cfg.max_grad_norm)
            opt.step()
            if step == 1 or step == steps or step % max(1, steps // 5) == 0:
                print(
                    f"[{log_label}/mappo] bc_pretrain_walk_and_shoot step={step}/{steps} "
                    f"loss={float(loss.item()):.4f} "
                    f"cont_loss={float(cont_loss.item()):.4f} "
                    f"binary_loss={float(binary_loss.item()):.4f} "
                    f"aim_aux_rmse={float(aim_aux_rmse.item()):.4f}",
                    flush=True,
                )
