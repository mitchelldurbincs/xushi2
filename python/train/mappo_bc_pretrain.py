from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.mappo_model import (
    _OWN_POSITION_SLICE,
    MappoActorCritic,
    MappoConfig,
    aim_aux_loss_and_rmse,
    mode_aux_loss_and_accuracy,
    mode_aux_targets,
    target_selection_aux_loss_and_accuracy,
)
from xushi2.entity_obs import entity_obs_self_position
from xushi2.obs_manifest import actor_field_slice

_AIM_ACTION_INDEX = 2
_ENEMY_ALIVE_SLICE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS_SLICE = actor_field_slice("enemy_relative_position")
_OWN_AIM_UNIT_SLICE = actor_field_slice("own_aim_unit")
_AIM_DELTA_LIMIT = float(np.pi / 4.0)


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


def load_bc_aim_target_model(
    checkpoint_path: str | Path,
    cfg: MappoConfig,
) -> MappoActorCritic:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg_raw = dict(raw.get("config", {}).get("mappo", {}))
    if not ckpt_cfg_raw:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain config.mappo")
    ckpt_cfg = MappoConfig(**ckpt_cfg_raw)
    if ckpt_cfg.obs_encoder != "flat":
        raise ValueError("BC aim-target checkpoint inference currently supports only flat obs")
    compatibility = {
        "n_agents": (ckpt_cfg.n_agents, cfg.n_agents),
        "obs_dim": (ckpt_cfg.obs_dim, cfg.obs_dim),
        "action_dim": (ckpt_cfg.action_dim, cfg.action_dim),
        "continuous_action_dim": (
            ckpt_cfg.continuous_action_dim,
            cfg.continuous_action_dim,
        ),
    }
    mismatches = {k: v for k, v in compatibility.items() if v[0] != v[1]}
    if mismatches:
        raise ValueError(f"BC aim-target checkpoint is incompatible: {mismatches}")
    if ckpt_cfg.continuous_action_dim <= _AIM_ACTION_INDEX:
        raise ValueError("BC aim-target checkpoint has no continuous aim action")

    model = MappoActorCritic(ckpt_cfg)
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


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
        enemy_alive = obs[:, _ENEMY_ALIVE_SLICE]
        enemy_rel_pos = obs[:, _ENEMY_REL_POS_SLICE]
        own_aim_unit = obs[:, _OWN_AIM_UNIT_SLICE]

        # Action index 2 is a delta from current aim, not an absolute angle.
        enemy_angle = torch.atan2(enemy_rel_pos[:, 1:2], enemy_rel_pos[:, 0:1])
        current_angle = torch.atan2(own_aim_unit[:, 0:1], own_aim_unit[:, 1:2])
        aim_delta = torch.atan2(
            torch.sin(enemy_angle - current_angle),
            torch.cos(enemy_angle - current_angle),
        )
        aim = torch.clamp(aim_delta / _AIM_DELTA_LIMIT, -1.0, 1.0)
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


@torch.no_grad()
def _replace_aim_target_from_model(
    target: torch.Tensor,
    obs: torch.Tensor,
    h: torch.Tensor,
    aim_target_model: MappoActorCritic,
) -> tuple[torch.Tensor, torch.Tensor]:
    if obs.shape[1] != aim_target_model.cfg.obs_dim:
        raise ValueError(
            "BC aim-target model obs_dim mismatch: "
            f"obs={obs.shape[1]}, model={aim_target_model.cfg.obs_dim}"
        )
    device = next(aim_target_model.parameters()).device
    obs_for_model = obs.to(device=device, dtype=next(aim_target_model.parameters()).dtype)
    h_for_model = h.to(device=device, dtype=obs_for_model.dtype)
    features, h_next = aim_target_model.actor_head_features(obs_for_model, h_for_model)
    aim = torch.tanh(aim_target_model.actor_mean_head(features))[:, _AIM_ACTION_INDEX]
    visible_enemy = obs_for_model[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    out = target.clone()
    out[:, _AIM_ACTION_INDEX] = torch.where(
        visible_enemy.to(device=target.device),
        aim.to(device=target.device, dtype=target.dtype),
        out[:, _AIM_ACTION_INDEX],
    )
    return out, h_next.detach()


def _collect_walk_bc_sequence(
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    batch_size: int,
    seed: int,
    target_fn: Callable[[torch.Tensor, MappoConfig], torch.Tensor] = _walk_to_objective_targets,
    aim_target_model: MappoActorCritic | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    max_decisions = max(1, int(np.ceil(float(batch_size) / float(cfg.n_agents))))
    env = env_fn()
    aim_h = aim_target_model.init_hidden(cfg.n_agents) if aim_target_model is not None else None
    try:
        obs, _info = env.reset(seed=seed)
        for _ in range(max_decisions):
            obs_parts.append(obs.astype(np.float32, copy=True))
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            target = target_fn(obs_tensor, cfg)
            if aim_target_model is not None:
                assert aim_h is not None
                target, aim_h = _replace_aim_target_from_model(
                    target, obs_tensor, aim_h, aim_target_model
                )
            target_parts.append(target.numpy().astype(np.float32, copy=True))
            obs, _reward, term, trunc, _info = env.step(target.numpy())
            if term or trunc:
                obs, _info = env.reset(seed=seed + len(obs_parts))
                if aim_target_model is not None:
                    aim_h = aim_target_model.init_hidden(cfg.n_agents)
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
    device = next(model.parameters()).device
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
            obs_seq = obs_seq.to(device)
            target_seq = target_seq.to(device)
            h = model.init_hidden(cfg.n_agents).to(device)
            cont_losses = []
            binary_losses = []
            aim_aux_losses = []
            aim_aux_rmses = []
            aim_aux_counts = []
            target_aux_losses = []
            target_aux_accs = []
            target_aux_counts = []
            mode_aux_losses = []
            mode_aux_accs = []
            mode_aux_counts = []
            for t in range(obs_seq.shape[0]):
                features, h = model.actor_head_features(obs_seq[t], h)
                mean, logits, target_selection_logits = model.policy_heads_from_features(
                    obs_seq[t], features
                )
                mode_logits = model.mode_logits_from_features(features)
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
                target_aux_loss, target_aux_acc, target_aux_count = (
                    target_selection_aux_loss_and_accuracy(
                        target_selection_logits, obs_seq[t], cfg
                    )
                )
                target_aux_losses.append(target_aux_loss)
                target_aux_accs.append(target_aux_acc)
                target_aux_counts.append(target_aux_count)
                mode_target = torch.zeros(
                    obs_seq.shape[1], dtype=torch.long, device=obs_seq.device
                )
                mode_loss, mode_acc, mode_count = mode_aux_loss_and_accuracy(
                    mode_logits, obs_seq[t], cfg, labels=mode_target
                )
                mode_aux_losses.append(mode_loss)
                mode_aux_accs.append(mode_acc)
                mode_aux_counts.append(mode_count)
            cont_loss = torch.stack(cont_losses).mean()
            binary_loss = torch.stack(binary_losses).mean()
            aim_aux_loss = torch.stack(aim_aux_losses).mean()
            aim_aux_count = torch.stack(aim_aux_counts).sum()
            aim_aux_rmse = (
                torch.stack(aim_aux_rmses).mean()
                if float(aim_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            target_aux_loss = torch.stack(target_aux_losses).mean()
            target_aux_count = torch.stack(target_aux_counts).sum()
            target_aux_acc = (
                torch.stack(target_aux_accs).mean()
                if float(target_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            mode_aux_loss = torch.stack(mode_aux_losses).mean()
            mode_aux_count = torch.stack(mode_aux_counts).sum()
            mode_aux_acc = (
                torch.stack(mode_aux_accs).mean()
                if float(mode_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            loss = (
                cont_loss
                + 0.1 * binary_loss
                + cfg.aim_aux_coef * aim_aux_loss
                + (cfg.mode_aux_coef * mode_aux_loss if cfg.mode_gated_combat else 0.0)
                + cfg.target_selection_aux_coef * target_aux_loss
            )
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
                    f"mode_acc={float(mode_aux_acc.item()):.3f} "
                    f"aim_aux_rmse={float(aim_aux_rmse.item()):.4f} "
                    f"target_aux_acc={float(target_aux_acc.item()):.3f}",
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
    aim_target_checkpoint: str | Path | None = None,
    aim_rehearsal_env_fn: Callable[[], gym.Env] | None = None,
    aim_rehearsal_batch_size: int = 0,
) -> None:
    """BC pretrain that walks to cap, aims at enemies, and fires when visible."""
    if steps <= 0:
        return
    device = next(model.parameters()).device
    aim_target_model = (
        load_bc_aim_target_model(aim_target_checkpoint, cfg)
        if aim_target_checkpoint is not None
        else None
    )
    with _freeze_actor_aim_for_bc(model, cfg, enabled=freeze_actor_aim):
        opt = torch.optim.Adam(_bc_trainable_parameters(model), lr=float(learning_rate))
        if freeze_actor_aim:
            print(
                f"[{log_label}/mappo] bc_pretrain_walk_and_shoot freeze_actor_aim=true",
                flush=True,
            )
        if aim_target_checkpoint is not None:
            print(
                f"[{log_label}/mappo] bc_pretrain_walk_and_shoot "
                f"aim_target_checkpoint={aim_target_checkpoint}",
                flush=True,
            )
        for step in range(1, int(steps) + 1):
            obs_seq, target_seq = _collect_walk_bc_sequence(
                env_fn,
                cfg,
                batch_size=int(batch_size),
                seed=int(seed) + step,
                target_fn=_walk_and_shoot_to_objective_targets,
                aim_target_model=aim_target_model,
            )
            obs_seq = obs_seq.to(device)
            target_seq = target_seq.to(device)
            cont_losses = []
            binary_losses = []
            aim_aux_losses = []
            aim_aux_rmses = []
            aim_aux_counts = []
            target_aux_losses = []
            target_aux_accs = []
            target_aux_counts = []
            mode_aux_losses = []
            mode_aux_accs = []
            mode_aux_counts = []

            sequences = [(obs_seq, target_seq)]
            if (
                aim_target_model is not None
                and aim_rehearsal_env_fn is not None
                and int(aim_rehearsal_batch_size) > 0
            ):
                rehearsal_obs_seq, rehearsal_target_seq = _collect_walk_bc_sequence(
                    aim_rehearsal_env_fn,
                    cfg,
                    batch_size=int(aim_rehearsal_batch_size),
                    seed=int(seed) + 1_000_000 + step,
                    target_fn=_walk_and_shoot_to_objective_targets,
                    aim_target_model=aim_target_model,
                )
                rehearsal_obs_seq = rehearsal_obs_seq.to(device)
                rehearsal_target_seq = rehearsal_target_seq.to(device)
                sequences.append((rehearsal_obs_seq, rehearsal_target_seq))

            for seq_obs, seq_target in sequences:
                h = model.init_hidden(cfg.n_agents).to(device)
                for t in range(seq_obs.shape[0]):
                    features, h = model.actor_head_features(seq_obs[t], h)
                    mean, logits, target_selection_logits = model.policy_heads_from_features(
                        seq_obs[t], features
                    )
                    mode_logits = model.mode_logits_from_features(features)
                    pred_cont = torch.tanh(mean)
                    target = seq_target[t]
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
                        aim_pred, seq_obs[t], cfg
                    )
                    aim_aux_losses.append(aim_loss)
                    aim_aux_rmses.append(aim_rmse)
                    aim_aux_counts.append(aim_count)
                    target_aux_loss, target_aux_acc, target_aux_count = (
                        target_selection_aux_loss_and_accuracy(
                            target_selection_logits, seq_obs[t], cfg
                        )
                    )
                    target_aux_losses.append(target_aux_loss)
                    target_aux_accs.append(target_aux_acc)
                    target_aux_counts.append(target_aux_count)
                    mode_labels, _mode_valid = mode_aux_targets(seq_obs[t], cfg)
                    mode_loss, mode_acc, mode_count = mode_aux_loss_and_accuracy(
                        mode_logits, seq_obs[t], cfg, labels=mode_labels
                    )
                    mode_aux_losses.append(mode_loss)
                    mode_aux_accs.append(mode_acc)
                    mode_aux_counts.append(mode_count)
            cont_loss = torch.stack(cont_losses).mean()
            binary_loss = torch.stack(binary_losses).mean()
            aim_aux_loss = torch.stack(aim_aux_losses).mean()
            aim_aux_count = torch.stack(aim_aux_counts).sum()
            aim_aux_rmse = (
                torch.stack(aim_aux_rmses).mean()
                if float(aim_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            target_aux_loss = torch.stack(target_aux_losses).mean()
            target_aux_count = torch.stack(target_aux_counts).sum()
            target_aux_acc = (
                torch.stack(target_aux_accs).mean()
                if float(target_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            mode_aux_loss = torch.stack(mode_aux_losses).mean()
            mode_aux_count = torch.stack(mode_aux_counts).sum()
            mode_aux_acc = (
                torch.stack(mode_aux_accs).mean()
                if float(mode_aux_count.item()) > 0.0
                else obs_seq.new_tensor(0.0)
            )
            loss = (
                cont_loss
                + 0.1 * binary_loss
                + cfg.aim_aux_coef * aim_aux_loss
                + (cfg.mode_aux_coef * mode_aux_loss if cfg.mode_gated_combat else 0.0)
                + cfg.target_selection_aux_coef * target_aux_loss
            )
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
                    f"mode_acc={float(mode_aux_acc.item()):.3f} "
                    f"aim_aux_rmse={float(aim_aux_rmse.item()):.4f} "
                    f"target_aux_acc={float(target_aux_acc.item()):.3f}",
                    flush=True,
                )
