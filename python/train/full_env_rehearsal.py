from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_model import (
    MappoActorCritic,
    MappoConfig,
    target_selection_aux_loss_and_accuracy,
)
from xushi2.entity_obs import ENTITY_TOKEN_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.obs_manifest import actor_field_slice
from xushi2 import xushi2_cpp as _cpp

_MOVE_ACTION_INDICES = (0, 1)
_AIM_ACTION_INDEX = 2
_PRIMARY_FIRE_BINARY_INDEX = 0
_AIM_DELTA_LIMIT = math.pi / 4.0

_OWN_AIM = actor_field_slice("own_aim_unit")
_OWN_POSITION = actor_field_slice("own_position")
_SELF_ON_POINT = actor_field_slice("self_on_point")
_ENEMY_ALIVE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS = actor_field_slice("enemy_relative_position")

_SELF_TOKEN = 0
_FIRST_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 4
_ENTITY_POSITION = slice(8, 10)
_ENTITY_AIM = slice(12, 14)
_ENTITY_AUX = 17


@dataclass(frozen=True)
class FullEnvRehearsalGate:
    status: str
    passed: bool
    path: Path
    metrics: dict[str, float]
    thresholds: dict[str, float]


@dataclass(frozen=True)
class PolicyStateBatch:
    obs: torch.Tensor
    target_cont: torch.Tensor
    target_binary: torch.Tensor
    policy_cont: torch.Tensor
    policy_binary: torch.Tensor


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def scripted_objective_focus_fire_targets(
    actor_obs: torch.Tensor,
    cfg: MappoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Actor-observation-only full-env rehearsal labels.

    This function deliberately accepts only actor observations plus the model
    config. It must not read env, sim, critic obs, info dicts, or full slot
    state; that keeps the rehearsal teacher inside the actor information
    boundary.
    """
    if cfg.obs_encoder != "flat":
        raise ValueError("full-env rehearsal scripted teacher supports only flat obs")
    obs = actor_obs
    cont = torch.zeros(
        obs.shape[0],
        cfg.continuous_action_dim,
        dtype=obs.dtype,
        device=obs.device,
    )
    binary = torch.zeros(
        obs.shape[0],
        cfg.binary_action_dim,
        dtype=obs.dtype,
        device=obs.device,
    )

    own_pos = obs[:, _OWN_POSITION]
    self_on_point = obs[:, _SELF_ON_POINT].squeeze(-1) > 0.5
    to_objective = -own_pos
    dist = torch.linalg.vector_norm(to_objective, dim=-1, keepdim=True)
    move = torch.where(dist > 1.0e-6, to_objective / dist.clamp(min=1.0e-6), to_objective)
    move = torch.where(self_on_point[:, None], torch.zeros_like(move), move)
    cont[:, list(_MOVE_ACTION_INDICES)] = move.clamp(-1.0, 1.0)

    enemy_alive = obs[:, _ENEMY_ALIVE].squeeze(-1) > 0.5
    rel = obs[:, _ENEMY_REL_POS]
    rel_norm = torch.linalg.vector_norm(rel, dim=-1)
    visible = enemy_alive & (rel_norm > 1.0e-6)
    target_angle = torch.atan2(rel[:, 0], rel[:, 1])
    aim_unit = obs[:, _OWN_AIM]
    current_angle = torch.atan2(aim_unit[:, 0], aim_unit[:, 1])
    delta = _wrap_angle(target_angle - current_angle).clamp(
        -_AIM_DELTA_LIMIT, _AIM_DELTA_LIMIT
    )
    if cfg.continuous_action_dim > _AIM_ACTION_INDEX:
        cont[:, _AIM_ACTION_INDEX] = torch.where(
            visible,
            delta / _AIM_DELTA_LIMIT,
            torch.zeros_like(delta),
        )
    if cfg.binary_action_dim > _PRIMARY_FIRE_BINARY_INDEX:
        binary[:, _PRIMARY_FIRE_BINARY_INDEX] = visible.to(obs.dtype)
    return cont, binary


def multi_enemy_visible_targets(
    actor_obs: torch.Tensor,
    cfg: MappoConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervised labels from current visible enemy tokens only.

    The labels mirror the direct ``multi_enemy_visible`` diagnostic policy:
    move toward the objective until on point, then aim/fire at the nearest
    currently visible enemy token. Hidden enemy tokens are ignored via the
    actor-visible mask embedded in the observation.
    """
    if cfg.obs_encoder != "entity_attention_grid":
        raise ValueError("multi_enemy_visible teacher requires entity_attention_grid obs")
    if cfg.entity_token_count != MULTI_ENEMY_TOKEN_COUNT:
        raise ValueError(
            "multi_enemy_visible teacher requires the multi-enemy token layout"
        )
    obs = actor_obs
    token_width = cfg.entity_token_count * cfg.entity_token_dim
    if cfg.entity_token_dim != ENTITY_TOKEN_DIM or obs.shape[-1] < token_width:
        raise ValueError("multi_enemy_visible teacher received incompatible token shape")
    tokens = obs[:, :token_width].view(
        obs.shape[0], cfg.entity_token_count, cfg.entity_token_dim
    )
    mask = obs[:, token_width : token_width + cfg.entity_token_count] > 0.5

    cont = torch.zeros(
        obs.shape[0],
        cfg.continuous_action_dim,
        dtype=obs.dtype,
        device=obs.device,
    )
    binary = torch.zeros(
        obs.shape[0],
        cfg.binary_action_dim,
        dtype=obs.dtype,
        device=obs.device,
    )

    objective_rel = tokens[:, _OBJECTIVE_TOKEN, _ENTITY_POSITION]
    self_on_point = tokens[:, _SELF_TOKEN, _ENTITY_AUX] > 0.5
    obj_dist = torch.linalg.vector_norm(objective_rel, dim=-1, keepdim=True)
    move = torch.where(
        obj_dist > 1.0e-6,
        objective_rel / obj_dist.clamp(min=1.0e-6),
        torch.zeros_like(objective_rel),
    )
    move = torch.where(self_on_point[:, None], torch.zeros_like(move), move)
    cont[:, list(_MOVE_ACTION_INDICES)] = move.clamp(-1.0, 1.0)

    enemy_tokens = tokens[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN, :]
    enemy_mask = mask[:, _FIRST_ENEMY_TOKEN:_OBJECTIVE_TOKEN]
    enemy_rel = enemy_tokens[:, :, _ENTITY_POSITION]
    enemy_dist = torch.linalg.vector_norm(enemy_rel, dim=-1)
    masked_dist = enemy_dist.masked_fill(~enemy_mask, float("inf"))
    target_idx = masked_dist.argmin(dim=-1)
    has_target = torch.isfinite(masked_dist.gather(1, target_idx[:, None]).squeeze(1))
    target_rel = enemy_rel[torch.arange(obs.shape[0], device=obs.device), target_idx]

    target_angle = torch.atan2(target_rel[:, 1], target_rel[:, 0])
    aim_unit = tokens[:, _SELF_TOKEN, _ENTITY_AIM]
    current_angle = torch.atan2(aim_unit[:, 0], aim_unit[:, 1])
    delta = _wrap_angle(target_angle - current_angle).clamp(
        -_AIM_DELTA_LIMIT, _AIM_DELTA_LIMIT
    )
    if cfg.continuous_action_dim > _AIM_ACTION_INDEX:
        cont[:, _AIM_ACTION_INDEX] = torch.where(
            has_target,
            delta / _AIM_DELTA_LIMIT,
            torch.zeros_like(delta),
        )
    if cfg.binary_action_dim > _PRIMARY_FIRE_BINARY_INDEX:
        binary[:, _PRIMARY_FIRE_BINARY_INDEX] = has_target.to(obs.dtype)
    return cont, binary


def full_env_rehearsal_loss(
    model: MappoActorCritic,
    actor_obs: torch.Tensor,
    config: dict[str, Any] | None = None,
    target_cont: torch.Tensor | None = None,
    target_binary: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = model.cfg
    options = dict(config or {})
    device = next(model.parameters()).device
    obs = actor_obs.to(device=device, dtype=torch.float32)
    if target_cont is None or target_binary is None:
        target_cont, target_binary = scripted_objective_focus_fire_targets(obs, cfg)
    else:
        target_cont = target_cont.to(device=device, dtype=torch.float32)
        target_binary = target_binary.to(device=device, dtype=torch.float32)
    h = model.init_hidden(obs.shape[0]).to(device)
    features, _h_next = model.actor_head_features(obs, h)
    mean, logits, target_selection_logits = model.policy_heads_from_features(obs, features)
    logits = model.masked_binary_logits(obs, logits)
    pred_cont = torch.tanh(mean)

    move_loss = torch.nn.functional.mse_loss(
        pred_cont[:, list(_MOVE_ACTION_INDICES)],
        target_cont[:, list(_MOVE_ACTION_INDICES)],
    )
    aim_loss = torch.nn.functional.mse_loss(
        pred_cont[:, _AIM_ACTION_INDEX],
        target_cont[:, _AIM_ACTION_INDEX],
    )
    fire_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, _PRIMARY_FIRE_BINARY_INDEX],
        target_binary[:, _PRIMARY_FIRE_BINARY_INDEX],
    )
    target_loss = obs.new_tensor(0.0)
    target_acc = obs.new_tensor(0.0)
    target_count = obs.new_tensor(0.0)
    if cfg.target_selection_dim > 0 and float(options.get("target_selection_coef", 0.0)) > 0.0:
        target_loss, target_acc, target_count = target_selection_aux_loss_and_accuracy(
            target_selection_logits,
            obs,
            cfg,
        )
    loss = (
        float(options.get("movement_coef", 1.0)) * move_loss
        + float(options.get("aim_coef", 1.0)) * aim_loss
        + float(options.get("fire_coef", 1.0)) * fire_loss
        + float(options.get("target_selection_coef", 0.0)) * target_loss
    )
    return loss, {
        "move_loss": move_loss.detach(),
        "aim_loss": aim_loss.detach(),
        "fire_loss": fire_loss.detach(),
        "target_selection_loss": target_loss.detach(),
        "target_selection_accuracy": target_acc.detach(),
        "target_selection_count": target_count.detach(),
    }


def _cpp_bot_targets(env: gym.Env, cfg: MappoConfig, bot_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    sim = getattr(env, "_sim", None)
    slots = tuple(getattr(env, "_own_slots", (0, 1, 2)))
    if sim is None:
        raise RuntimeError("cpp teacher requires a reset Phase4-style env with _sim")
    cont = torch.zeros(len(slots), cfg.continuous_action_dim, dtype=torch.float32)
    binary = torch.zeros(len(slots), cfg.binary_action_dim, dtype=torch.float32)
    for row, slot in enumerate(slots):
        scripted = _cpp.scripted_bot_action(sim, int(slot), bot_name)
        move_sign = -1.0 if int(slot) >= 3 else 1.0
        if cfg.continuous_action_dim >= 1:
            cont[row, 0] = float(move_sign * scripted.move_x)
        if cfg.continuous_action_dim >= 2:
            cont[row, 1] = float(move_sign * scripted.move_y)
        if cfg.continuous_action_dim > _AIM_ACTION_INDEX:
            cont[row, _AIM_ACTION_INDEX] = float(scripted.aim_delta / _AIM_DELTA_LIMIT)
        if cfg.binary_action_dim > 0:
            binary[row, 0] = 1.0 if scripted.primary_fire else 0.0
        if cfg.binary_action_dim > 1:
            binary[row, 1] = 1.0 if scripted.ability_1 else 0.0
        if cfg.binary_action_dim > 2:
            binary[row, 2] = 1.0 if scripted.ability_2 else 0.0
    return cont.clamp(-1.0, 1.0), binary.clamp(0.0, 1.0)


def _targets_for_teacher(
    env: gym.Env,
    obs_t: torch.Tensor,
    cfg: MappoConfig,
    teacher: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher in ("scripted_objective_focus_fire", "actor_obs_scripted"):
        return scripted_objective_focus_fire_targets(obs_t, cfg)
    if teacher == "multi_enemy_visible":
        return multi_enemy_visible_targets(obs_t, cfg)
    if teacher.startswith("cpp_"):
        return _cpp_bot_targets(env, cfg, teacher.removeprefix("cpp_"))
    raise ValueError(f"unknown full-env rehearsal teacher {teacher!r}")


def _collect_scripted_batch(
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    batch_size: int,
    seed: int,
    teacher: str = "scripted_objective_focus_fire",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[np.ndarray] = []
    target_cont_rows: list[np.ndarray] = []
    target_binary_rows: list[np.ndarray] = []
    env = env_fn()
    try:
        obs, _info = env.reset(seed=int(seed))
        max_steps = max(1, int(math.ceil(float(batch_size) / float(cfg.n_agents))))
        for idx in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            cont, binary = _targets_for_teacher(env, obs_t, cfg, teacher)
            pieces = [cont, binary]
            if cfg.target_action_dim > 0:
                pieces.append(torch.zeros(obs_t.shape[0], 1, dtype=obs_t.dtype))
            action = torch.cat(pieces, dim=-1).cpu().numpy()
            rows.append(obs.astype(np.float32, copy=True))
            target_cont_rows.append(cont.cpu().numpy().astype(np.float32, copy=True))
            target_binary_rows.append(binary.cpu().numpy().astype(np.float32, copy=True))
            obs, _reward, term, trunc, _info = env.step(action)
            if term or trunc:
                obs, _info = env.reset(seed=int(seed) + idx + 1)
    finally:
        env.close()
    flat = np.concatenate(rows, axis=0)[: int(batch_size)]
    flat_cont = np.concatenate(target_cont_rows, axis=0)[: int(batch_size)]
    flat_binary = np.concatenate(target_binary_rows, axis=0)[: int(batch_size)]
    return (
        torch.as_tensor(flat, dtype=torch.float32),
        torch.as_tensor(flat_cont, dtype=torch.float32),
        torch.as_tensor(flat_binary, dtype=torch.float32),
    )


def _policy_action_without_state_update(
    model: MappoActorCritic,
    obs: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_obs = obs.reshape(-1, model.cfg.obs_dim)
    flat_h = h.reshape(-1, model.cfg.gru_hidden)
    with torch.no_grad():
        action, h_next = model.greedy_action(flat_obs, flat_h)
    cont = action[:, : model.cfg.continuous_action_dim]
    binary_start = model.cfg.continuous_action_dim
    binary_end = binary_start + model.cfg.binary_action_dim
    binary = action[:, binary_start:binary_end]
    return (
        action.view(obs.shape[0], model.cfg.n_agents, model.cfg.action_dim),
        h_next.view(obs.shape[0], model.cfg.n_agents, model.cfg.gru_hidden),
        torch.cat((cont, binary), dim=-1),
    )


def _collect_policy_state_batch(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    *,
    batch_size: int,
    seed: int,
    teacher: str = "multi_enemy_visible",
) -> PolicyStateBatch:
    if teacher != "multi_enemy_visible":
        raise ValueError("closed-loop bridge only supports teacher='multi_enemy_visible'")
    cfg = model.cfg
    rows: list[np.ndarray] = []
    target_cont_rows: list[np.ndarray] = []
    target_binary_rows: list[np.ndarray] = []
    policy_cont_rows: list[np.ndarray] = []
    policy_binary_rows: list[np.ndarray] = []
    env = env_fn()
    was_training = model.training
    model.eval()
    try:
        obs_np, _info = env.reset(seed=int(seed))
        device = next(model.parameters()).device
        h = model.init_hidden(cfg.n_agents).view(1, cfg.n_agents, cfg.gru_hidden).to(device)
        max_steps = max(1, int(math.ceil(float(batch_size) / float(cfg.n_agents))))
        for idx in range(max_steps):
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).view(
                1, cfg.n_agents, cfg.obs_dim
            )
            flat_obs = obs.reshape(cfg.n_agents, cfg.obs_dim)
            cont, binary = _targets_for_teacher(env, flat_obs.cpu(), cfg, teacher)
            action_3d, h_next, policy_action = _policy_action_without_state_update(model, obs, h)
            action_np = action_3d.cpu().numpy()[0]
            rows.append(obs_np.astype(np.float32, copy=True))
            target_cont_rows.append(cont.cpu().numpy().astype(np.float32, copy=True))
            target_binary_rows.append(binary.cpu().numpy().astype(np.float32, copy=True))
            policy_cont_rows.append(
                policy_action[:, : cfg.continuous_action_dim].cpu().numpy().astype(np.float32, copy=True)
            )
            policy_binary_rows.append(
                policy_action[:, cfg.continuous_action_dim :].cpu().numpy().astype(np.float32, copy=True)
            )
            obs_np, _reward, term, trunc, _info = env.step(action_np)
            if term or trunc:
                obs_np, _info = env.reset(seed=int(seed) + idx + 1)
                h = model.init_hidden(cfg.n_agents).view(1, cfg.n_agents, cfg.gru_hidden).to(device)
            else:
                h = h_next
    finally:
        env.close()
        if was_training:
            model.train()
    flat = np.concatenate(rows, axis=0)[: int(batch_size)]
    flat_cont = np.concatenate(target_cont_rows, axis=0)[: int(batch_size)]
    flat_binary = np.concatenate(target_binary_rows, axis=0)[: int(batch_size)]
    flat_policy_cont = np.concatenate(policy_cont_rows, axis=0)[: int(batch_size)]
    flat_policy_binary = np.concatenate(policy_binary_rows, axis=0)[: int(batch_size)]
    return PolicyStateBatch(
        obs=torch.as_tensor(flat, dtype=torch.float32),
        target_cont=torch.as_tensor(flat_cont, dtype=torch.float32),
        target_binary=torch.as_tensor(flat_binary, dtype=torch.float32),
        policy_cont=torch.as_tensor(flat_policy_cont, dtype=torch.float32),
        policy_binary=torch.as_tensor(flat_policy_binary, dtype=torch.float32),
    )


def policy_state_agreement_metrics(batch: PolicyStateBatch) -> dict[str, float]:
    target_move = batch.target_cont[:, list(_MOVE_ACTION_INDICES)]
    policy_move = batch.policy_cont[:, list(_MOVE_ACTION_INDICES)]
    move_mse = torch.mean((policy_move - target_move) ** 2)
    target_aim = batch.target_cont[:, _AIM_ACTION_INDEX]
    policy_aim = batch.policy_cont[:, _AIM_ACTION_INDEX]
    aim_abs_error = torch.mean(torch.abs(policy_aim - target_aim))
    target_fire = batch.target_binary[:, _PRIMARY_FIRE_BINARY_INDEX] > 0.5
    policy_fire = batch.policy_binary[:, _PRIMARY_FIRE_BINARY_INDEX] > 0.5
    fire_agreement = (target_fire == policy_fire).to(torch.float32).mean()
    fire_positive_agreement = ((target_fire & policy_fire).to(torch.float32).sum() / target_fire.to(torch.float32).sum().clamp(min=1.0))
    return {
        "move_mse": float(move_mse.item()),
        "aim_abs_error": float(aim_abs_error.item()),
        "fire_accuracy": float(fire_agreement.item()),
        "fire_positive_recall": float(fire_positive_agreement.item()),
        "policy_fire_rate": float(policy_fire.to(torch.float32).mean().item()),
        "teacher_fire_rate": float(target_fire.to(torch.float32).mean().item()),
    }


def full_env_rehearsal_pretrain(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    config: dict[str, Any],
) -> dict[str, float]:
    steps = int(config.get("steps", 0))
    if steps <= 0:
        return {}
    batch_size = int(config.get("batch_size", 256))
    learning_rate = float(config.get("learning_rate", 1.0e-4))
    seed = int(config.get("seed", 0))
    teacher = str(config.get("teacher", "scripted_objective_focus_fire"))
    log_label = str(config.get("log_label", "phase4"))
    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    last: dict[str, float] = {}
    for step in range(1, steps + 1):
        batch, target_cont, target_binary = _collect_scripted_batch(
            env_fn,
            model.cfg,
            batch_size=batch_size,
            seed=seed + step,
            teacher=teacher,
        )
        loss, parts = full_env_rehearsal_loss(
            model,
            batch,
            config,
            target_cont=target_cont,
            target_binary=target_binary,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"full-env rehearsal loss is non-finite at step {step}")
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), model.cfg.max_grad_norm)
        opt.step()
        last = {
            "loss": float(loss.item()),
            "move_loss": float(parts["move_loss"].item()),
            "aim_loss": float(parts["aim_loss"].item()),
            "fire_loss": float(parts["fire_loss"].item()),
            "target_selection_loss": float(parts["target_selection_loss"].item()),
            "target_selection_accuracy": float(parts["target_selection_accuracy"].item()),
            "target_selection_count": float(parts["target_selection_count"].item()),
        }
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(
                f"[{log_label}/mappo] full_env_rehearsal step={step}/{steps} "
                f"loss={last['loss']:.4f} move={last['move_loss']:.4f} "
                f"aim={last['aim_loss']:.4f} fire={last['fire_loss']:.4f} "
                f"target={last['target_selection_loss']:.4f}",
                flush=True,
            )
    return last


def closed_loop_supervised_bridge_pretrain(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    config: dict[str, Any],
) -> dict[str, float]:
    closed_cfg = dict(config.get("closed_loop", {}))
    if not bool(closed_cfg.get("enabled", False)):
        return {}
    teacher = str(config.get("teacher", "multi_enemy_visible"))
    if teacher != "multi_enemy_visible":
        raise ValueError("closed-loop supervised bridge only supports multi_enemy_visible teacher")
    rounds = int(closed_cfg.get("rounds", 0))
    updates_per_round = int(closed_cfg.get("updates_per_round", 0))
    if rounds <= 0 or updates_per_round <= 0:
        raise ValueError("closed-loop supervised bridge requires positive bounded rounds and updates_per_round")
    batch_size = int(closed_cfg.get("batch_size", config.get("batch_size", 256)))
    learning_rate = float(closed_cfg.get("learning_rate", config.get("learning_rate", 1.0e-4)))
    seed = int(closed_cfg.get("seed", config.get("seed", 0)))
    log_label = str(config.get("log_label", "phase4"))
    opt = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    last: dict[str, float] = {}
    diagnostics: list[dict[str, float]] = []
    device = next(model.parameters()).device
    for round_idx in range(1, rounds + 1):
        batch = _collect_policy_state_batch(
            model,
            env_fn,
            batch_size=batch_size,
            seed=seed + round_idx,
            teacher=teacher,
        )
        agreement = policy_state_agreement_metrics(batch)
        diagnostics.append({"round": float(round_idx), **agreement})
        obs = batch.obs.to(device=device)
        target_cont = batch.target_cont.to(device=device)
        target_binary = batch.target_binary.to(device=device)
        for update_idx in range(1, updates_per_round + 1):
            loss, parts = full_env_rehearsal_loss(
                model,
                obs,
                config,
                target_cont=target_cont,
                target_binary=target_binary,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "closed-loop supervised bridge loss is non-finite at "
                    f"round {round_idx} update {update_idx}"
                )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), model.cfg.max_grad_norm)
            opt.step()
            last = {
                "loss": float(loss.item()),
                "move_loss": float(parts["move_loss"].item()),
                "aim_loss": float(parts["aim_loss"].item()),
                "fire_loss": float(parts["fire_loss"].item()),
                "target_selection_loss": float(parts["target_selection_loss"].item()),
                "target_selection_accuracy": float(parts["target_selection_accuracy"].item()),
                "target_selection_count": float(parts["target_selection_count"].item()),
                "round": float(round_idx),
                "update": float(update_idx),
                **{f"agreement_{k}": float(v) for k, v in agreement.items()},
            }
        print(
            f"[{log_label}/mappo] closed_loop_supervised_bridge "
            f"round={round_idx}/{rounds} updates={updates_per_round} "
            f"loss={last['loss']:.4f} move={last['move_loss']:.4f} "
            f"aim={last['aim_loss']:.4f} fire={last['fire_loss']:.4f} "
            f"fire_acc={agreement['fire_accuracy']:.3f} "
            f"policy_fire={agreement['policy_fire_rate']:.3f} "
            f"teacher_fire={agreement['teacher_fire_rate']:.3f}",
            flush=True,
        )
    if diagnostics:
        final_diag = diagnostics[-1]
        last.update({f"final_agreement_{k}": float(v) for k, v in final_diag.items() if k != "round"})
    return last


def run_full_env_rehearsal_gate(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    *,
    gate: dict[str, Any],
    output_dir: Path,
    seed: int,
    checkpoint_path: Path,
) -> FullEnvRehearsalGate:
    gate_cfg = dict(gate or {})
    episodes = int(gate_cfg.get("episodes", 50))
    min_hit_fire = float(gate_cfg.get("min_team_a_hit_fire", 0.04))
    min_visible_fire_rate = float(gate_cfg.get("min_team_a_visible_fire_rate", 0.0))
    min_on_point = float(gate_cfg.get("min_objective_on_point", 0.25))
    min_score_a = float(gate_cfg.get("min_mean_score_a", 0.0))
    max_losses = int(gate_cfg.get("max_losses", 49))
    output_name = str(gate_cfg.get("output", "full_env_rehearsal_gate.json"))
    stats = evaluate_mappo(model, env_fn, episodes=episodes, seed=int(seed))
    objective_on_point = float(stats.mean_majority_on_point_seconds_a) / max(
        1.0, float(stats.mean_final_tick) / 30.0
    )
    passed = (
        float(stats.team_a_hit_fire) >= min_hit_fire
        and float(stats.team_a_visible_fire_rate) >= min_visible_fire_rate
        and objective_on_point >= min_on_point
        and float(stats.mean_team_a_score) >= min_score_a
        and int(stats.losses) <= max_losses
    )
    metrics = {
        "team_a_hit_fire": float(stats.team_a_hit_fire),
        "team_a_visible_fire_rate": float(stats.team_a_visible_fire_rate),
        "objective_on_point": objective_on_point,
        "losses": float(stats.losses),
        "wins": float(stats.wins),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
    }
    thresholds = {
        "min_team_a_hit_fire": min_hit_fire,
        "min_team_a_visible_fire_rate": min_visible_fire_rate,
        "min_objective_on_point": min_on_point,
        "min_mean_score_a": min_score_a,
        "max_losses": float(max_losses),
    }
    path = output_dir / output_name
    payload = {
        "status": "PASSED" if passed else "NOT_REACHED",
        "reason": "full_env_rehearsal_pre_ppo_gate",
        "checkpoint_path": str(checkpoint_path),
        "episodes": episodes,
        "metrics": metrics,
        "thresholds": thresholds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return FullEnvRehearsalGate(
        status=str(payload["status"]),
        passed=passed,
        path=path,
        metrics=metrics,
        thresholds=thresholds,
    )
