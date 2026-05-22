from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from train.composition_rehearsal import (
    build_mappo_env_fn_with_overrides,
    evaluate_objective_on_point,
    load_frozen_mappo_teacher,
)
from train.mappo_evaluate import evaluate_mappo
from train.mappo_model import MappoActorCritic, MappoConfig

_MOVE_ACTION_INDICES = (0, 1)
_AIM_ACTION_INDEX = 2
_PRIMARY_FIRE_BINARY_INDEX = 0


@dataclass(frozen=True)
class CapDuelDistillBatch:
    obs_seq: torch.Tensor
    teacher_cont_seq: torch.Tensor
    teacher_binary_prob_seq: torch.Tensor
    mask_seq: torch.Tensor
    done_seq: torch.Tensor


def _assert_teacher_compatible(
    teacher: MappoActorCritic,
    student_cfg: MappoConfig,
) -> None:
    fields = {
        "n_agents": (teacher.cfg.n_agents, student_cfg.n_agents),
        "obs_dim": (teacher.cfg.obs_dim, student_cfg.obs_dim),
        "action_dim": (teacher.cfg.action_dim, student_cfg.action_dim),
        "continuous_action_dim": (
            teacher.cfg.continuous_action_dim,
            student_cfg.continuous_action_dim,
        ),
        "binary_action_dim": (teacher.cfg.binary_action_dim, student_cfg.binary_action_dim),
    }
    mismatches = {key: value for key, value in fields.items() if value[0] != value[1]}
    if mismatches:
        raise ValueError(f"cap_duel_distill teacher is incompatible: {mismatches}")
    if student_cfg.continuous_action_dim <= _AIM_ACTION_INDEX:
        raise ValueError("cap_duel_distill requires an aim row in continuous actions")
    if student_cfg.binary_action_dim <= _PRIMARY_FIRE_BINARY_INDEX:
        raise ValueError("cap_duel_distill requires a primary_fire binary head")


@torch.no_grad()
def _teacher_policy_targets(
    teacher: MappoActorCritic,
    obs: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, h_next = teacher.actor_head_features(obs, h)
    mean, logits, _target_selection_logits = teacher.policy_heads_from_features(obs, features)
    logits = teacher.masked_binary_logits(obs, logits)
    cont = torch.tanh(mean)
    binary_prob = torch.sigmoid(logits)
    greedy_binary = (binary_prob >= 0.5).to(obs.dtype)
    action = torch.cat((cont, greedy_binary), dim=-1)
    if teacher.cfg.target_action_dim > 0:
        target_logits = teacher.actor_target_head(features)
        target_logits = teacher._masked_target_logits(target_logits, teacher._target_mask(obs))
        if target_logits is None:
            raise RuntimeError("target_action_dim requires target logits")
        action = torch.cat(
            (action, target_logits.argmax(dim=-1).to(obs.dtype).unsqueeze(-1)),
            dim=-1,
        )
    return cont, binary_prob, action, h_next.detach()


def _loss_mask_from_info(info: dict[str, Any], n_agents: int) -> np.ndarray:
    raw = info.get("loss_mask")
    if raw is None:
        return np.ones(n_agents, dtype=np.float32)
    mask = np.asarray(raw, dtype=np.float32).reshape(-1)
    if mask.shape != (n_agents,):
        raise ValueError(f"loss_mask must have shape ({n_agents},), got {mask.shape}")
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denom = weights.sum().clamp(min=1.0)
    return (values * weights).sum() / denom


def cap_duel_distill_loss(
    student_model: MappoActorCritic,
    batch: CapDuelDistillBatch,
    *,
    aim_coef: float = 1.0,
    fire_coef: float = 1.0,
    move_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = student_model.cfg
    device = next(student_model.parameters()).device
    obs_seq = batch.obs_seq.to(device)
    teacher_cont_seq = batch.teacher_cont_seq.to(device)
    teacher_binary_seq = batch.teacher_binary_prob_seq.to(device)
    mask_seq = batch.mask_seq.to(device)
    done_seq = batch.done_seq.to(device)

    aim_losses: list[torch.Tensor] = []
    fire_losses: list[torch.Tensor] = []
    move_losses: list[torch.Tensor] = []
    student_fire_probs: list[torch.Tensor] = []
    teacher_fire_probs: list[torch.Tensor] = []
    fire_agreements: list[torch.Tensor] = []

    h = student_model.init_hidden(cfg.n_agents).to(device)
    for t in range(obs_seq.shape[0]):
        obs_t = obs_seq[t]
        mask_t = mask_seq[t].reshape(cfg.n_agents)
        features, h = student_model.actor_head_features(obs_t, h)
        mean, logits, _target_selection_logits = student_model.policy_heads_from_features(
            obs_t, features
        )
        logits = student_model.masked_binary_logits(obs_t, logits)
        pred_cont = torch.tanh(mean)

        aim_err = (
            pred_cont[:, _AIM_ACTION_INDEX]
            - teacher_cont_seq[t, :, _AIM_ACTION_INDEX]
        ) ** 2
        fire_target = teacher_binary_seq[t, :, _PRIMARY_FIRE_BINARY_INDEX]
        fire_logits = logits[:, _PRIMARY_FIRE_BINARY_INDEX]
        fire_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fire_logits,
            fire_target,
            reduction="none",
        )
        aim_losses.append(_masked_mean(aim_err, mask_t))
        fire_losses.append(_masked_mean(fire_loss, mask_t))
        if move_coef > 0.0:
            move_err = (
                pred_cont[:, list(_MOVE_ACTION_INDICES)]
                - teacher_cont_seq[t, :, list(_MOVE_ACTION_INDICES)]
            ) ** 2
            move_losses.append(_masked_mean(move_err.mean(dim=-1), mask_t))
        with torch.no_grad():
            student_fire = torch.sigmoid(fire_logits)
            teacher_fire = fire_target
            student_fire_probs.append(_masked_mean(student_fire, mask_t))
            teacher_fire_probs.append(_masked_mean(teacher_fire, mask_t))
            fire_agreements.append(
                _masked_mean(
                    ((student_fire >= 0.5) == (teacher_fire >= 0.5)).to(obs_t.dtype),
                    mask_t,
                )
            )
        if bool(done_seq[t].item()):
            h = h * 0.0

    aim_loss = torch.stack(aim_losses).mean()
    fire_loss = torch.stack(fire_losses).mean()
    move_loss = (
        torch.stack(move_losses).mean()
        if move_losses
        else obs_seq.new_tensor(0.0, device=device)
    )
    loss = float(aim_coef) * aim_loss + float(fire_coef) * fire_loss + float(move_coef) * move_loss
    metrics = {
        "loss": loss.detach(),
        "aim_loss": aim_loss.detach(),
        "fire_loss": fire_loss.detach(),
        "move_loss": move_loss.detach(),
        "student_fire_prob": torch.stack(student_fire_probs).mean().detach(),
        "teacher_fire_prob": torch.stack(teacher_fire_probs).mean().detach(),
        "fire_agreement": torch.stack(fire_agreements).mean().detach(),
        "active_samples": mask_seq.sum().detach(),
    }
    return loss, metrics


class CapDuelDistillAnchor:
    def __init__(
        self,
        *,
        teacher: MappoActorCritic,
        env_fn: Callable[[], gym.Env],
        batch_size: int,
        coef: float,
        aim_coef: float,
        fire_coef: float,
        move_coef: float,
        every_updates: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("cap_duel_distill.batch_size must be positive")
        if every_updates <= 0:
            raise ValueError("cap_duel_distill.every_updates must be positive")
        for name, value in {
            "coef": coef,
            "aim_coef": aim_coef,
            "fire_coef": fire_coef,
            "move_coef": move_coef,
        }.items():
            if value < 0.0:
                raise ValueError(f"cap_duel_distill.{name} must be non-negative")
        self.teacher = teacher
        self.env_fn = env_fn
        self.batch_size = int(batch_size)
        self.coef = float(coef)
        self.aim_coef = float(aim_coef)
        self.fire_coef = float(fire_coef)
        self.move_coef = float(move_coef)
        self.every_updates = int(every_updates)
        self.seed = int(seed)

    def should_run(self, update_idx: int) -> bool:
        return int(update_idx) > 0 and int(update_idx) % self.every_updates == 0

    def collect_batch(self, *, update_idx: int, device: torch.device) -> CapDuelDistillBatch:
        self.teacher.to(device)
        cfg = self.teacher.cfg
        obs_parts: list[np.ndarray] = []
        cont_parts: list[np.ndarray] = []
        binary_parts: list[np.ndarray] = []
        mask_parts: list[np.ndarray] = []
        done_parts: list[float] = []

        active_samples = 0
        env = self.env_fn()
        try:
            reset_seed = self.seed + int(update_idx) * 10_000
            obs, info = env.reset(seed=reset_seed)
            h = self.teacher.init_hidden(cfg.n_agents).to(device)
            episode_idx = 0
            while active_samples < self.batch_size:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                cont, binary, action, h = _teacher_policy_targets(
                    self.teacher, obs_tensor, h
                )
                mask = _loss_mask_from_info(dict(info), cfg.n_agents)
                obs_parts.append(obs.astype(np.float32, copy=True))
                cont_parts.append(cont.cpu().numpy().astype(np.float32, copy=True))
                binary_parts.append(binary.cpu().numpy().astype(np.float32, copy=True))
                mask_parts.append(mask.astype(np.float32, copy=True))
                active_samples += int(np.count_nonzero(mask > 0.0))

                obs, _reward, term, trunc, info = env.step(action.cpu().numpy())
                done = bool(term or trunc)
                done_parts.append(1.0 if done else 0.0)
                if done:
                    episode_idx += 1
                    obs, info = env.reset(seed=reset_seed + episode_idx)
                    h = self.teacher.init_hidden(cfg.n_agents).to(device)
        finally:
            env.close()
        return CapDuelDistillBatch(
            obs_seq=torch.as_tensor(np.stack(obs_parts, axis=0), dtype=torch.float32),
            teacher_cont_seq=torch.as_tensor(np.stack(cont_parts, axis=0), dtype=torch.float32),
            teacher_binary_prob_seq=torch.as_tensor(
                np.stack(binary_parts, axis=0), dtype=torch.float32
            ),
            mask_seq=torch.as_tensor(np.stack(mask_parts, axis=0), dtype=torch.float32),
            done_seq=torch.as_tensor(np.asarray(done_parts, dtype=np.float32)),
        )

    def loss_for_model(
        self,
        student_model: MappoActorCritic,
        batch: CapDuelDistillBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = cap_duel_distill_loss(
            student_model,
            batch,
            aim_coef=self.aim_coef,
            fire_coef=self.fire_coef,
            move_coef=self.move_coef,
        )
        scaled_loss = self.coef * loss
        if not torch.isfinite(scaled_loss):
            raise RuntimeError("cap_duel_distill produced non-finite loss")
        metrics = dict(metrics)
        metrics["scaled_loss"] = scaled_loss.detach()
        metrics["updates"] = scaled_loss.new_tensor(1.0)
        return scaled_loss, metrics


def build_cap_duel_distill_anchor(
    *,
    base_env_cfg: dict[str, Any],
    student_cfg: MappoConfig,
    distill_cfg: dict[str, Any],
    seed: int,
) -> CapDuelDistillAnchor | None:
    if not bool(distill_cfg.get("enabled", False)):
        return None
    teacher_checkpoint = distill_cfg.get("teacher_checkpoint")
    if not teacher_checkpoint:
        raise ValueError("run.cap_duel_distill.teacher_checkpoint is required")
    teacher = load_frozen_mappo_teacher(Path(str(teacher_checkpoint)))
    _assert_teacher_compatible(teacher, student_cfg)
    env_overrides = dict(distill_cfg.get("env", {}))
    if env_overrides.get("mini_game") != "cap_duel":
        raise ValueError("run.cap_duel_distill.env.mini_game must be 'cap_duel'")
    env_fn = build_mappo_env_fn_with_overrides(base_env_cfg, env_overrides)
    return CapDuelDistillAnchor(
        teacher=teacher,
        env_fn=env_fn,
        batch_size=int(distill_cfg.get("batch_size", 256)),
        coef=float(distill_cfg.get("coef", 0.05)),
        aim_coef=float(distill_cfg.get("aim_coef", 1.0)),
        fire_coef=float(distill_cfg.get("fire_coef", 1.0)),
        move_coef=float(distill_cfg.get("move_coef", 0.0)),
        every_updates=int(distill_cfg.get("every_updates", 1)),
        seed=int(seed),
    )


def configure_cap_duel_distill_anchor(context: Any, trainer: Any) -> bool:
    distill_cfg = dict(context.run_cfg.get("cap_duel_distill", {}))
    anchor = build_cap_duel_distill_anchor(
        base_env_cfg=dict(context.ckpt_env_cfg),
        student_cfg=context.cfg,
        distill_cfg=distill_cfg,
        seed=int(context.seed_base) + 600_000,
    )
    trainer.set_cap_duel_distill_anchor(anchor)
    if anchor is None:
        return False
    print(
        "[phase4/mappo] cap_duel_distill enabled "
        f"teacher={distill_cfg.get('teacher_checkpoint')} "
        f"batch_size={anchor.batch_size} coef={anchor.coef:.4f}",
        flush=True,
    )
    return True


@torch.no_grad()
def run_cap_duel_distill_diagnostics(
    student_model: MappoActorCritic,
    *,
    anchor: CapDuelDistillAnchor,
    objective_env_fn: Callable[[], gym.Env],
    full_env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
) -> dict[str, float]:
    objective_stats = evaluate_mappo(
        student_model,
        objective_env_fn,
        episodes=int(episodes),
        seed=int(seed),
    )
    objective_on_point = evaluate_objective_on_point(
        student_model,
        objective_env_fn,
        episodes=int(episodes),
        seed=int(seed) + 10_000,
    )
    cap_duel_stats = evaluate_mappo(
        student_model,
        anchor.env_fn,
        episodes=int(episodes),
        seed=int(seed) + 20_000,
    )
    full_stats = evaluate_mappo(
        student_model,
        full_env_fn,
        episodes=int(episodes),
        seed=int(seed) + 30_000,
    )
    device = next(student_model.parameters()).device
    batch = anchor.collect_batch(update_idx=1, device=device)
    loss, metrics = cap_duel_distill_loss(
        student_model,
        batch,
        aim_coef=anchor.aim_coef,
        fire_coef=anchor.fire_coef,
        move_coef=anchor.move_coef,
    )
    return {
        "objective_on_point": float(objective_on_point),
        "objective_losses": float(objective_stats.losses),
        "cap_duel_kills": float(cap_duel_stats.mean_team_a_kills),
        "full_hit_fire": float(full_stats.team_a_hit_fire),
        "full_aim_error": float(full_stats.team_a_aim_error_rad),
        "distill_loss": float(loss.item()),
        "aim_mse": float(metrics["aim_loss"].item()),
        "fire_bce": float(metrics["fire_loss"].item()),
        "teacher_fire_prob": float(metrics["teacher_fire_prob"].item()),
        "student_fire_prob": float(metrics["student_fire_prob"].item()),
        "fire_agreement": float(metrics["fire_agreement"].item()),
    }
