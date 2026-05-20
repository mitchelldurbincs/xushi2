from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_model import MappoActorCritic, MappoConfig, mode_aux_loss_and_accuracy
from envs import make_mappo_match_env
from xushi2.obs_manifest import actor_field_slice

_MOVE_ACTION_INDICES = (0, 1)
_AIM_ACTION_INDEX = 2
_PRIMARY_FIRE_BINARY_INDEX = 0


@dataclass(frozen=True)
class CompositionDiagnostics:
    objective_on_point: float
    objective_losses: int
    combat_kills: float
    full_hit_fire: float
    full_aim_error: float
    passed: bool
    metrics: dict[str, float]


def load_frozen_mappo_teacher(checkpoint_path: str | Path) -> MappoActorCritic:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg_raw = dict(raw.get("config", {}).get("mappo", {}))
    if not ckpt_cfg_raw:
        raise ValueError(f"checkpoint {checkpoint_path} does not contain config.mappo")
    model = MappoActorCritic(MappoConfig(**ckpt_cfg_raw))
    model.load_state_dict(raw["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _assert_teacher_compatible(
    teacher: MappoActorCritic,
    student_cfg: MappoConfig,
    *,
    label: str,
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
    mismatches = {k: v for k, v in fields.items() if v[0] != v[1]}
    if mismatches:
        raise ValueError(f"{label} teacher is incompatible with student: {mismatches}")
    if student_cfg.continuous_action_dim <= _AIM_ACTION_INDEX:
        raise ValueError("composition rehearsal requires move_x, move_y, and aim_delta rows")
    if student_cfg.binary_action_dim <= _PRIMARY_FIRE_BINARY_INDEX:
        raise ValueError("composition rehearsal requires a primary_fire binary head")


@torch.no_grad()
def _teacher_policy_targets(
    teacher: MappoActorCritic,
    obs: torch.Tensor,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, h_next = teacher.actor_head_features(obs, h)
    mean, logits, _target_selection_logits = teacher.policy_heads_from_features(obs, features)
    cont_mean = torch.tanh(mean)
    binary_prob = torch.sigmoid(teacher.masked_binary_logits(obs, logits))
    greedy_binary = (binary_prob >= 0.5).to(obs.dtype)
    action = torch.cat((cont_mean, greedy_binary), dim=-1)
    if teacher.cfg.target_action_dim > 0:
        target_logits = teacher.actor_target_head(features)
        target_logits = teacher._masked_target_logits(target_logits, teacher._target_mask(obs))
        action = torch.cat(
            (action, target_logits.argmax(dim=-1).to(obs.dtype).unsqueeze(-1)),
            dim=-1,
        )
    return cont_mean, binary_prob, action, h_next.detach()


def _collect_teacher_sequence(
    env_fn: Callable[[], gym.Env],
    teacher: MappoActorCritic,
    *,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = teacher.cfg
    obs_parts: list[np.ndarray] = []
    cont_parts: list[np.ndarray] = []
    binary_parts: list[np.ndarray] = []
    max_decisions = max(1, int(np.ceil(float(batch_size) / float(cfg.n_agents))))
    env = env_fn()
    try:
        obs, _info = env.reset(seed=int(seed))
        h = teacher.init_hidden(cfg.n_agents)
        device = next(teacher.parameters()).device
        for idx in range(max_decisions):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            cont, binary, action, h = _teacher_policy_targets(teacher, obs_tensor, h)
            obs_parts.append(obs.astype(np.float32, copy=True))
            cont_parts.append(cont.cpu().numpy().astype(np.float32, copy=True))
            binary_parts.append(binary.cpu().numpy().astype(np.float32, copy=True))
            obs, _reward, term, trunc, _info = env.step(action.cpu().numpy())
            if term or trunc:
                obs, _info = env.reset(seed=int(seed) + idx + 1)
                h = teacher.init_hidden(cfg.n_agents)
    finally:
        env.close()
    return (
        torch.as_tensor(np.stack(obs_parts, axis=0), dtype=torch.float32),
        torch.as_tensor(np.stack(cont_parts, axis=0), dtype=torch.float32),
        torch.as_tensor(np.stack(binary_parts, axis=0), dtype=torch.float32),
    )


def composition_rehearsal_losses(
    student_model: MappoActorCritic,
    objective_obs_seq: torch.Tensor,
    objective_teacher_cont_seq: torch.Tensor,
    combat_obs_seq: torch.Tensor,
    combat_teacher_cont_seq: torch.Tensor,
    combat_teacher_binary_seq: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = student_model.cfg
    device = next(student_model.parameters()).device
    objective_obs_seq = objective_obs_seq.to(device)
    objective_teacher_cont_seq = objective_teacher_cont_seq.to(device)
    combat_obs_seq = combat_obs_seq.to(device)
    combat_teacher_cont_seq = combat_teacher_cont_seq.to(device)
    combat_teacher_binary_seq = combat_teacher_binary_seq.to(device)

    move_losses: list[torch.Tensor] = []
    aim_losses: list[torch.Tensor] = []
    fire_losses: list[torch.Tensor] = []
    mode_losses: list[torch.Tensor] = []
    mode_accs: list[torch.Tensor] = []

    h = student_model.init_hidden(cfg.n_agents).to(device)
    for t in range(objective_obs_seq.shape[0]):
        features, h = student_model.actor_head_features(objective_obs_seq[t], h)
        mean, _logits, _target_selection_logits = student_model.policy_heads_from_features(
            objective_obs_seq[t], features
        )
        mode_logits = student_model.mode_logits_from_features(features)
        pred_cont = torch.tanh(mean)
        move_losses.append(
            torch.nn.functional.mse_loss(
                pred_cont[:, list(_MOVE_ACTION_INDICES)],
                objective_teacher_cont_seq[t, :, list(_MOVE_ACTION_INDICES)],
            )
        )
        objective_mode = torch.zeros(cfg.n_agents, dtype=torch.long, device=device)
        mode_loss, mode_acc, _mode_count = mode_aux_loss_and_accuracy(
            mode_logits, objective_obs_seq[t], cfg, labels=objective_mode
        )
        mode_losses.append(mode_loss)
        mode_accs.append(mode_acc)

    h = student_model.init_hidden(cfg.n_agents).to(device)
    for t in range(combat_obs_seq.shape[0]):
        features, h = student_model.actor_head_features(combat_obs_seq[t], h)
        mean, logits, _target_selection_logits = student_model.policy_heads_from_features(
            combat_obs_seq[t], features
        )
        mode_logits = student_model.mode_logits_from_features(features)
        pred_cont = torch.tanh(mean)
        aim_losses.append(
            torch.nn.functional.mse_loss(
                pred_cont[:, _AIM_ACTION_INDEX],
                combat_teacher_cont_seq[t, :, _AIM_ACTION_INDEX],
            )
        )
        fire_losses.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, _PRIMARY_FIRE_BINARY_INDEX],
                combat_teacher_binary_seq[t, :, _PRIMARY_FIRE_BINARY_INDEX],
            )
        )
        combat_mode = torch.ones(cfg.n_agents, dtype=torch.long, device=device)
        mode_loss, mode_acc, _mode_count = mode_aux_loss_and_accuracy(
            mode_logits, combat_obs_seq[t], cfg, labels=combat_mode
        )
        mode_losses.append(mode_loss)
        mode_accs.append(mode_acc)

    move_loss = torch.stack(move_losses).mean()
    aim_loss = torch.stack(aim_losses).mean()
    fire_loss = torch.stack(fire_losses).mean()
    mode_loss = torch.stack(mode_losses).mean()
    mode_acc = torch.stack(mode_accs).mean()
    loss = move_loss + aim_loss + fire_loss + (
        cfg.mode_aux_coef * mode_loss if cfg.mode_gated_combat else 0.0
    )
    return loss, {
        "move_loss": move_loss.detach(),
        "aim_loss": aim_loss.detach(),
        "fire_loss": fire_loss.detach(),
        "mode_loss": mode_loss.detach(),
        "mode_accuracy": mode_acc.detach(),
    }


def composition_rehearsal_pretrain(
    student_model: MappoActorCritic,
    objective_teacher: MappoActorCritic,
    combat_teacher: MappoActorCritic,
    objective_env: Callable[[], gym.Env],
    combat_env: Callable[[], gym.Env],
    config: dict[str, Any],
) -> dict[str, float]:
    cfg = student_model.cfg
    _assert_teacher_compatible(objective_teacher, cfg, label="objective")
    _assert_teacher_compatible(combat_teacher, cfg, label="combat")
    device = next(student_model.parameters()).device
    objective_teacher.to(device)
    combat_teacher.to(device)
    steps = int(config.get("steps", 1000))
    if steps <= 0:
        return {}
    objective_batch_size = int(config.get("objective_batch_size", 256))
    combat_batch_size = int(config.get("combat_batch_size", 256))
    seed = int(config.get("seed", 0))
    learning_rate = float(config.get("learning_rate", 1.0e-3))
    log_label = str(config.get("log_label", "phase4"))
    opt = torch.optim.Adam(
        [param for param in student_model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    last_metrics: dict[str, float] = {}
    for step in range(1, steps + 1):
        objective_obs, objective_cont, _objective_binary = _collect_teacher_sequence(
            objective_env,
            objective_teacher,
            batch_size=objective_batch_size,
            seed=seed + step,
        )
        combat_obs, combat_cont, combat_binary = _collect_teacher_sequence(
            combat_env,
            combat_teacher,
            batch_size=combat_batch_size,
            seed=seed + 1_000_000 + step,
        )
        loss, parts = composition_rehearsal_losses(
            student_model,
            objective_obs,
            objective_cont,
            combat_obs,
            combat_cont,
            combat_binary,
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student_model.parameters(), cfg.max_grad_norm)
        opt.step()
        last_metrics = {
            "loss": float(loss.item()),
            "move_loss": float(parts["move_loss"].item()),
            "aim_loss": float(parts["aim_loss"].item()),
            "fire_loss": float(parts["fire_loss"].item()),
            "mode_loss": float(parts["mode_loss"].item()),
            "mode_accuracy": float(parts["mode_accuracy"].item()),
        }
        if step == 1 or step == steps or step % max(1, steps // 5) == 0:
            print(
                f"[{log_label}/mappo] composition_pretrain step={step}/{steps} "
                f"loss={last_metrics['loss']:.4f} "
                f"move_loss={last_metrics['move_loss']:.4f} "
                f"aim_loss={last_metrics['aim_loss']:.4f} "
                f"fire_loss={last_metrics['fire_loss']:.4f} "
                f"mode_acc={last_metrics['mode_accuracy']:.3f}",
                flush=True,
            )
    return last_metrics


def build_phase4_env_fn_with_overrides(
    base_env_cfg: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> Callable[[], gym.Env]:
    env_cfg = copy.deepcopy(base_env_cfg)
    for key, value in dict(overrides or {}).items():
        if key == "reward":
            merged = dict(env_cfg.get("reward", {}))
            merged.update(dict(value))
            env_cfg["reward"] = merged
        elif key == "sim":
            merged = dict(env_cfg.get("sim", {}))
            merged.update(dict(value))
            env_cfg["sim"] = merged
        elif key == "mini_game_config":
            merged = dict(env_cfg.get("mini_game_config", {}))
            merged.update(dict(value))
            env_cfg["mini_game_config"] = merged
        else:
            env_cfg[key] = value
    mini_game = env_cfg.get("mini_game")
    mini_game_cfg = dict(env_cfg.get("mini_game_config", {}))
    return lambda: make_mappo_match_env(
        sim_cfg=dict(env_cfg.get("sim", {})),
        opponent_bot=str(env_cfg.get("opponent_bot", "basic")),
        learner_team=str(env_cfg.get("learner_team", "A")),
        reward_cfg=dict(env_cfg.get("reward", {})),
        actor_obs=str(env_cfg.get("actor_obs", "flat")),
        fog_mode=str(env_cfg.get("fog_mode", "none")),
        visible_radius=float(env_cfg.get("visible_radius", 0.65)),
        map_randomization=dict(env_cfg.get("map_randomization", {})),
        mini_game=None if mini_game is None else str(mini_game),
        mini_game_cfg=mini_game_cfg,
        self_play=bool(dict(env_cfg.get("self_play", {})).get("enabled", False)),
        self_play_schedule=(
            dict(env_cfg.get("self_play_schedule", {}))
            if "self_play_schedule" in env_cfg
            else None
        ),
        snapshot_paths=tuple(str(p) for p in env_cfg.get("snapshot_paths", ())),
        snapshot_league=(
            dict(env_cfg.get("snapshot_league", {})) if "snapshot_league" in env_cfg else None
        ),
        target_slot=bool(env_cfg.get("target_slot", False)),
        n_agents=int(env_cfg.get("n_agents", env_cfg.get("team_size", 3))),
    )


def evaluate_objective_on_point(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    *,
    episodes: int,
    seed: int,
) -> float:
    device = next(model.parameters()).device
    self_on_point_slice = actor_field_slice("self_on_point")
    values: list[float] = []
    was_training = model.training
    model.eval()
    try:
        for episode in range(int(episodes)):
            env = env_fn()
            try:
                obs_np, _info = env.reset(seed=int(seed) + episode)
                h = model.init_hidden(model.cfg.n_agents)
                done = False
                while not done:
                    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                    values.extend(obs_np[:, self_on_point_slice].reshape(-1).tolist())
                    with torch.no_grad():
                        action, h = model.greedy_action(obs, h)
                    obs_np, _reward, term, trunc, _info = env.step(action.cpu().numpy())
                    done = bool(term or trunc)
            finally:
                env.close()
    finally:
        if was_training:
            model.train()
    return float(np.mean(values)) if values else 0.0


def run_composition_diagnostics(
    model: MappoActorCritic,
    *,
    objective_env_fn: Callable[[], gym.Env],
    combat_env_fn: Callable[[], gym.Env],
    full_env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
    gate: dict[str, Any] | None = None,
) -> CompositionDiagnostics:
    gate_cfg = dict(gate or {})
    min_objective_on_point = float(gate_cfg.get("objective_on_point_gate", 0.25))
    max_objective_losses = int(gate_cfg.get("objective_losses_gate", 0))
    min_combat_kills = float(gate_cfg.get("combat_kills_gate", 12.0))
    min_full_hit_fire = float(gate_cfg.get("hit_fire_gate", 0.02))
    max_full_aim_error = float(gate_cfg.get("aim_error_gate", 1.55))
    objective_stats = evaluate_mappo(model, objective_env_fn, episodes=episodes, seed=seed)
    objective_on_point = evaluate_objective_on_point(
        model, objective_env_fn, episodes=episodes, seed=seed + 10_000
    )
    combat_stats = evaluate_mappo(model, combat_env_fn, episodes=episodes, seed=seed + 20_000)
    full_stats = evaluate_mappo(model, full_env_fn, episodes=episodes, seed=seed + 30_000)
    passed = (
        objective_on_point > min_objective_on_point
        and objective_stats.losses <= max_objective_losses
        and combat_stats.mean_team_a_kills >= min_combat_kills
        and full_stats.team_a_hit_fire > min_full_hit_fire
        and full_stats.team_a_aim_error_rad < max_full_aim_error
    )
    metrics: dict[str, float] = {
        "objective_on_point": objective_on_point,
        "objective_losses": float(objective_stats.losses),
        "combat_kills": float(combat_stats.mean_team_a_kills),
        "full_hit_fire": float(full_stats.team_a_hit_fire),
        "full_aim_error": float(full_stats.team_a_aim_error_rad),
        "gate_objective_on_point": min_objective_on_point,
        "gate_objective_losses": float(max_objective_losses),
        "gate_combat_kills": min_combat_kills,
        "gate_hit_fire": min_full_hit_fire,
        "gate_aim_error": max_full_aim_error,
        "passed": float(passed),
    }
    metrics.update(
        {f"objective_{k}": float(v) for k, v in eval_stats_dict(objective_stats).items()}
    )
    metrics.update({f"combat_{k}": float(v) for k, v in eval_stats_dict(combat_stats).items()})
    metrics.update({f"full_{k}": float(v) for k, v in eval_stats_dict(full_stats).items()})
    return CompositionDiagnostics(
        objective_on_point=objective_on_point,
        objective_losses=int(objective_stats.losses),
        combat_kills=float(combat_stats.mean_team_a_kills),
        full_hit_fire=float(full_stats.team_a_hit_fire),
        full_aim_error=float(full_stats.team_a_aim_error_rad),
        passed=passed,
        metrics=metrics,
    )
