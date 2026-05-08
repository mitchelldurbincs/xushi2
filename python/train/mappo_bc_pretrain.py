from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from train.mappo_model import MappoActorCritic, MappoConfig, _OWN_POSITION_SLICE
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


def _collect_walk_bc_sequence(
    env_fn: Callable[[], gym.Env],
    cfg: MappoConfig,
    *,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    max_decisions = max(1, int(np.ceil(float(batch_size) / float(cfg.n_agents))))
    env = env_fn()
    try:
        obs, _info = env.reset(seed=seed)
        for _ in range(max_decisions):
            obs_parts.append(obs.astype(np.float32, copy=True))
            target = _walk_to_objective_targets(
                torch.as_tensor(obs, dtype=torch.float32), cfg
            )
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
            mean, _log_std, logits, _target_logits, h = model.policy_outputs(
                obs_seq[t], h
            )
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
                        cfg.continuous_action_dim :
                        cfg.continuous_action_dim + cfg.binary_action_dim,
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


