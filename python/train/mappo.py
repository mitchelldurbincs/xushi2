"""Phase-4 recurrent MAPPO trainer.

This is the first CTDE training path: a shared recurrent actor consumes
per-agent actor observations while a centralized critic consumes the
team-level critic observation supplied by ``Phase4MappoEnv``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from train.phases import resolve_phase
from train.ppo_recurrent.losses import _masked_mean, action_logprob_and_entropy
from train.ppo_recurrent.lr_schedule import lr_for_update
from xushi2.entity_obs import (
    ENTITY_OBS_DIM,
    ENTITY_TOKEN_COUNT,
    ENTITY_TOKEN_DIM,
    entity_obs_self_position,
)
from xushi2.grid_obs import GRID_CHANNELS, GRID_SIZE
from xushi2.mappo_eval_gate import check_eval_gate
from xushi2.mappo_matrix_gate import check_matrix_gate
from xushi2.obs_manifest import actor_field_slice
from xushi2.snapshot_retention import SnapshotRetention
from xushi2.vector_env import make_xushi_vector_env

_LOG2 = 0.6931471805599453
_OWN_POSITION_SLICE = actor_field_slice("own_position")


@dataclass(frozen=True)
class MappoConfig:
    num_envs: int
    n_agents: int
    rollout_len: int
    obs_dim: int
    critic_obs_dim: int
    action_dim: int
    continuous_action_dim: int
    binary_action_dim: int
    embed_dim: int
    gru_hidden: int
    head_hidden: int
    action_log_std_init: float
    gamma: float
    gae_lambda: float
    clip_ratio: float
    value_clip_ratio: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    learning_rate: float
    num_epochs: int
    minibatch_size: int
    target_action_dim: int = 0
    value_per_agent: bool = False
    lr_schedule: str = "constant"
    lr_final_ratio: float = 1.0
    warmup_updates: int = 0
    value_normalization: bool = True
    torch_num_threads: int = 0
    vector_env: str = "sync"
    obs_encoder: str = "flat"
    entity_token_count: int = 0
    entity_token_dim: int = 0
    entity_num_heads: int = 1
    grid_channels: int = 0
    grid_size: int = 0
    agent_loss_mask: tuple[float, ...] = ()


@dataclass(frozen=True)
class MappoEvalStats:
    mean_reward: float
    episodes: int
    wins: int
    losses: int
    draws: int
    terminated: int
    truncated: int
    mean_final_tick: float
    mean_team_a_score: float
    mean_team_b_score: float
    mean_team_a_kills: float
    mean_team_b_kills: float


class MappoActorCritic(nn.Module):
    def __init__(self, cfg: MappoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor_embed: nn.Module | None
        self.actor_entity_encoder: nn.Module | None
        self.actor_grid_encoder: nn.Module | None = None
        self.actor_fusion: nn.Module | None = None
        if cfg.obs_encoder == "flat":
            self.actor_embed = nn.Sequential(
                nn.Linear(cfg.obs_dim, cfg.embed_dim),
                nn.ReLU(),
            )
            self.actor_entity_encoder = None
        elif cfg.obs_encoder == "entity_attention":
            if cfg.obs_dim != ENTITY_OBS_DIM:
                raise ValueError(
                    f"entity_attention obs_dim must be {ENTITY_OBS_DIM}, "
                    f"got {cfg.obs_dim}"
                )
            if (
                cfg.entity_token_count != ENTITY_TOKEN_COUNT
                or cfg.entity_token_dim != ENTITY_TOKEN_DIM
            ):
                raise ValueError(
                    "entity_attention token shape must match "
                    f"({ENTITY_TOKEN_COUNT}, {ENTITY_TOKEN_DIM})"
                )
            from train.entity_attention import EntityAttentionEncoder

            self.actor_embed = None
            self.actor_entity_encoder = EntityAttentionEncoder(
                entity_dim=cfg.entity_token_dim,
                embed_dim=cfg.embed_dim,
                num_heads=cfg.entity_num_heads,
                output_dim=cfg.embed_dim,
            )
        elif cfg.obs_encoder == "entity_attention_grid":
            base_obs_dim = (
                cfg.entity_token_count * cfg.entity_token_dim
                + cfg.entity_token_count
                + cfg.grid_channels * cfg.grid_size * cfg.grid_size
            )
            expected_obs_dim = base_obs_dim + (
                cfg.target_action_dim if cfg.target_action_dim > 0 else 0
            )
            if cfg.obs_dim != expected_obs_dim:
                raise ValueError(
                    f"entity_attention_grid obs_dim must be {expected_obs_dim}, "
                    f"got {cfg.obs_dim}"
                )
            if cfg.entity_token_count <= 0 or cfg.entity_token_dim != ENTITY_TOKEN_DIM:
                raise ValueError(
                    "entity_attention_grid token shape must match "
                    f"(positive, {ENTITY_TOKEN_DIM})"
                )
            if cfg.grid_channels != GRID_CHANNELS or cfg.grid_size != GRID_SIZE:
                raise ValueError(
                    "entity_attention_grid grid shape must match "
                    f"({GRID_CHANNELS}, {GRID_SIZE}, {GRID_SIZE})"
                )
            from train.entity_attention import EntityAttentionEncoder

            self.actor_embed = None
            self.actor_entity_encoder = EntityAttentionEncoder(
                entity_dim=cfg.entity_token_dim,
                embed_dim=cfg.embed_dim,
                num_heads=cfg.entity_num_heads,
                output_dim=cfg.embed_dim,
            )
            self.actor_grid_encoder = nn.Sequential(
                nn.Conv2d(cfg.grid_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * (cfg.grid_size // 2) * (cfg.grid_size // 2), cfg.embed_dim),
                nn.ReLU(),
            )
            self.actor_fusion = nn.Sequential(
                nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"unknown obs_encoder {cfg.obs_encoder!r}")
        self.actor_gru = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.actor_mean_head = nn.Linear(cfg.head_hidden, cfg.continuous_action_dim)
        self.actor_binary_head = nn.Linear(cfg.head_hidden, cfg.binary_action_dim)
        self.actor_target_head = (
            nn.Linear(cfg.head_hidden, cfg.target_action_dim)
            if cfg.target_action_dim > 0
            else None
        )
        self.log_std = nn.Parameter(
            torch.ones(cfg.continuous_action_dim) * cfg.action_log_std_init
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.critic_obs_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, 1),
        )

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(
            batch_size, self.cfg.gru_hidden, device=p.device, dtype=p.dtype
        )

    def policy_outputs(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        emb = self._actor_features(obs)
        h_next = self.actor_gru(emb, h)
        features = self.actor_body(h_next)
        mean = self.actor_mean_head(features)
        logits = self.actor_binary_head(features)
        target_logits = (
            self.actor_target_head(features)
            if self.actor_target_head is not None
            else None
        )
        return mean, self.log_std, logits, target_logits, h_next

    def _actor_features(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.obs_encoder == "flat":
            assert self.actor_embed is not None
            return self.actor_embed(obs)
        assert self.actor_entity_encoder is not None
        token_width = self.cfg.entity_token_count * self.cfg.entity_token_dim
        tokens = obs[:, :token_width].view(
            obs.shape[0], self.cfg.entity_token_count, self.cfg.entity_token_dim
        )
        mask = obs[:, token_width : token_width + self.cfg.entity_token_count] > 0.5
        features, _weights = self.actor_entity_encoder(tokens, mask)
        if self.cfg.obs_encoder == "entity_attention":
            return features

        grid_offset = token_width + self.cfg.entity_token_count
        grid_end = grid_offset + self.cfg.grid_channels * self.cfg.grid_size * self.cfg.grid_size
        grid = obs[:, grid_offset:grid_end].view(
            obs.shape[0],
            self.cfg.grid_channels,
            self.cfg.grid_size,
            self.cfg.grid_size,
        )
        assert self.actor_grid_encoder is not None
        assert self.actor_fusion is not None
        grid_features = self.actor_grid_encoder(grid)
        return self.actor_fusion(torch.cat((features, grid_features), dim=-1))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def _target_mask(self, obs: torch.Tensor) -> torch.Tensor | None:
        if self.cfg.target_action_dim <= 0:
            return None
        mask = obs[:, -self.cfg.target_action_dim :] > 0.5
        fallback = torch.zeros_like(mask)
        fallback[:, 0] = True
        return torch.where(mask.any(dim=-1, keepdim=True), mask, fallback)

    @staticmethod
    def _masked_target_logits(
        target_logits: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if target_logits is None:
            return None
        if target_mask is None:
            return target_logits
        return target_logits.masked_fill(~target_mask, -1.0e9)

    def sample_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, logits, target_logits, h_next = self.policy_outputs(obs, h)
        target_logits = self._masked_target_logits(target_logits, self._target_mask(obs))
        pieces: list[torch.Tensor] = []
        logprob = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        if self.cfg.continuous_action_dim > 0:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            u = dist.rsample()
            cont = torch.tanh(u)
            correction = 2.0 * (_LOG2 - u - torch.nn.functional.softplus(-2.0 * u))
            logprob = logprob + dist.log_prob(u).sum(-1) - correction.sum(-1)
            pieces.append(cont)
        if self.cfg.binary_action_dim > 0:
            binary_dist = torch.distributions.Bernoulli(logits=logits)
            binary = binary_dist.sample()
            logprob = logprob + binary_dist.log_prob(binary).sum(-1)
            pieces.append(binary)
        if self.cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target = target_dist.sample()
            logprob = logprob + target_dist.log_prob(target)
            pieces.append(target.to(obs.dtype).unsqueeze(-1))
        return torch.cat(pieces, dim=-1), logprob, h_next

    def greedy_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _log_std, logits, target_logits, h_next = self.policy_outputs(obs, h)
        target_logits = self._masked_target_logits(target_logits, self._target_mask(obs))
        pieces = [torch.tanh(mean), (logits >= 0.0).to(obs.dtype)]
        if self.cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            pieces.append(target_logits.argmax(dim=-1).to(obs.dtype).unsqueeze(-1))
        action = torch.cat(pieces, dim=-1)
        return action, h_next


class MappoRollout:
    def __init__(self, cfg: MappoConfig) -> None:
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        self.actor_obs = torch.zeros(N, A, L, cfg.obs_dim)
        if cfg.value_per_agent:
            self.critic_obs = torch.zeros(N, A, L, cfg.critic_obs_dim)
            self.value = torch.zeros(N, A, L)
            self.advantages = torch.zeros(N, A, L)
            self.returns = torch.zeros(N, A, L)
            self.last_value = torch.zeros(N, A)
        else:
            self.critic_obs = torch.zeros(N, L, cfg.critic_obs_dim)
            self.value = torch.zeros(N, L)
            self.advantages = torch.zeros(N, L)
            self.returns = torch.zeros(N, L)
            self.last_value = torch.zeros(N)
        self.action = torch.zeros(N, A, L, cfg.action_dim)
        self.logprob = torch.zeros(N, A, L)
        self.reward = torch.zeros(N, A, L)
        self.done = torch.zeros(N, L)
        self.h_init = torch.zeros(N, A, L, cfg.gru_hidden)
        self.last_done = torch.zeros(N)
        raw_mask = cfg.agent_loss_mask or tuple(1.0 for _ in range(A))
        self.agent_loss_mask = torch.as_tensor(
            raw_mask, dtype=torch.float32
        ).view(1, A, 1).expand(N, A, L).clone()

    def compute_gae(self, cfg: MappoConfig) -> None:
        if cfg.value_per_agent:
            last_gae = torch.zeros(cfg.num_envs, cfg.n_agents)
            for t in reversed(range(cfg.rollout_len)):
                if t == cfg.rollout_len - 1:
                    next_value = self.last_value
                    next_nonterminal = (1.0 - self.last_done).view(cfg.num_envs, 1)
                else:
                    next_value = self.value[:, :, t + 1]
                    next_nonterminal = (1.0 - self.done[:, t]).view(cfg.num_envs, 1)
                delta = (
                    self.reward[:, :, t]
                    + cfg.gamma * next_value * next_nonterminal
                    - self.value[:, :, t]
                )
                last_gae = (
                    delta
                    + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                )
                self.advantages[:, :, t] = last_gae
            self.returns = self.advantages + self.value
            return

        active_count = self.agent_loss_mask.sum(dim=1).clamp(min=1.0)
        reward = (self.reward * self.agent_loss_mask).sum(dim=1) / active_count
        last_gae = torch.zeros(cfg.num_envs)
        for t in reversed(range(cfg.rollout_len)):
            if t == cfg.rollout_len - 1:
                next_value = self.last_value
                next_nonterminal = 1.0 - self.last_done
            else:
                next_value = self.value[:, t + 1]
                next_nonterminal = 1.0 - self.done[:, t]
            delta = (
                reward[:, t]
                + cfg.gamma * next_value * next_nonterminal
                - self.value[:, t]
            )
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            self.advantages[:, t] = last_gae
        self.returns = self.advantages + self.value


class MappoTrainer:
    def __init__(
        self, env_fn: Callable[[], gym.Env], cfg: MappoConfig, seed: int
    ) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        if cfg.torch_num_threads > 0:
            torch.set_num_threads(cfg.torch_num_threads)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.vec_env = make_xushi_vector_env(
            [env_fn for _ in range(cfg.num_envs)],
            critic_obs_dim=(
                cfg.critic_obs_dim * cfg.n_agents
                if cfg.value_per_agent
                else cfg.critic_obs_dim
            ),
            seed_base=self.seed,
            backend=cfg.vector_env,
        )
        obs, _critic_obs, _infos = self.vec_env.reset(seed=self.seed)
        self.last_obs = torch.as_tensor(obs, dtype=torch.float32)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = MappoActorCritic(cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.current_learning_rate = cfg.learning_rate
        self.h = self.model.init_hidden(cfg.num_envs * cfg.n_agents).view(
            cfg.num_envs, cfg.n_agents, cfg.gru_hidden
        )
        self._sampling_rng_state = torch.get_rng_state()
        self._update_counter = 0
        self._actor_params: list[torch.nn.Parameter] = []
        self._critic_params: list[torch.nn.Parameter] = []
        self._trunk_params: list[torch.nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if name.startswith(
                (
                    "actor_body",
                    "actor_mean_head",
                    "actor_binary_head",
                    "actor_target_head",
                )
            ):
                self._actor_params.append(p)
            elif name == "log_std":
                self._actor_params.append(p)
            elif name.startswith("critic"):
                self._critic_params.append(p)
            else:
                self._trunk_params.append(p)

    def close(self) -> None:
        self.vec_env.close()

    def set_learning_rate(self, lr: float) -> None:
        self.current_learning_rate = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_learning_rate

    @staticmethod
    def _group_grad_norm(params: list[torch.nn.Parameter]) -> float:
        total_sq = 0.0
        for p in params:
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum().item())
        return float(total_sq ** 0.5)

    def _critic_obs(self) -> torch.Tensor:
        critic_obs = torch.as_tensor(self.vec_env.critic_obs(), dtype=torch.float32)
        if self.cfg.value_per_agent:
            return critic_obs.view(
                self.cfg.num_envs, self.cfg.n_agents, self.cfg.critic_obs_dim
            )
        return critic_obs

    def collect_rollout(self) -> MappoRollout:
        cfg = self.cfg
        rollout = MappoRollout(cfg)
        obs = self.last_obs
        h = self.h
        for t in range(cfg.rollout_len):
            critic_obs = self._critic_obs()
            flat_obs = obs.reshape(cfg.num_envs * cfg.n_agents, cfg.obs_dim)
            flat_h = h.reshape(cfg.num_envs * cfg.n_agents, cfg.gru_hidden)
            with torch.no_grad():
                prev_rng = torch.get_rng_state()
                torch.set_rng_state(self._sampling_rng_state)
                try:
                    action, logprob, h_next = self.model.sample_action(flat_obs, flat_h)
                    self._sampling_rng_state = torch.get_rng_state()
                finally:
                    torch.set_rng_state(prev_rng)
                if cfg.value_per_agent:
                    value = self.model.value(
                        critic_obs.reshape(cfg.num_envs * cfg.n_agents, cfg.critic_obs_dim)
                    ).view(cfg.num_envs, cfg.n_agents)
                else:
                    value = self.model.value(critic_obs)
            action_3d = action.view(cfg.num_envs, cfg.n_agents, cfg.action_dim)
            action_np = action_3d.cpu().numpy()
            next_obs_np, reward_np, terminated, truncated, _next_critic_obs, _infos = (
                self.vec_env.step(action_np)
            )
            done_np = np.logical_or(terminated, truncated)
            rollout.actor_obs[:, :, t] = obs
            if cfg.value_per_agent:
                rollout.critic_obs[:, :, t] = critic_obs
            else:
                rollout.critic_obs[:, t] = critic_obs
            rollout.action[:, :, t] = action_3d
            rollout.logprob[:, :, t] = logprob.view(cfg.num_envs, cfg.n_agents)
            rollout.reward[:, :, t] = torch.as_tensor(reward_np, dtype=torch.float32)
            rollout.agent_loss_mask[:, :, t] = self._step_loss_mask(_infos)
            if cfg.value_per_agent:
                rollout.value[:, :, t] = value
            else:
                rollout.value[:, t] = value
            rollout.done[:, t] = torch.as_tensor(done_np, dtype=torch.float32)
            h = h_next.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
            rollout.h_init[:, :, t] = flat_h.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
            for e, done in enumerate(done_np):
                if bool(done):
                    h[e] = 0.0
            obs = torch.as_tensor(next_obs_np, dtype=torch.float32)
        with torch.no_grad():
            critic_obs = self._critic_obs()
            if cfg.value_per_agent:
                rollout.last_value = self.model.value(
                    critic_obs.reshape(cfg.num_envs * cfg.n_agents, cfg.critic_obs_dim)
                ).view(cfg.num_envs, cfg.n_agents)
            else:
                rollout.last_value = self.model.value(critic_obs)
        rollout.last_done = rollout.done[:, -1].clone()
        self.last_obs = obs
        self.h = h
        return rollout

    def _step_loss_mask(self, infos: list[dict]) -> torch.Tensor:
        cfg = self.cfg
        static = torch.as_tensor(cfg.agent_loss_mask, dtype=torch.float32)
        masks = torch.zeros(cfg.num_envs, cfg.n_agents, dtype=torch.float32)
        for env_idx, info in enumerate(infos):
            raw = info.get("loss_mask")
            if raw is None:
                final_info = info.get("final_info")
                if isinstance(final_info, dict):
                    raw = final_info.get("loss_mask")
            if raw is None:
                masks[env_idx] = static
                continue
            mask = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            if mask.numel() != cfg.n_agents:
                raise ValueError(
                    f"env loss_mask length must be {cfg.n_agents}, got {mask.numel()}"
                )
            mask = torch.clamp(mask, min=0.0) * static
            if float(mask.sum().item()) <= 0.0:
                raise ValueError("env loss_mask must leave at least one active agent")
            masks[env_idx] = mask
        return masks

    def update(self, rollout: MappoRollout) -> dict[str, float]:
        cfg = self.cfg
        rollout.compute_gae(cfg)
        rollout_metrics = self._rollout_metrics(rollout)
        if cfg.value_normalization:
            if cfg.value_per_agent:
                valid_agent = rollout.agent_loss_mask.expand_as(rollout.returns)
                ret_mean_t = _masked_mean(rollout.returns, valid_agent)
                ret_std_t = _masked_mean(
                    (rollout.returns - ret_mean_t) ** 2,
                    valid_agent,
                ).sqrt().clamp(min=1e-6)
                ret_mean = float(ret_mean_t.item())
                ret_std = float(ret_std_t.item())
            else:
                ret_mean = float(rollout.returns.mean().item())
                ret_std = float(
                    rollout.returns.std(unbiased=False).clamp(min=1e-6).item()
                )
        else:
            ret_mean, ret_std = 0.0, 1.0

        losses = []
        for _epoch in range(cfg.num_epochs):
            losses.append(self._update_full_rollout(rollout, ret_mean, ret_std))
        self._update_counter += 1
        metrics = {k: float(np.mean([m[k] for m in losses])) for k in losses[0]}
        metrics.update(rollout_metrics)
        return metrics

    def _rollout_metrics(self, rollout: MappoRollout) -> dict[str, float]:
        cfg = self.cfg
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
            own_pos = torch.as_tensor(
                own_pos_np, dtype=rollout.actor_obs.dtype, device=rollout.actor_obs.device
            )
            self_on_point = torch.zeros_like(own_pos[..., :1])
        else:
            own_pos = rollout.actor_obs[:, :, :, _OWN_POSITION_SLICE]
            self_on_point = rollout.actor_obs[:, :, :, self_on_point_slice]
        distance_to_objective = torch.linalg.vector_norm(own_pos, dim=-1)

        out = {
            "active_agent_fraction": float(agent_mask.mean().item()),
            "rollout_reward_mean": float(_masked_mean(reward, agent_mask).item()),
            "rollout_reward_std": float(
                _masked_mean(
                    (reward - _masked_mean(reward, agent_mask)) ** 2,
                    agent_mask,
                ).sqrt().item()
            ),
            "rollout_reward_min": float(reward[agent_mask > 0.0].min().item()),
            "rollout_reward_max": float(reward[agent_mask > 0.0].max().item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "advantage_min": float(advantages.min().item()),
            "advantage_max": float(advantages.max().item()),
            "return_mean": float(returns.mean().item()),
            "return_std": float(returns.std(unbiased=False).item()),
            "action_move_mag_mean": float(_masked_mean(move_mag, agent_mask).item()),
            "action_cont_mean": float(
                _masked_mean(cont, agent_mask.unsqueeze(-1).expand_as(cont)).item()
            ),
            "action_cont_std": float(
                _masked_mean(
                    (
                        cont
                        - _masked_mean(cont, agent_mask.unsqueeze(-1).expand_as(cont))
                    ) ** 2,
                    agent_mask.unsqueeze(-1).expand_as(cont),
                ).sqrt().item()
            ),
            "mean_distance_to_objective": float(
                _masked_mean(distance_to_objective, agent_mask).item()
            ),
            "self_on_point_fraction": float(
                _masked_mean(
                    self_on_point, agent_mask.unsqueeze(-1).expand_as(self_on_point)
                ).item()
            ),
        }
        if binary.numel() > 0:
            out["action_binary_mean"] = float(
                _masked_mean(binary, agent_mask.unsqueeze(-1).expand_as(binary)).item()
            )
        else:
            out["action_binary_mean"] = 0.0
        if target is not None and target.numel() > 0:
            out["action_target_slot_mean"] = float(
                _masked_mean(target, agent_mask.unsqueeze(-1).expand_as(target)).item()
            )
        return out

    def _action_logprob_and_entropy(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        binary_logits: torch.Tensor,
        target_logits: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        base_end = cfg.continuous_action_dim + cfg.binary_action_dim
        logp, ent = action_logprob_and_entropy(
            mean, log_std, binary_logits, action[:, :base_end]
        )
        if cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            target = action[:, base_end].long().clamp(0, cfg.target_action_dim - 1)
            dist = torch.distributions.Categorical(logits=target_logits)
            logp = logp + dist.log_prob(target)
            ent = ent + dist.entropy()
        return logp, ent

    def _update_full_rollout(
        self, rollout: MappoRollout, return_mean: float, return_std: float
    ) -> dict[str, float]:
        cfg = self.cfg
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        flat_h = rollout.h_init[:, :, 0].reshape(N * A, cfg.gru_hidden)
        logprobs, entropies = [], []
        h = flat_h
        for t in range(L):
            obs_t = rollout.actor_obs[:, :, t].reshape(N * A, cfg.obs_dim)
            mean, log_std, logits, target_logits, h = self.model.policy_outputs(obs_t, h)
            target_logits = self.model._masked_target_logits(
                target_logits, self.model._target_mask(obs_t)
            )
            action_t = rollout.action[:, :, t].reshape(N * A, cfg.action_dim)
            logp, ent = self._action_logprob_and_entropy(
                mean, log_std, logits, target_logits, action_t
            )
            logprobs.append(logp.view(N, A))
            entropies.append(ent.view(N, A))
            done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
            h = h.view(N, A, cfg.gru_hidden)
            h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)
        new_logprob = torch.stack(logprobs, dim=2)
        entropy = torch.stack(entropies, dim=2)
        valid_agent = rollout.agent_loss_mask.expand(N, A, L)
        if cfg.value_per_agent:
            value = self.model.value(
                rollout.critic_obs.permute(0, 2, 1, 3).reshape(
                    N * L * A, cfg.critic_obs_dim
                )
            ).view(N, L, A).permute(0, 2, 1)
            advantage = rollout.advantages
        else:
            value = self.model.value(
                rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)
            ).view(N, L)
            advantage = rollout.advantages[:, None, :].expand(N, A, L)
        adv_mean = _masked_mean(advantage, valid_agent)
        adv_var = _masked_mean((advantage - adv_mean) ** 2, valid_agent)
        norm_adv = (advantage - adv_mean) / adv_var.clamp(min=1e-8).sqrt()

        ratio = (new_logprob - rollout.logprob).exp()
        pg1 = ratio * norm_adv
        pg2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * norm_adv
        policy_loss = _masked_mean(-torch.min(pg1, pg2), valid_agent)

        if cfg.value_per_agent:
            value_n = (value - return_mean) / return_std
            old_value_n = (rollout.value - return_mean) / return_std
            return_n = (rollout.returns - return_mean) / return_std
            value_mask = valid_agent
        else:
            value_n = (value - return_mean) / return_std
            old_value_n = (rollout.value - return_mean) / return_std
            return_n = (rollout.returns - return_mean) / return_std
            value_mask = (valid_agent.sum(dim=1) > 0.0).to(valid_agent.dtype)
        value_clipped_n = old_value_n + torch.clamp(
            value_n - old_value_n, -cfg.value_clip_ratio, cfg.value_clip_ratio
        )
        vl_unclipped = (value_n - return_n) ** 2
        vl_clipped = (value_clipped_n - return_n) ** 2
        value_loss = _masked_mean(
            0.5 * torch.max(vl_unclipped, vl_clipped), value_mask
        )
        entropy_mean = _masked_mean(entropy, valid_agent)
        total_loss = (
            policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        actor_grad_norm = self._group_grad_norm(self._actor_params)
        critic_grad_norm = self._group_grad_norm(self._critic_params)
        trunk_grad_norm = self._group_grad_norm(self._trunk_params)
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = _masked_mean(rollout.logprob - new_logprob, valid_agent)
            clip_fraction = _masked_mean(
                ((ratio - 1.0).abs() > cfg.clip_ratio).float(), valid_agent
            )
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "total_loss": float(total_loss.item()),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "trunk_grad_norm": trunk_grad_norm,
            "lr": self.current_learning_rate,
        }


def make_mappo_config(config: dict) -> MappoConfig:
    phase, phase_spec = resolve_phase(config)
    if phase not in (4, 5, 6, 7, 8, 9, 10, 11):
        raise ValueError(
            f"MAPPO trainer only supports phases 4-11, got phase={phase!r}"
        )
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    obs_encoder = str(model_cfg.get("obs_encoder", "flat"))
    n_agents = int(phase_spec["n_agents"])
    raw_agent_loss_mask = ppo_cfg.get("agent_loss_mask", [1.0] * n_agents)
    agent_loss_mask = tuple(float(v) for v in raw_agent_loss_mask)
    if len(agent_loss_mask) != n_agents:
        raise ValueError(
            f"ppo.agent_loss_mask length must be {n_agents}, "
            f"got {len(agent_loss_mask)}"
        )
    if any(v < 0.0 for v in agent_loss_mask):
        raise ValueError("ppo.agent_loss_mask values must be non-negative")
    if not any(v > 0.0 for v in agent_loss_mask):
        raise ValueError("ppo.agent_loss_mask must leave at least one active agent")
    return MappoConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        n_agents=n_agents,
        agent_loss_mask=agent_loss_mask,
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=int(phase_spec["obs_dim"]),
        critic_obs_dim=int(phase_spec["critic_obs_dim"]),
        action_dim=int(phase_spec["action_dim"]),
        continuous_action_dim=int(phase_spec["continuous_action_dim"]),
        binary_action_dim=int(phase_spec["binary_action_dim"]),
        target_action_dim=int(phase_spec.get("target_action_dim", 0)),
        value_per_agent=bool(ppo_cfg.get("value_per_agent", False)),
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
        action_log_std_init=float(model_cfg["action_log_std_init"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_ratio=float(ppo_cfg["clip_ratio"]),
        value_clip_ratio=float(ppo_cfg["value_clip_ratio"]),
        value_coef=float(ppo_cfg["value_coef"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        learning_rate=float(ppo_cfg["learning_rate"]),
        num_epochs=int(ppo_cfg["num_epochs"]),
        minibatch_size=int(ppo_cfg["minibatch_size"]),
        lr_schedule=str(ppo_cfg.get("lr_schedule", "constant")),
        lr_final_ratio=float(ppo_cfg.get("lr_final_ratio", 1.0)),
        warmup_updates=int(ppo_cfg.get("warmup_updates", 0)),
        value_normalization=bool(ppo_cfg.get("value_normalization", True)),
        torch_num_threads=int(ppo_cfg.get("torch_num_threads", 0)),
        vector_env=str(ppo_cfg.get("vector_env", "sync")),
        obs_encoder=obs_encoder,
        entity_token_count=int(model_cfg.get("entity_token_count", 0)),
        entity_token_dim=int(model_cfg.get("entity_token_dim", 0)),
        entity_num_heads=int(model_cfg.get("entity_num_heads", 1)),
        grid_channels=int(model_cfg.get("grid_channels", 0)),
        grid_size=int(model_cfg.get("grid_size", 0)),
    )


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


def evaluate_mappo(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
) -> MappoEvalStats:
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    final_ticks: list[int] = []
    team_a_scores: list[float] = []
    team_b_scores: list[float] = []
    team_a_kills: list[int] = []
    team_b_kills: list[int] = []
    wins = 0
    losses = 0
    draws = 0
    terminated_count = 0
    truncated_count = 0
    for i in range(int(episodes)):
        env = env_fn()
        try:
            obs, _info = env.reset(seed=int(seed) + i)
            h = model.init_hidden(model.cfg.n_agents)
            done = False
            term = False
            trunc = False
            ep_reward = 0.0
            info = {}
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, h = model.greedy_action(obs_t, h)
                obs, reward, term, trunc, info = env.step(action.cpu().numpy())
                ep_reward += float(np.mean(reward))
                done = bool(term or trunc)
            rewards.append(ep_reward)

            winner = str(info.get("winner", ""))
            learner_team = str(info.get("learner_team", ""))
            if winner in ("A", "B") and learner_team in ("A", "B"):
                if winner == learner_team:
                    wins += 1
                else:
                    losses += 1
            elif winner == "Neutral" or trunc:
                draws += 1

            terminated_count += int(bool(term))
            truncated_count += int(bool(trunc))
            final_ticks.append(int(info.get("tick", 0)))
            team_a_scores.append(float(info.get("team_a_score", 0.0)))
            team_b_scores.append(float(info.get("team_b_score", 0.0)))
            team_a_kills.append(int(info.get("team_a_kills", 0)))
            team_b_kills.append(int(info.get("team_b_kills", 0)))
        finally:
            env.close()
    if was_training:
        model.train()
    return MappoEvalStats(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        episodes=len(rewards),
        wins=wins,
        losses=losses,
        draws=draws,
        terminated=terminated_count,
        truncated=truncated_count,
        mean_final_tick=float(np.mean(final_ticks)) if final_ticks else 0.0,
        mean_team_a_score=float(np.mean(team_a_scores)) if team_a_scores else 0.0,
        mean_team_b_score=float(np.mean(team_b_scores)) if team_b_scores else 0.0,
        mean_team_a_kills=float(np.mean(team_a_kills)) if team_a_kills else 0.0,
        mean_team_b_kills=float(np.mean(team_b_kills)) if team_b_kills else 0.0,
    )


def _eval_stats_dict(stats: MappoEvalStats) -> dict[str, float | int]:
    episodes = max(1, int(stats.episodes))
    return {
        "episodes": int(stats.episodes),
        "wins": int(stats.wins),
        "losses": int(stats.losses),
        "draws": int(stats.draws),
        "win_rate": float(stats.wins) / float(episodes),
        "loss_rate": float(stats.losses) / float(episodes),
        "draw_rate": float(stats.draws) / float(episodes),
        "mean_reward": float(stats.mean_reward),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
        "mean_kills_a": float(stats.mean_team_a_kills),
        "mean_kills_b": float(stats.mean_team_b_kills),
        "mean_final_tick": float(stats.mean_final_tick),
        "terminated": int(stats.terminated),
        "truncated": int(stats.truncated),
    }


def _run_eval_gate(
    *,
    phase_label: str,
    stats: MappoEvalStats,
    gate_cfg: dict,
    output_dir: Path,
) -> dict:
    gate = check_eval_gate(_eval_stats_dict(stats), gate_cfg)
    gate_path = output_dir / str(gate_cfg.get("output", "eval_gate.json"))
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{phase_label}/mappo] eval_gate "
        f"{'pass' if gate['passed'] else 'fail'} wrote {gate_path}",
        flush=True,
    )
    if not gate["passed"]:
        raise RuntimeError("MAPPO eval gate failed: " + "; ".join(gate["failures"]))
    return gate


def _matrix_native_bot_env_fn(phase: int, ckpt_env_cfg: dict, bot: str):
    eval_phase = 8 if int(phase) == 9 else int(phase)
    if eval_phase not in (4, 5, 6, 7, 8, 10):
        raise ValueError(f"matrix bot eval does not support phase {phase}")
    env_cfg = dict(ckpt_env_cfg)
    env_cfg["opponent_bot"] = str(bot)
    env_cfg["learner_team"] = "A"
    _phase, spec = resolve_phase({"phase": eval_phase, "env": env_cfg})
    env_fn, _meta, _seed = spec["env_bundle"]({"phase": eval_phase, "env": env_cfg})
    return env_fn


def _matrix_snapshot_env_fn(ckpt_env_cfg: dict, snapshot_path: str):
    env_cfg = dict(ckpt_env_cfg)
    env_cfg["opponent_bot"] = "snapshot"
    env_cfg["learner_team"] = "A"
    env_cfg["snapshot_paths"] = [snapshot_path]
    env_cfg["snapshot_league"] = {
        "latest": [snapshot_path],
        "weights": {"latest": 1.0},
    }
    _phase, spec = resolve_phase({"phase": 9, "env": env_cfg})
    env_fn, _meta, _seed = spec["env_bundle"]({"phase": 9, "env": env_cfg})
    return env_fn


def _mappo_matrix_row(
    *,
    learner: str,
    opponent: str,
    opponent_type: str,
    stats: MappoEvalStats,
) -> dict:
    episodes = max(1, int(stats.episodes))
    return {
        "learner": learner,
        "opponent": opponent,
        "opponent_type": opponent_type,
        "episodes": int(stats.episodes),
        "win_rate": float(stats.wins) / float(episodes),
        "loss_rate": float(stats.losses) / float(episodes),
        "draw_rate": float(stats.draws) / float(episodes),
        "mean_reward": float(stats.mean_reward),
        "mean_score_a": float(stats.mean_team_a_score),
        "mean_score_b": float(stats.mean_team_b_score),
        "mean_kills_a": float(stats.mean_team_a_kills),
        "mean_kills_b": float(stats.mean_team_b_kills),
        "mean_final_tick": float(stats.mean_final_tick),
    }


def _matrix_retention_summary(
    rows: list[dict],
    gate: dict | None = None,
) -> dict[str, float | int | bool | None]:
    if not rows:
        return {
            "matrix_score": 0.0,
            "matrix_rows": 0,
            "matrix_gate_passed": False if gate is not None else None,
        }
    scores = [
        float(row.get("win_rate", 0.0)) - float(row.get("loss_rate", 0.0))
        for row in rows
    ]
    return {
        "matrix_score": float(np.mean(scores)),
        "matrix_rows": len(rows),
        "matrix_gate_passed": bool(gate.get("passed", False))
        if gate is not None
        else None,
    }


def _matrix_gate_label(value: bool | None) -> str:
    if value is None:
        return "ungated"
    return "pass" if bool(value) else "fail"


def _run_mappo_matrix_eval(
    *,
    model: MappoActorCritic,
    phase: int,
    ckpt_env_cfg: dict,
    matrix_cfg: dict,
    output_dir: Path,
    seed: int,
) -> list[dict]:
    episodes = int(matrix_cfg.get("episodes", 1))
    anchor_bots = [str(bot) for bot in matrix_cfg.get("anchor_bots", ())]
    opponent_checkpoints = [
        str(path) for path in matrix_cfg.get("opponent_checkpoints", ())
    ]
    rows: list[dict] = []
    if model.cfg.n_agents == 6 and int(phase) == 11:
        if bool(matrix_cfg.get("current_selfplay", True)):
            env_cfg = dict(ckpt_env_cfg)
            env_cfg["self_play_schedule"] = {
                "weights": {"current": 1.0, "snapshot": 0.0, "anchor": 0.0}
            }
            _phase, spec = resolve_phase({"phase": 11, "env": env_cfg})
            env_fn, _meta, _seed = spec["env_bundle"](
                {"phase": 11, "env": env_cfg}
            )
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=episodes,
                seed=int(seed) + 720_000,
            )
            rows.append(
                _mappo_matrix_row(
                    learner="ckpt_final.pt",
                    opponent="current",
                    opponent_type="selfplay",
                    stats=stats,
                )
            )
        for bot_idx, bot in enumerate(anchor_bots):
            env_cfg = dict(ckpt_env_cfg)
            env_cfg["self_play_schedule"] = {
                "weights": {"current": 0.0, "snapshot": 0.0, "anchor": 1.0},
                "anchor_bot": bot,
            }
            _phase, spec = resolve_phase({"phase": 11, "env": env_cfg})
            env_fn, _meta, _seed = spec["env_bundle"](
                {"phase": 11, "env": env_cfg}
            )
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=episodes,
                seed=int(seed) + 730_000 + 100 * bot_idx,
            )
            rows.append(
                _mappo_matrix_row(
                    learner="ckpt_final.pt",
                    opponent=bot,
                    opponent_type="bot",
                    stats=stats,
                )
            )
        for opp_idx, opponent in enumerate(opponent_checkpoints):
            env_cfg = dict(ckpt_env_cfg)
            env_cfg["self_play_schedule"] = {
                "weights": {"current": 0.0, "snapshot": 1.0, "anchor": 0.0}
            }
            env_cfg["snapshot_league"] = {
                "latest": [opponent],
                "weights": {"latest": 1.0},
            }
            _phase, spec = resolve_phase({"phase": 11, "env": env_cfg})
            env_fn, _meta, _seed = spec["env_bundle"](
                {"phase": 11, "env": env_cfg}
            )
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=episodes,
                seed=int(seed) + 740_000 + 100 * opp_idx,
            )
            rows.append(
                _mappo_matrix_row(
                    learner="ckpt_final.pt",
                    opponent=Path(opponent).name,
                    opponent_type="snapshot",
                    stats=stats,
                )
            )
    elif model.cfg.n_agents != 3:
        raise ValueError(
            "run.matrix_eval currently supports 3-agent MAPPO checkpoints; "
            f"got n_agents={model.cfg.n_agents}"
        )
    else:
        for bot_idx, bot in enumerate(anchor_bots):
            env_fn = _matrix_native_bot_env_fn(phase, ckpt_env_cfg, bot)
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=episodes,
                seed=int(seed) + 700_000 + 100 * bot_idx,
            )
            rows.append(
                _mappo_matrix_row(
                    learner="ckpt_final.pt",
                    opponent=bot,
                    opponent_type="bot",
                    stats=stats,
                )
            )
        for opp_idx, opponent in enumerate(opponent_checkpoints):
            env_fn = _matrix_snapshot_env_fn(ckpt_env_cfg, opponent)
            stats = evaluate_mappo(
                model,
                env_fn,
                episodes=episodes,
                seed=int(seed) + 710_000 + 100 * opp_idx,
            )
            rows.append(
                _mappo_matrix_row(
                    learner="ckpt_final.pt",
                    opponent=Path(opponent).name,
                    opponent_type="snapshot",
                    stats=stats,
                )
            )
    if not rows:
        return rows
    output_name = str(matrix_cfg.get("output", "matrix_eval.json"))
    output_path = output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    gate: dict | None = None
    if matrix_cfg.get("gate"):
        gate = check_matrix_gate(rows, dict(matrix_cfg.get("gate", {})))
        gate_path = output_dir / str(
            matrix_cfg.get("gate_output", "matrix_gate.json")
        )
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(
            f"[phase{phase}/mappo] matrix_gate "
            f"{'pass' if gate['passed'] else 'fail'} wrote {gate_path}",
            flush=True,
        )
        if not gate["passed"]:
            raise RuntimeError(
                "MAPPO matrix gate failed: " + "; ".join(gate["failures"])
            )
    for row in rows:
        print(
            f"[phase{phase}/mappo] matrix "
            f"opponent={row['opponent_type']}:{row['opponent']} "
            f"win={row['win_rate']:.3f} draw={row['draw_rate']:.3f} "
            f"reward={row['mean_reward']:+.3f} "
            f"score={row['mean_score_a']:.2f}/{row['mean_score_b']:.2f}",
            flush=True,
        )
    print(f"[phase{phase}/mappo] matrix wrote {output_path}", flush=True)
    return rows


def train_phase4_from_config(config: dict) -> dict[str, float]:
    phase, phase_spec = resolve_phase(config)
    phase_label = str(phase_spec["label"])
    env_fn, ckpt_env_cfg, seed_base = phase_spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    run_cfg = config.get("run", {})
    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 10))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    output_dir = Path(str(run_cfg.get("output_dir", "runs/phase4_mappo"))) / "mappo"
    output_dir.mkdir(parents=True, exist_ok=True)
    retention: SnapshotRetention | None = None
    if run_cfg.get("snapshot_retention"):
        retention_cfg = dict(run_cfg.get("snapshot_retention", {}))
        env_cfg = config.get("env", {})
        retention = SnapshotRetention(
            output_dir / str(retention_cfg.get("manifest", "snapshot_league.json")),
            max_latest=int(retention_cfg.get("max_latest", 20)),
            preserve_best=int(retention_cfg.get("preserve_best", 3)),
            anchor_paths=tuple(
                retention_cfg.get("anchor_paths", env_cfg.get("snapshot_paths", ()))
            )
            if bool(retention_cfg.get("include_config_anchors", True))
            else (),
            weights=dict(
                retention_cfg.get(
                    "weights",
                    env_cfg.get("snapshot_league", {}).get(
                        "weights",
                        {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
                    ),
                )
            ),
        )

    trainer = MappoTrainer(env_fn, cfg, seed=seed_base)
    best_eval = float("-inf")
    best_state: dict | None = None
    last_eval = float("nan")
    try:
        bc_steps = int(run_cfg.get("bc_pretrain_steps", 0))
        if bc_steps > 0:
            bc_pretrain_walk_to_objective(
                trainer.model,
                env_fn,
                cfg,
                steps=bc_steps,
                batch_size=int(run_cfg.get("bc_batch_size", 1024)),
                learning_rate=float(run_cfg.get("bc_learning_rate", 1.0e-3)),
                seed=seed_base + 50_000,
                log_label=phase_label,
            )
            eval_stats = evaluate_mappo(
                trainer.model,
                env_fn,
                episodes=eval_episodes,
                seed=seed_base + 90_000,
            )
            last_eval = eval_stats.mean_reward
            if last_eval > best_eval:
                best_eval = last_eval
                best_state = copy.deepcopy(trainer.model.state_dict())
            print(
                f"[{phase_label}/mappo] bc_eval "
                f"mean_reward={eval_stats.mean_reward:+.3f} "
                f"wins={eval_stats.wins}/{eval_stats.episodes} "
                f"draws={eval_stats.draws}/{eval_stats.episodes} "
                f"score={eval_stats.mean_team_a_score:.2f}/"
                f"{eval_stats.mean_team_b_score:.2f}",
                flush=True,
            )
        for update_idx in range(1, total_updates + 1):
            lr = lr_for_update(
                update_idx,
                total_updates,
                base_lr=cfg.learning_rate,
                schedule=cfg.lr_schedule,
                lr_final_ratio=cfg.lr_final_ratio,
                warmup_updates=cfg.warmup_updates,
            )
            trainer.set_learning_rate(lr)
            metrics = trainer.update(trainer.collect_rollout())
            if update_idx % int(run_cfg.get("log_every", 1)) == 0:
                print(
                    f"[{phase_label}/mappo] update={update_idx}/{total_updates} "
                    f"policy_loss={metrics['policy_loss']:.3f} "
                    f"value_loss={metrics['value_loss']:.3f} "
                    f"entropy={metrics['entropy']:.3f} "
                    f"rew={metrics['rollout_reward_mean']:+.3f}"
                    f"/{metrics['rollout_reward_std']:.3f} "
                    f"adv={metrics['advantage_mean']:+.3f}"
                    f"/{metrics['advantage_std']:.3f} "
                    f"move={metrics['action_move_mag_mean']:.3f} "
                    f"bin={metrics['action_binary_mean']:.3f} "
                    f"dist={metrics['mean_distance_to_objective']:.3f} "
                    f"onpt={metrics['self_on_point_fraction']:.3f} "
                    f"gn={metrics['actor_grad_norm']:.2e}/"
                    f"{metrics['critic_grad_norm']:.2e}/"
                    f"{metrics['trunk_grad_norm']:.2e} "
                    f"lr={lr:.2e}",
                    flush=True,
                )
            if update_idx % eval_every == 0 or update_idx == total_updates:
                eval_stats = evaluate_mappo(
                    trainer.model,
                    env_fn,
                    episodes=eval_episodes,
                    seed=seed_base + 100_000 + update_idx,
                )
                last_eval = eval_stats.mean_reward
                print(
                    f"[{phase_label}/mappo] eval update={update_idx}/{total_updates} "
                    f"mean_reward={eval_stats.mean_reward:+.3f} "
                    f"wins={eval_stats.wins}/{eval_stats.episodes} "
                    f"losses={eval_stats.losses}/{eval_stats.episodes} "
                    f"draws={eval_stats.draws}/{eval_stats.episodes} "
                    f"term={eval_stats.terminated} trunc={eval_stats.truncated} "
                    f"tick={eval_stats.mean_final_tick:.1f} "
                    f"score={eval_stats.mean_team_a_score:.2f}/"
                    f"{eval_stats.mean_team_b_score:.2f} "
                    f"kills={eval_stats.mean_team_a_kills:.1f}/"
                    f"{eval_stats.mean_team_b_kills:.1f}",
                    flush=True,
                )
                if last_eval > best_eval:
                    best_eval = last_eval
                    best_state = copy.deepcopy(trainer.model.state_dict())
                if run_cfg.get("eval_gate"):
                    _run_eval_gate(
                        phase_label=phase_label,
                        stats=eval_stats,
                        gate_cfg=dict(run_cfg.get("eval_gate", {})),
                        output_dir=output_dir,
                    )
            if update_idx % checkpoint_every == 0 or update_idx == total_updates:
                checkpoint_path = output_dir / f"ckpt_{update_idx:04d}.pt"
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict(),
                        "config": {
                            "phase": phase,
                            "env": ckpt_env_cfg,
                            "mappo": cfg.__dict__,
                        },
                    },
                    checkpoint_path,
                )
                if retention is not None:
                    manifest = retention.record_checkpoint(
                        checkpoint_path,
                        update=update_idx,
                        score=last_eval,
                    )
                    print(
                        f"[{phase_label}/mappo] snapshot_pool "
                        f"latest={len(manifest['latest'])} "
                        f"historical={len(manifest['historical'])} "
                        f"anchor={len(manifest['anchor'])}",
                        flush=True,
                    )
    finally:
        trainer.close()
    final_state = best_state if best_state is not None else trainer.model.state_dict()
    torch.save(
        {
            "model_state_dict": final_state,
            "config": {"phase": phase, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        output_dir / "ckpt_final.pt",
    )
    if run_cfg.get("matrix_eval"):
        matrix_model = MappoActorCritic(cfg)
        matrix_model.load_state_dict(final_state)
        matrix_model.eval()
        rows = _run_mappo_matrix_eval(
            model=matrix_model,
            phase=phase,
            ckpt_env_cfg=ckpt_env_cfg,
            matrix_cfg=dict(run_cfg.get("matrix_eval", {})),
            output_dir=output_dir,
            seed=seed_base,
        )
        if retention is not None:
            gate: dict | None = None
            matrix_cfg = dict(run_cfg.get("matrix_eval", {}))
            if matrix_cfg.get("gate"):
                gate_path = output_dir / str(
                    matrix_cfg.get("gate_output", "matrix_gate.json")
                )
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
            summary = _matrix_retention_summary(rows, gate)
            manifest = retention.record_checkpoint(
                output_dir / "ckpt_final.pt",
                update=total_updates,
                score=best_eval if best_eval > float("-inf") else last_eval,
                matrix_score=float(summary["matrix_score"]),
                matrix_gate_passed=(
                    bool(summary["matrix_gate_passed"])
                    if summary["matrix_gate_passed"] is not None
                    else None
                ),
                matrix_rows=int(summary["matrix_rows"]),
            )
            print(
                f"[{phase_label}/mappo] snapshot_pool_matrix "
                f"score={float(summary['matrix_score']):+.3f} "
                f"gate={_matrix_gate_label(summary['matrix_gate_passed'])} "
                f"latest={len(manifest['latest'])} "
                f"historical={len(manifest['historical'])} "
                f"anchor={len(manifest['anchor'])}",
                flush=True,
            )
    return {"mappo": best_eval if best_eval > float("-inf") else last_eval}
