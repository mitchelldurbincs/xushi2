"""Phase-4 recurrent MAPPO trainer.

This is the first CTDE training path: a shared recurrent actor consumes
per-agent actor observations while a centralized critic consumes the
team-level critic observation supplied by ``Phase4MappoEnv``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from xushi2.entity_obs import (
    ENTITY_OBS_DIM,
    ENTITY_TOKEN_COUNT,
    ENTITY_TOKEN_DIM,
)
from xushi2.grid_obs import GRID_CHANNELS, GRID_SIZE
from xushi2.obs_manifest import actor_field_slice

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
    # OAI Five-style team_spirit ramp on per-agent shaped rewards. The ramp
    # is linear from `team_spirit_initial` at update 0 to `team_spirit_final`
    # at `team_spirit_ramp_fraction * total_updates`, then held at final.
    # All-zero defaults keep team_spirit OFF for back-compat.
    team_spirit_initial: float = 0.0
    team_spirit_final: float = 0.0
    team_spirit_ramp_fraction: float = 0.3
    # Escape Protocol 5.1: optional auxiliary supervised loss that predicts
    # the angle to the visible enemy from the actor path.
    aim_aux_coef: float = 0.0


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


def compute_team_spirit(
    *,
    update: int,
    total: int,
    initial: float,
    final: float,
    ramp_fraction: float,
) -> float:
    """Linear ramp from ``initial`` at update 0 (1-indexed: update=1 is the
    first update) to ``final`` at ``ramp_fraction * total``, then held at
    ``final``. ``ramp_fraction <= 0`` jumps to ``final`` immediately."""
    if ramp_fraction <= 0.0:
        return final
    ramp_end_update = max(1, int(ramp_fraction * total))
    if update >= ramp_end_update:
        return final
    progress = update / ramp_end_update
    return initial + progress * (final - initial)


def _eval_outcome_counts(
    *,
    winner: str,
    learner_team: str,
    truncated: bool,
) -> tuple[int, int, int]:
    if learner_team == "both":
        return (0, 0, 1)
    if winner in ("A", "B") and learner_team in ("A", "B"):
        return (1, 0, 0) if winner == learner_team else (0, 1, 0)
    if winner == "Neutral" or truncated:
        return (0, 0, 1)
    return (0, 0, 0)


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
                    f"entity_attention obs_dim must be {ENTITY_OBS_DIM}, got {cfg.obs_dim}"
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
                    f"entity_attention_grid obs_dim must be {expected_obs_dim}, got {cfg.obs_dim}"
                )
            if cfg.entity_token_count <= 0 or cfg.entity_token_dim != ENTITY_TOKEN_DIM:
                raise ValueError(
                    f"entity_attention_grid token shape must match (positive, {ENTITY_TOKEN_DIM})"
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
        if cfg.aim_aux_coef > 0.0 and cfg.obs_encoder != "flat":
            raise ValueError("aim_aux_coef currently supports only flat Phase-4 observations")
        self.actor_gru = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.actor_mean_head = nn.Linear(cfg.head_hidden, cfg.continuous_action_dim)
        self.actor_binary_head = nn.Linear(cfg.head_hidden, cfg.binary_action_dim)
        self.actor_target_head = (
            nn.Linear(cfg.head_hidden, cfg.target_action_dim) if cfg.target_action_dim > 0 else None
        )
        self.actor_aim_aux_head = (
            nn.Linear(cfg.head_hidden, 1) if cfg.aim_aux_coef > 0.0 else None
        )
        self.log_std = nn.Parameter(torch.ones(cfg.continuous_action_dim) * cfg.action_log_std_init)
        self.critic = nn.Sequential(
            nn.Linear(cfg.critic_obs_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(),
            nn.Linear(cfg.head_hidden, 1),
        )

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(batch_size, self.cfg.gru_hidden, device=p.device, dtype=p.dtype)

    def policy_outputs(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        features, h_next = self.actor_head_features(obs, h)
        mean = self.actor_mean_head(features)
        logits = self.actor_binary_head(features)
        target_logits = (
            self.actor_target_head(features) if self.actor_target_head is not None else None
        )
        return mean, self.log_std, logits, target_logits, h_next

    def actor_head_features(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._actor_features(obs)
        h_next = self.actor_gru(emb, h)
        return self.actor_body(h_next), h_next

    def aim_aux_prediction_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.actor_aim_aux_head is None:
            return None
        return self.actor_aim_aux_head(features).squeeze(-1)

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


def aim_aux_targets(
    obs: torch.Tensor, cfg: MappoConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target enemy angle in radians and a visibility mask.

    Phase-4 flat actor observations expose the nearest enemy alive flag at
    index 10 and relative position at indices 12:14. The target is the angle
    from the agent to that visible enemy in the actor observation frame.
    """
    if cfg.obs_encoder != "flat":
        target = torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
        mask = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        return target, mask
    enemy_alive = obs[:, 10] > 0.5
    enemy_rel_pos = obs[:, 12:14]
    rel_norm = torch.linalg.vector_norm(enemy_rel_pos, dim=-1)
    target = torch.atan2(enemy_rel_pos[:, 1], enemy_rel_pos[:, 0])
    return target, enemy_alive & (rel_norm > 1.0e-6)


def wrapped_angle_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smallest signed angle difference in radians."""
    return torch.atan2(torch.sin(pred - target), torch.cos(pred - target))


def aim_aux_loss_and_rmse(
    pred: torch.Tensor | None,
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred is None:
        zero = obs.new_tensor(0.0)
        return zero, zero, zero
    target, visible = aim_aux_targets(obs, cfg)
    valid = visible
    if mask is not None:
        valid = valid & (mask.reshape(-1) > 0.0)
    count = valid.to(obs.dtype).sum()
    if float(count.item()) <= 0.0:
        zero = obs.new_tensor(0.0)
        return zero, zero, count
    err = wrapped_angle_error(pred, target)
    mse = (err[valid] ** 2).mean()
    return mse, mse.sqrt().clamp(max=math.pi), count
