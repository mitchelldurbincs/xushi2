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

from xushi2.multi_enemy_obs import ENTITY_TOKEN_DIM, GRID_CHANNELS, GRID_SIZE
from xushi2.obs_manifest import actor_field_slice

_LOG2 = 0.6931471805599453
_OWN_POSITION_SLICE = actor_field_slice("own_position")
_ENEMY_ALIVE_SLICE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS_SLICE = actor_field_slice("enemy_relative_position")
_ENEMY_HP_SLICE = actor_field_slice("enemy_hp")


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
    # Escape Protocol 5.2: optional per-action entropy coefficients. When
    # omitted, each component defaults to `entropy_coef` for back-compat.
    entropy_coef_move: float | None = None
    entropy_coef_aim: float | None = None
    entropy_coef_binary: float | None = None
    # Entropy-bonus anneal (2026-07-29): multiplicative scale on the whole
    # entropy bonus, 1.0 -> entropy_final_scale over entropy_anneal_updates.
    # 0 anneal updates = off (scale pinned at 1.0). Motivated by the v3
    # temperature ablation: the converting mean is unreachable under the
    # policy's own sampling noise unless the bonus propping it up decays.
    # NOTE (v4 post-mortem, 2026-07-30): this alone does NOT shrink std —
    # PPO gives the global log_std parameter almost no gradient. Use the
    # log_std anneal below for actual sharpening.
    entropy_anneal_updates: int = 0
    entropy_final_scale: float = 0.0
    # Direct log_std anneal (2026-07-30): the trainer overwrites the global
    # log_std parameter each update with warm-start base + a linear offset
    # 0 -> log_std_final_offset over log_std_anneal_updates, then holds.
    # Sampling, logprobs, and entropy all shift coherently, and checkpoints
    # carry the annealed value. 0 anneal updates = off. ln(0.25) = -1.386
    # quarters the continuous-action std by the end of the anneal.
    log_std_anneal_updates: int = 0
    log_std_final_offset: float = 0.0
    target_action_dim: int = 0
    value_per_agent: bool = False
    mask_fire_when_no_visible_enemy: bool = False
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
    # Phase 4 structural probe: internal three-way enemy target selection
    # head. This does not change the simulator action space.
    target_selection_dim: int = 0
    target_conditioned_combat: bool = False
    target_selection_aux_coef: float = 0.0
    target_selection_aux_mode: str = "nearest_visible"
    target_selection_objective_proximity_coef: float = 0.1
    mode_gated_combat: bool = False
    mode_aux_coef: float = 0.3
    # Warm-start stabilizers (2026-07-09, see
    # docs/reports/2026-07-09-phase4-3v3-review-recommendations.md).
    # ``critic_warmup_updates``: for the first N updates only the value loss
    # is optimized (the actor receives no gradient), so a critic that is
    # random or fit to a different reward scheme cannot wreck a warm-started
    # policy through garbage advantages.
    critic_warmup_updates: int = 0
    # ``anchor_kl_coef``: weight of an analytic KL(pi_current || pi_anchor)
    # penalty toward a frozen copy of the policy taken at PPO start (after
    # warm start + any BC/bridge pretrain). Annealed linearly to zero over
    # ``anchor_kl_anneal_updates`` (<=0 holds it constant). Zero disables.
    anchor_kl_coef: float = 0.0
    anchor_kl_anneal_updates: int = 0
    # ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"auto"`` (CUDA if available
    # else CPU). Resolved to ``torch.device`` once in the trainer.
    device: str = "cpu"


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
    team_a_hit_fire: float = 0.0
    team_b_hit_fire: float = 0.0
    team_a_visible_fire_rate: float = 0.0
    team_b_visible_fire_rate: float = 0.0
    team_a_aim_error_rad: float = 0.0
    team_b_aim_error_rad: float = 0.0
    team_a_target_entropy: float = 0.0
    team_b_target_entropy: float = 0.0
    team_a_same_target_fraction: float = 0.0
    team_b_same_target_fraction: float = 0.0
    team_a_target_selection_entropy: float = 0.0
    team_b_target_selection_entropy: float = 0.0
    team_a_damage_per_fire: float = 0.0
    team_b_damage_per_fire: float = 0.0
    mean_p_combat: float = 0.0
    mode_accuracy: float = 0.0
    intentional_fire_fraction: float = 0.0
    objective_focus_fraction: float = 0.0
    mean_uncontested_on_point_seconds_a: float = 0.0
    mean_uncontested_on_point_seconds_b: float = 0.0
    mean_majority_on_point_seconds_a: float = 0.0
    mean_majority_on_point_seconds_b: float = 0.0
    mean_alive_edge_no_score_seconds_a: float = 0.0
    mean_alive_edge_no_score_seconds_b: float = 0.0
    mean_cap_progress_gain_ticks: float = 0.0
    mean_cap_progress_loss_ticks: float = 0.0
    mean_first_team_a_alive_edge_to_score_seconds: float = -1.0
    majority_to_uncontested_within_n_fraction_a: float = 0.0
    majority_to_uncontested_within_n_fraction_b: float = 0.0
    contested_majority_windows_per_episode_a: float = 0.0
    contested_majority_windows_per_episode_b: float = 0.0
    contested_majority_window_mean_seconds_a: float = 0.0
    contested_majority_window_mean_seconds_b: float = 0.0
    contested_majority_window_p50_seconds_a: float = 0.0
    contested_majority_window_p50_seconds_b: float = 0.0
    contested_majority_window_p90_seconds_a: float = 0.0
    contested_majority_window_p90_seconds_b: float = 0.0
    on_point_nearest_enemy_distance_mean_a: float = 0.0
    on_point_nearest_enemy_distance_mean_b: float = 0.0
    on_point_enemy_los_fraction_a: float = 0.0
    on_point_enemy_los_fraction_b: float = 0.0
    contested_majority_hit_fire_a: float = 0.0
    contested_majority_hit_fire_b: float = 0.0
    contested_majority_damage_per_fire_a: float = 0.0
    contested_majority_damage_per_fire_b: float = 0.0
    objective_unlock_seconds: float = 0.0
    objective_capture_seconds: float = 0.0
    respawn_ticks: int = 0
    std_team_a_score: float = 0.0
    std_team_b_score: float = 0.0


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


def compute_entropy_scale(
    *,
    update: int,
    anneal_updates: int,
    final_scale: float,
) -> float:
    """Linear anneal of the entropy-bonus scale from 1.0 to ``final_scale``
    over ``anneal_updates``, then held. ``anneal_updates <= 0`` disables the
    anneal (scale stays 1.0)."""
    if anneal_updates <= 0:
        return 1.0
    if update >= anneal_updates:
        return float(final_scale)
    progress = max(0.0, float(update) / float(anneal_updates))
    return 1.0 + progress * (float(final_scale) - 1.0)


def compute_log_std_offset(
    *,
    update: int,
    anneal_updates: int,
    final_offset: float,
) -> float:
    """Linear anneal of the log_std offset from 0.0 to ``final_offset`` over
    ``anneal_updates``, then held. ``anneal_updates <= 0`` disables (0.0)."""
    if anneal_updates <= 0:
        return 0.0
    if update >= anneal_updates:
        return float(final_offset)
    progress = max(0.0, float(update) / float(anneal_updates))
    return progress * float(final_offset)


def compute_opponent_handicap(
    *,
    update: int,
    initial_aim_noise: float,
    final_aim_noise: float,
    initial_fire_cadence: int,
    final_fire_cadence: int,
    anneal_updates: int,
) -> tuple[float, int]:
    """Linear anneal of the opponent handicap (aim noise radians, fire
    cadence ticks) from initial to final over ``anneal_updates``, then held.
    ``anneal_updates <= 0`` holds the initial values."""
    if anneal_updates <= 0:
        return float(initial_aim_noise), int(initial_fire_cadence)
    progress = min(1.0, max(0.0, float(update) / float(anneal_updates)))
    noise = float(initial_aim_noise) + progress * (
        float(final_aim_noise) - float(initial_aim_noise)
    )
    cadence = int(round(
        float(initial_fire_cadence)
        + progress * (float(final_fire_cadence) - float(initial_fire_cadence))
    ))
    return noise, max(1, cadence)


def compute_majority_on_point_alpha(
    *,
    update: int,
    initial: float,
    anneal_updates: int,
) -> float:
    """Linear anneal from ``initial`` to zero.

    ``anneal_updates <= 0`` intentionally holds the coefficient constant for
    diagnostic runs.
    """
    if initial <= 0.0:
        return 0.0
    if anneal_updates <= 0:
        return float(initial)
    if update >= anneal_updates:
        return 0.0
    progress = max(0.0, float(update) / float(anneal_updates))
    return float(initial) * (1.0 - progress)


def compute_respawn_ticks(
    *,
    update: int,
    initial_ticks: int,
    final_ticks: int,
    anneal_updates: int,
) -> int:
    """Linear respawn-time curriculum in ticks.

    Mirrors ``compute_objective_timing_seconds``: ``anneal_updates <= 0``
    holds the initial value for fixed-easy diagnostic runs. Values are
    rounded to whole ticks.
    """
    initial = int(initial_ticks)
    final = int(final_ticks)
    if min(initial, final) <= 0:
        raise ValueError("respawn ticks must be > 0")
    if anneal_updates <= 0:
        return initial
    if update >= anneal_updates:
        return final
    progress = max(0.0, float(update) / float(anneal_updates))
    return round(initial + progress * (final - initial))


def compute_anchor_kl_coef(
    *,
    update: int,
    initial: float,
    anneal_updates: int,
) -> float:
    """Linear anneal of the anchor-KL coefficient from ``initial`` to zero.

    ``anneal_updates <= 0`` holds the coefficient constant. Same semantics as
    ``compute_majority_on_point_alpha``; kept separate for a legible name at
    call sites.
    """
    return compute_majority_on_point_alpha(
        update=update, initial=initial, anneal_updates=anneal_updates
    )


def compute_objective_timing_seconds(
    *,
    update: int,
    initial_unlock_seconds: float,
    final_unlock_seconds: float,
    initial_capture_seconds: float,
    final_capture_seconds: float,
    anneal_updates: int,
) -> tuple[float, float]:
    """Linear objective timing curriculum.

    ``anneal_updates <= 0`` holds the initial timing for fixed-easy
    diagnostic runs.
    """
    initial_unlock = float(initial_unlock_seconds)
    final_unlock = float(final_unlock_seconds)
    initial_capture = float(initial_capture_seconds)
    final_capture = float(final_capture_seconds)
    if min(initial_unlock, final_unlock, initial_capture, final_capture) <= 0.0:
        raise ValueError("objective timing seconds must be > 0")
    if anneal_updates <= 0:
        return initial_unlock, initial_capture
    if update >= anneal_updates:
        return final_unlock, final_capture
    progress = max(0.0, float(update) / float(anneal_updates))
    unlock = initial_unlock + progress * (final_unlock - initial_unlock)
    capture = initial_capture + progress * (final_capture - initial_capture)
    return unlock, capture


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
        if cfg.mask_fire_when_no_visible_enemy and cfg.obs_encoder != "flat":
            raise ValueError(
                "mask_fire_when_no_visible_enemy currently supports only flat observations"
            )
        if (
            cfg.target_conditioned_combat
            or cfg.target_selection_dim > 0
            or cfg.target_selection_aux_coef > 0.0
        ):
            if cfg.obs_encoder != "flat":
                raise ValueError("target-conditioned combat currently supports only flat obs")
            expected_target_dim = 4 if cfg.target_selection_aux_mode == "team_focus_low_hp" else 3
            if cfg.target_selection_dim != expected_target_dim:
                raise ValueError(
                    "target-conditioned combat requires target_selection_dim="
                    f"{expected_target_dim} for mode {cfg.target_selection_aux_mode!r}"
                )
            if cfg.n_agents != 3:
                raise ValueError("target-conditioned combat requires three Phase-4 agents")
            if cfg.target_selection_aux_mode not in (
                "nearest_visible",
                "lowest_hp",
                "team_focus_low_hp",
            ):
                raise ValueError(
                    "target_selection_aux_mode must be 'nearest_visible', "
                    "'lowest_hp', or 'team_focus_low_hp'"
                )
        self.actor_gru = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.actor_target_selection_head = (
            nn.Linear(cfg.head_hidden, cfg.target_selection_dim)
            if cfg.target_selection_dim > 0
            else None
        )
        self.actor_target_condition = (
            nn.Sequential(nn.Linear(4, cfg.head_hidden), nn.ReLU())
            if cfg.target_conditioned_combat
            else None
        )
        self.actor_mean_head = nn.Linear(cfg.head_hidden, cfg.continuous_action_dim)
        self.actor_binary_head = nn.Linear(cfg.head_hidden, cfg.binary_action_dim)
        self.actor_mode_head = nn.Linear(cfg.head_hidden, 2) if cfg.mode_gated_combat else None
        self.actor_target_head = (
            nn.Linear(cfg.head_hidden, cfg.target_action_dim) if cfg.target_action_dim > 0 else None
        )
        self.actor_aim_aux_head = nn.Linear(cfg.head_hidden, 1) if cfg.aim_aux_coef > 0.0 else None
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
        mean, logits, _target_selection_logits = self.policy_heads_from_features(obs, features)
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

    def target_selection_logits_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.actor_target_selection_head is None:
            return None
        return self.actor_target_selection_head(features)

    def mode_logits_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if self.actor_mode_head is None:
            return None
        return self.actor_mode_head(features)

    @staticmethod
    def combat_probability(mode_logits: torch.Tensor | None) -> torch.Tensor | None:
        if mode_logits is None:
            return None
        return torch.softmax(mode_logits, dim=-1)[:, 1]

    def gated_binary_logits(
        self,
        logits: torch.Tensor,
        mode_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.cfg.mode_gated_combat or mode_logits is None or logits.shape[-1] == 0:
            return logits
        p_fire_raw = torch.sigmoid(logits[:, 0])
        p_combat = self.combat_probability(mode_logits)
        assert p_combat is not None
        p_fire_actual = (p_fire_raw * p_combat).clamp(1.0e-6, 1.0 - 1.0e-6)
        out = logits.clone()
        out[:, 0] = torch.logit(p_fire_actual)
        return out

    def policy_heads_from_features(
        self, obs: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_selection_logits = self.target_selection_logits_from_features(features)
        head_features = features
        target_context: _TargetContext | None = None
        if self.actor_target_condition is not None:
            target_context = target_selection_context(obs, self.cfg, target_selection_logits)
            head_features = features + self.actor_target_condition(target_context.features)
        mean = self.actor_mean_head(head_features)
        logits = self.actor_binary_head(head_features)
        mode_logits = self.mode_logits_from_features(head_features)
        if target_context is not None and logits.shape[-1] > 0 and self.cfg.binary_action_dim > 0:
            fire_gate = (target_context.selected_visible * target_context.confidence).clamp(
                min=1.0e-3
            )
            logits = logits.clone()
            logits[:, 0] = logits[:, 0] + fire_gate.log()
        logits = self.gated_binary_logits(logits, mode_logits)
        return mean, logits, target_selection_logits

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

    def fire_valid_mask(self, obs: torch.Tensor) -> torch.Tensor | None:
        if not self.cfg.mask_fire_when_no_visible_enemy:
            return None
        _target, visible = aim_aux_targets(obs, self.cfg)
        return visible

    def masked_binary_logits(self, obs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        fire_valid = self.fire_valid_mask(obs)
        if fire_valid is None or logits.shape[-1] == 0:
            return logits
        masked = logits.clone()
        masked[:, 0] = masked[:, 0].masked_fill(~fire_valid, -1.0e9)
        return masked

    def sample_action(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, logits, target_logits, h_next = self.policy_outputs(obs, h)
        logits = self.masked_binary_logits(obs, logits)
        target_logits = self._masked_target_logits(target_logits, self._target_mask(obs))
        pieces: list[torch.Tensor] = []
        logprob = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        if self.cfg.continuous_action_dim > 0:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            u = mean + std * noise
            cont = torch.tanh(u)
            correction = 2.0 * (_LOG2 - u - torch.nn.functional.softplus(-2.0 * u))
            logprob = logprob + dist.log_prob(u).sum(-1) - correction.sum(-1)
            pieces.append(cont)
        if self.cfg.binary_action_dim > 0:
            binary_dist = torch.distributions.Bernoulli(logits=logits)
            probs = torch.sigmoid(logits)
            uniforms = torch.rand(
                probs.shape,
                dtype=probs.dtype,
                device=probs.device,
                generator=generator,
            )
            binary = (uniforms < probs).to(probs.dtype)
            logprob = logprob + binary_dist.log_prob(binary).sum(-1)
            pieces.append(binary)
        if self.cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            target_dist = torch.distributions.Categorical(logits=target_logits)
            target = torch.multinomial(
                torch.softmax(target_logits, dim=-1),
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            logprob = logprob + target_dist.log_prob(target)
            pieces.append(target.to(obs.dtype).unsqueeze(-1))
        return torch.cat(pieces, dim=-1), logprob, h_next

    def greedy_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _log_std, logits, target_logits, h_next = self.policy_outputs(obs, h)
        logits = self.masked_binary_logits(obs, logits)
        target_logits = self._masked_target_logits(target_logits, self._target_mask(obs))
        pieces = [torch.tanh(mean), (logits >= 0.0).to(obs.dtype)]
        if self.cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            pieces.append(target_logits.argmax(dim=-1).to(obs.dtype).unsqueeze(-1))
        action = torch.cat(pieces, dim=-1)
        return action, h_next


def aim_aux_targets(obs: torch.Tensor, cfg: MappoConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target enemy angle in radians and a visibility mask.

    Phase-4 flat actor observations expose the nearest enemy alive flag at
    index 10 and relative position at indices 12:14. The target is the angle
    from the agent to that visible enemy in the actor observation frame.
    """
    if cfg.obs_encoder != "flat":
        target = torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
        mask = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
        return target, mask
    enemy_alive = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    enemy_rel_pos = obs[:, _ENEMY_REL_POS_SLICE]
    rel_norm = torch.linalg.vector_norm(enemy_rel_pos, dim=-1)
    target = torch.atan2(enemy_rel_pos[:, 1], enemy_rel_pos[:, 0])
    return target, enemy_alive & (rel_norm > 1.0e-6)


def mode_aux_targets(
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heuristic mode labels: close visible enemy => combat, else objective."""
    labels = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    valid = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
    if cfg.obs_encoder == "flat":
        enemy_alive = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
        enemy_rel_pos = obs[:, _ENEMY_REL_POS_SLICE]
        close = torch.linalg.vector_norm(enemy_rel_pos, dim=-1) <= 0.4
        labels = torch.where(enemy_alive & close, torch.ones_like(labels), labels)
    if mask is not None:
        valid = valid & (mask.reshape(-1) > 0.0)
    return labels, valid


def mode_aux_loss_and_accuracy(
    logits: torch.Tensor | None,
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits is None:
        zero = obs.new_tensor(0.0)
        return zero, zero, zero
    if labels is None:
        labels, valid = mode_aux_targets(obs, cfg, mask=mask)
    else:
        labels = labels.to(device=obs.device, dtype=torch.long).reshape(-1)
        valid = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        if mask is not None:
            valid = valid & (mask.reshape(-1) > 0.0)
    count = valid.to(obs.dtype).sum()
    if float(count.item()) <= 0.0:
        zero = obs.new_tensor(0.0)
        return zero, zero, count
    loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
    acc = (logits[valid].argmax(dim=-1) == labels[valid]).to(obs.dtype).mean()
    return loss, acc, count


@dataclass(frozen=True)
class _TargetContext:
    features: torch.Tensor
    confidence: torch.Tensor
    selected_visible: torch.Tensor


def _target_candidate_tensors(
    obs: torch.Tensor, cfg: MappoConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    own_pos = obs[:, _OWN_POSITION_SLICE]
    enemy_alive = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    enemy_rel_pos = obs[:, _ENEMY_REL_POS_SLICE]
    enemy_hp = obs[:, _ENEMY_HP_SLICE].squeeze(-1)
    batch = obs.shape[0]
    if batch % cfg.n_agents == 0:
        groups = batch // cfg.n_agents
        own_group = own_pos.view(groups, cfg.n_agents, 2)
        enemy_pos = (own_pos + enemy_rel_pos).view(groups, cfg.n_agents, 2)
        rel = enemy_pos[:, None, :, :] - own_group[:, :, None, :]
        visible = enemy_alive.view(groups, cfg.n_agents)[:, None, :].expand(
            groups, cfg.n_agents, cfg.n_agents
        )
        hp = enemy_hp.view(groups, cfg.n_agents)[:, None, :].expand(
            groups, cfg.n_agents, cfg.n_agents
        )
        return (
            rel.reshape(batch, cfg.n_agents, 2),
            visible.reshape(batch, cfg.n_agents),
            hp.reshape(batch, cfg.n_agents),
            own_pos,
        )
    rel = enemy_rel_pos[:, None, :].expand(batch, cfg.n_agents, 2).clone()
    visible = enemy_alive[:, None].expand(batch, cfg.n_agents).clone()
    hp = enemy_hp[:, None].expand(batch, cfg.n_agents).clone()
    return rel, visible, hp, own_pos


def _target_selection_mask(
    visible: torch.Tensor,
    target_dim: int,
    *,
    row_visible: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_dim == visible.shape[-1]:
        fallback = torch.ones_like(visible, dtype=torch.bool)
        return torch.where(visible.any(dim=-1, keepdim=True), visible, fallback)
    if target_dim == visible.shape[-1] + 1:
        has_visible = visible.any(dim=-1, keepdim=True)
        if row_visible is not None:
            has_visible = has_visible & row_visible.view(-1, 1)
        no_target = ~has_visible
        enemy_mask = torch.where(has_visible, visible, torch.zeros_like(visible))
        return torch.cat((enemy_mask, no_target), dim=-1)
    raise ValueError(
        f"target_selection_dim must be {visible.shape[-1]} or {visible.shape[-1] + 1}, "
        f"got {target_dim}"
    )


def _masked_target_selection_logits(
    logits: torch.Tensor,
    visible: torch.Tensor,
    *,
    row_visible: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = _target_selection_mask(visible, logits.shape[-1], row_visible=row_visible)
    return logits.masked_fill(~mask, -1.0e9)


def target_selection_context(
    obs: torch.Tensor,
    cfg: MappoConfig,
    target_selection_logits: torch.Tensor | None,
) -> _TargetContext:
    rel, visible, _hp, _own_pos = _target_candidate_tensors(obs, cfg)
    if target_selection_logits is None:
        logits = torch.zeros(
            obs.shape[0], cfg.target_selection_dim, dtype=obs.dtype, device=obs.device
        )
    else:
        logits = target_selection_logits
    row_visible = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    masked_logits = _masked_target_selection_logits(logits, visible, row_visible=row_visible)
    if logits.shape[-1] == visible.shape[-1] + 1:
        weights = torch.softmax(masked_logits, dim=-1)[:, :-1]
    else:
        weights = torch.softmax(masked_logits, dim=-1)
    selected_rel = torch.sum(weights.unsqueeze(-1) * rel, dim=1)
    confidence = weights.max(dim=-1).values
    selected_visible = torch.sum(weights * visible.to(obs.dtype), dim=-1).clamp(0.0, 1.0)
    rel_norm = torch.linalg.vector_norm(selected_rel, dim=-1, keepdim=True)
    features = torch.cat(
        (
            selected_rel,
            selected_visible.unsqueeze(-1),
            confidence.unsqueeze(-1),
        ),
        dim=-1,
    )
    features = torch.cat((features[:, :2], features[:, 2:] * (rel_norm > 1.0e-6)), dim=-1)
    return _TargetContext(
        features=features,
        confidence=confidence,
        selected_visible=selected_visible,
    )


def target_selection_aux_targets(
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel, visible, hp, _own_pos = _target_candidate_tensors(obs, cfg)
    row_visible = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    valid = visible.any(dim=-1)
    if cfg.target_selection_aux_mode == "team_focus_low_hp":
        if cfg.target_selection_dim != cfg.n_agents + 1:
            raise ValueError("team_focus_low_hp requires an explicit no-target class")
        labels = torch.full(
            (obs.shape[0],),
            cfg.n_agents,
            dtype=torch.long,
            device=obs.device,
        )
        if obs.shape[0] % cfg.n_agents == 0:
            groups = obs.shape[0] // cfg.n_agents
            visible_g = visible.view(groups, cfg.n_agents, cfg.n_agents)[:, 0, :]
            hp_g = hp.view(groups, cfg.n_agents, cfg.n_agents)[:, 0, :]
            rel_g = rel.view(groups, cfg.n_agents, cfg.n_agents, 2)[:, 0, :, :]
            own_g = _own_pos.view(groups, cfg.n_agents, 2)[:, 0, :]
            enemy_pos_g = own_g[:, None, :] + rel_g
            dist_obj = torch.linalg.vector_norm(enemy_pos_g, dim=-1)
            score = 1.0 / hp_g.clamp(
                min=1.0e-6
            ) + cfg.target_selection_objective_proximity_coef / dist_obj.clamp(min=1.0e-6)
            score = score.masked_fill(~visible_g, -float("inf"))
            team_labels = score.argmax(dim=-1)
            has_team_target = visible_g.any(dim=-1)
            row_visible_g = row_visible.view(groups, cfg.n_agents)
            labels_g = labels.view(groups, cfg.n_agents)
            labels_g[:] = torch.where(
                has_team_target[:, None] & row_visible_g,
                team_labels[:, None],
                torch.full_like(labels_g, cfg.n_agents),
            )
            labels = labels_g.reshape(-1)
            valid = torch.ones_like(row_visible, dtype=torch.bool)
        else:
            score = 1.0 / hp.clamp(
                min=1.0e-6
            ) + cfg.target_selection_objective_proximity_coef / torch.linalg.vector_norm(
                _own_pos[:, None, :] + rel, dim=-1
            ).clamp(min=1.0e-6)
            score = score.masked_fill(~visible, -float("inf"))
            team_labels = score.argmax(dim=-1)
            labels = torch.where(row_visible & visible.any(dim=-1), team_labels, labels)
            valid = torch.ones_like(row_visible, dtype=torch.bool)
    elif cfg.target_selection_aux_mode == "lowest_hp":
        scores = hp.masked_fill(~visible, float("inf"))
        labels = scores.argmin(dim=-1)
    else:
        distances = torch.linalg.vector_norm(rel, dim=-1).masked_fill(~visible, float("inf"))
        labels = distances.argmin(dim=-1)
    if mask is not None:
        valid = valid & (mask.reshape(-1) > 0.0)
    return labels, valid


def target_selection_aux_metrics(
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    labels, valid = target_selection_aux_targets(obs, cfg, mask=mask)
    if mask is not None:
        valid = valid & (mask.reshape(-1) > 0.0)
    row_visible = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    focus_valid = valid & row_visible & (labels < cfg.n_agents)
    count = focus_valid.to(obs.dtype).sum()
    if float(count.item()) <= 0.0:
        zero = obs.new_tensor(0.0)
        return {
            "target_selection_label_entropy": zero,
            "target_selection_same_target_fraction": zero,
            "target_selection_fallback_rate": zero,
        }
    selected = labels[focus_valid]
    counts = torch.bincount(selected, minlength=cfg.n_agents).to(obs.dtype)
    probs = counts / counts.sum().clamp(min=1.0)
    positive = probs > 0.0
    entropy = -(probs[positive] * probs[positive].log()).sum()
    same_target_fraction = counts.max() / counts.sum().clamp(min=1.0)
    if obs.shape[0] % cfg.n_agents == 0:
        rel, _visible, _hp, own_pos = _target_candidate_tensors(obs, cfg)
        groups = obs.shape[0] // cfg.n_agents
        rel_g = rel.view(groups, cfg.n_agents, cfg.n_agents, 2)
        own_g = own_pos.view(groups, cfg.n_agents, 2)
        labels_g = labels.view(groups, cfg.n_agents)
        focus_g = focus_valid.view(groups, cfg.n_agents)
        agent_idx = torch.arange(cfg.n_agents, device=obs.device)
        own_seen_pos = own_g + rel_g[:, agent_idx, agent_idx, :]
        safe_labels = labels_g.clamp(0, cfg.n_agents - 1)
        selected_pos = own_g[:, 0, None, :] + rel_g[:, 0].gather(
            1, safe_labels[:, :, None].expand(groups, cfg.n_agents, 2)
        )
        fallback_mask = focus_g & (
            torch.linalg.vector_norm(own_seen_pos - selected_pos, dim=-1) > 1.0e-5
        )
        fallback_rate = fallback_mask.to(obs.dtype).sum() / count.clamp(min=1.0)
    else:
        fallback_rate = obs.new_tensor(0.0)
    return {
        "target_selection_label_entropy": entropy,
        "target_selection_same_target_fraction": same_target_fraction,
        "target_selection_fallback_rate": fallback_rate,
    }


def target_selection_policy_metrics(
    logits: torch.Tensor | None,
    obs: torch.Tensor,
    cfg: MappoConfig,
) -> dict[str, torch.Tensor]:
    if logits is None:
        zero = obs.new_tensor(0.0)
        return {
            "target_selection_policy_entropy": zero,
            "target_selection_policy_same_target_fraction": zero,
        }
    _rel, visible, _hp, _own_pos = _target_candidate_tensors(obs, cfg)
    row_visible = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    masked_logits = _masked_target_selection_logits(logits, visible, row_visible=row_visible)
    selected = masked_logits.argmax(dim=-1)
    valid = row_visible & (selected < cfg.n_agents)
    count = valid.to(obs.dtype).sum()
    if float(count.item()) <= 0.0:
        zero = obs.new_tensor(0.0)
        return {
            "target_selection_policy_entropy": zero,
            "target_selection_policy_same_target_fraction": zero,
        }
    counts = torch.bincount(selected[valid], minlength=cfg.n_agents).to(obs.dtype)
    probs = counts / counts.sum().clamp(min=1.0)
    positive = probs > 0.0
    entropy = -(probs[positive] * probs[positive].log()).sum()
    same_target_fraction = counts.max() / counts.sum().clamp(min=1.0)
    return {
        "target_selection_policy_entropy": entropy,
        "target_selection_policy_same_target_fraction": same_target_fraction,
    }


def target_selection_aux_loss_and_accuracy(
    logits: torch.Tensor | None,
    obs: torch.Tensor,
    cfg: MappoConfig,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits is None:
        zero = obs.new_tensor(0.0)
        return zero, zero, zero
    labels, valid = target_selection_aux_targets(obs, cfg, mask=mask)
    count = valid.to(obs.dtype).sum()
    if float(count.item()) <= 0.0:
        zero = obs.new_tensor(0.0)
        return zero, zero, count
    rel, visible, _hp, _own_pos = _target_candidate_tensors(obs, cfg)
    del rel, _hp, _own_pos
    row_visible = obs[:, _ENEMY_ALIVE_SLICE].squeeze(-1) > 0.5
    masked_logits = _masked_target_selection_logits(logits, visible, row_visible=row_visible)
    loss = torch.nn.functional.cross_entropy(masked_logits[valid], labels[valid])
    acc = (masked_logits[valid].argmax(dim=-1) == labels[valid]).to(obs.dtype).mean()
    return loss, acc, count


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
