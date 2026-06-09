from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym
    from train.cap_duel_distill import CapDuelDistillAnchor, CapDuelDistillBatch

from train.device import resolve_device
from train.mappo_advantage import compute_gae
from train.mappo_model import (
    _OWN_POSITION_SLICE,
    MappoActorCritic,
    MappoConfig,
    aim_aux_loss_and_rmse,
    mode_aux_loss_and_accuracy,
    target_selection_aux_loss_and_accuracy,
    target_selection_aux_metrics,
)
from train.mappo_rollout import collect_rollout, step_loss_mask
from train.losses import (
    _masked_mean,
    action_logprob_and_entropy_parts,
)
from train.recurrent_common import (
    apply_global_seeds,
    get_optimizer_learning_rate,
    grad_group_norm,
    next_update_sampling_state,
    set_optimizer_learning_rate,
)
from train.runtime_specs import resolve_runtime_spec
from xushi2.multi_enemy_obs import entity_obs_self_position
from xushi2.obs_manifest import actor_field_slice
from xushi2.vector_env import make_xushi_vector_env


class MappoRollout:
    def __init__(self, cfg: MappoConfig, device: torch.device | str | None = None) -> None:
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        self.device = resolve_device(cfg.device if device is None else device)
        dev = self.device
        self.actor_obs = torch.zeros(N, A, L, cfg.obs_dim, device=dev)
        if cfg.value_per_agent:
            self.critic_obs = torch.zeros(N, A, L, cfg.critic_obs_dim, device=dev)
            self.value = torch.zeros(N, A, L, device=dev)
            self.advantages = torch.zeros(N, A, L, device=dev)
            self.returns = torch.zeros(N, A, L, device=dev)
            self.last_value = torch.zeros(N, A, device=dev)
        else:
            self.critic_obs = torch.zeros(N, L, cfg.critic_obs_dim, device=dev)
            self.value = torch.zeros(N, L, device=dev)
            self.advantages = torch.zeros(N, L, device=dev)
            self.returns = torch.zeros(N, L, device=dev)
            self.last_value = torch.zeros(N, device=dev)
        self.action = torch.zeros(N, A, L, cfg.action_dim, device=dev)
        self.logprob = torch.zeros(N, A, L, device=dev)
        self.reward = torch.zeros(N, A, L, device=dev)
        self.done = torch.zeros(N, L, device=dev)
        self.h_init = torch.zeros(N, A, L, cfg.gru_hidden, device=dev)
        self.last_done = torch.zeros(N, device=dev)
        self.info_metrics: dict[str, float] = {}
        raw_mask = cfg.agent_loss_mask or tuple(1.0 for _ in range(A))
        self.agent_loss_mask = (
            torch.as_tensor(raw_mask, dtype=torch.float32, device=dev)
            .view(1, A, 1)
            .expand(N, A, L)
            .clone()
        )


class MappoTrainer:
    def __init__(self, env_fn: Callable[[], gym.Env], cfg: MappoConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.device = resolve_device(cfg.device)
        if cfg.torch_num_threads > 0:
            torch.set_num_threads(cfg.torch_num_threads)
        apply_global_seeds(self.seed)
        self.vec_env = make_xushi_vector_env(
            [env_fn for _ in range(cfg.num_envs)],
            critic_obs_dim=(
                cfg.critic_obs_dim * cfg.n_agents if cfg.value_per_agent else cfg.critic_obs_dim
            ),
            seed_base=self.seed,
            backend=cfg.vector_env,
        )
        obs, initial_critic_obs, _infos = self.vec_env.reset(seed=self.seed)
        self.last_obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.last_critic_obs = self._critic_obs_from_np(initial_critic_obs)
        apply_global_seeds(self.seed)
        self.model = MappoActorCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        set_optimizer_learning_rate(self.optimizer, cfg.learning_rate)
        self.h = self.model.init_hidden(cfg.num_envs * cfg.n_agents).view(
            cfg.num_envs, cfg.n_agents, cfg.gru_hidden
        )
        self.rollout_cls = MappoRollout
        self.policy_sampling_generator = torch.Generator(device=self.device.type)
        self.policy_sampling_generator.manual_seed(self.seed + 20_000)
        self._update_counter = 0
        self._active_update_idx = 0
        self.cap_duel_distill_anchor: CapDuelDistillAnchor | None = None
        self._actor_params: list[torch.nn.Parameter] = []
        self._critic_params: list[torch.nn.Parameter] = []
        self._trunk_params: list[torch.nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if (
                name.startswith(
                    (
                        "actor_body",
                        "actor_mean_head",
                        "actor_binary_head",
                        "actor_target_head",
                        "actor_target_selection_head",
                        "actor_target_condition",
                        "actor_aim_aux_head",
                        "actor_mode_head",
                    )
                )
                or name == "log_std"
            ):
                self._actor_params.append(p)
            elif name.startswith("critic"):
                self._critic_params.append(p)
            else:
                self._trunk_params.append(p)

    def close(self) -> None:
        self.vec_env.close()

    def set_learning_rate(self, lr: float) -> None:
        set_optimizer_learning_rate(self.optimizer, lr)

    def set_update_index(self, update_idx: int) -> None:
        self._active_update_idx = int(update_idx)

    def set_cap_duel_distill_anchor(self, anchor: CapDuelDistillAnchor | None) -> None:
        self.cap_duel_distill_anchor = anchor

    def set_team_spirit(self, value: float) -> None:
        """Push team_spirit value to every wrapped env via the vector wrapper.

        Envs whose reward calculator is in scalar mode silently ignore the
        update (their ``set_team_spirit`` is a no-op stash); only Phase 4+
        per-agent envs actually reweight their per-step rewards."""
        self.vec_env.set_team_spirit(float(value))

    def set_majority_on_point_alpha(self, value: float) -> None:
        self.vec_env.set_majority_on_point_alpha(float(value))

    def set_uncontested_on_point_alpha(self, value: float) -> None:
        self.vec_env.set_uncontested_on_point_alpha(float(value))

    def set_objective_timing_seconds(self, unlock_seconds: float, capture_seconds: float) -> None:
        self.vec_env.set_objective_timing_seconds(float(unlock_seconds), float(capture_seconds))

    @property
    def current_learning_rate(self) -> float:
        return get_optimizer_learning_rate(self.optimizer)

    @staticmethod
    def _group_grad_norm(params: list[torch.nn.Parameter]) -> float:
        return grad_group_norm(params)

    def _critic_obs_from_np(self, critic_obs_np: np.ndarray) -> torch.Tensor:
        critic_obs = torch.as_tensor(critic_obs_np, dtype=torch.float32, device=self.device)
        if self.cfg.value_per_agent:
            return critic_obs.view(self.cfg.num_envs, self.cfg.n_agents, self.cfg.critic_obs_dim)
        return critic_obs

    def collect_rollout(self) -> MappoRollout:
        return collect_rollout(self)

    def _step_loss_mask(self, infos: list[dict]) -> torch.Tensor:
        return step_loss_mask(self.cfg, infos)

    def update(self, rollout: MappoRollout) -> dict[str, float]:
        cfg = self.cfg
        compute_gae(rollout, cfg)
        rollout_metrics = self._rollout_metrics(rollout)
        if cfg.value_normalization:
            if cfg.value_per_agent:
                valid_agent = rollout.agent_loss_mask.expand_as(rollout.returns)
                ret_mean_t = _masked_mean(rollout.returns, valid_agent)
                ret_std_t = (
                    _masked_mean(
                        (rollout.returns - ret_mean_t) ** 2,
                        valid_agent,
                    )
                    .sqrt()
                    .clamp(min=1e-6)
                )
                ret_mean = float(ret_mean_t.item())
                ret_std = float(ret_std_t.item())
            else:
                ret_mean = float(rollout.returns.mean().item())
                ret_std = float(rollout.returns.std(unbiased=False).clamp(min=1e-6).item())
        else:
            ret_mean, ret_std = 0.0, 1.0

        distill_batch: CapDuelDistillBatch | None = None
        if self.cap_duel_distill_anchor is not None and self.cap_duel_distill_anchor.should_run(
            self._active_update_idx
        ):
            distill_batch = self.cap_duel_distill_anchor.collect_batch(
                update_idx=self._active_update_idx,
                device=self.device,
            )

        losses = []
        for _epoch in range(cfg.num_epochs):
            losses.append(
                self._update_full_rollout(
                    rollout,
                    ret_mean,
                    ret_std,
                    distill_batch=distill_batch,
                )
            )
        self._update_counter = next_update_sampling_state(
            self.seed, self._update_counter
        ).update_counter
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
                )
                .sqrt()
                .item()
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
                    (cont - _masked_mean(cont, agent_mask.unsqueeze(-1).expand_as(cont))) ** 2,
                    agent_mask.unsqueeze(-1).expand_as(cont),
                )
                .sqrt()
                .item()
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
        if cfg.mask_fire_when_no_visible_enemy:
            valid = self.model.fire_valid_mask(rollout.actor_obs.reshape(-1, cfg.obs_dim))
            if valid is not None:
                out["fire_valid_fraction"] = float(
                    _masked_mean(valid.to(agent_mask.dtype), agent_mask.reshape(-1)).item()
                )
        samples = float(rollout.info_metrics.get("info_metric_samples", 0.0))
        if samples > 0.0:
            for key, value in rollout.info_metrics.items():
                if key == "info_metric_samples":
                    continue
                out[f"rollout_{key}_mean"] = float(value) / samples
        return out

    def _action_logprob_and_entropy(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        binary_logits: torch.Tensor,
        target_logits: torch.Tensor | None,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        base_end = cfg.continuous_action_dim + cfg.binary_action_dim
        logp, move_ent, aim_ent, binary_ent = action_logprob_and_entropy_parts(
            mean,
            log_std,
            binary_logits,
            action[:, :base_end],
        )
        ent = move_ent + aim_ent + binary_ent
        if cfg.target_action_dim > 0:
            if target_logits is None:
                raise RuntimeError("target_action_dim requires target logits")
            target = action[:, base_end].long().clamp(0, cfg.target_action_dim - 1)
            dist = torch.distributions.Categorical(logits=target_logits)
            logp = logp + dist.log_prob(target)
            ent = ent + dist.entropy()
        return logp, ent, move_ent, aim_ent, binary_ent

    def _entropy_bonus(
        self,
        *,
        move_entropy: torch.Tensor,
        aim_entropy: torch.Tensor,
        binary_entropy: torch.Tensor,
        other_entropy: torch.Tensor,
        entropy: torch.Tensor,
        valid_agent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        move_mean = _masked_mean(move_entropy, valid_agent)
        aim_mean = _masked_mean(aim_entropy, valid_agent)
        binary_mean = _masked_mean(binary_entropy, valid_agent)
        other_mean = _masked_mean(other_entropy, valid_agent)
        if (
            cfg.entropy_coef_move is None
            and cfg.entropy_coef_aim is None
            and cfg.entropy_coef_binary is None
        ):
            bonus = cfg.entropy_coef * _masked_mean(entropy, valid_agent)
            return bonus, move_mean, aim_mean, binary_mean, other_mean
        move_coef = cfg.entropy_coef if cfg.entropy_coef_move is None else cfg.entropy_coef_move
        aim_coef = cfg.entropy_coef if cfg.entropy_coef_aim is None else cfg.entropy_coef_aim
        binary_coef = (
            cfg.entropy_coef if cfg.entropy_coef_binary is None else cfg.entropy_coef_binary
        )
        bonus = (
            move_coef * move_mean
            + aim_coef * aim_mean
            + binary_coef * binary_mean
            + cfg.entropy_coef * other_mean
        )
        return bonus, move_mean, aim_mean, binary_mean, other_mean

    def _update_full_rollout(
        self,
        rollout: MappoRollout,
        return_mean: float,
        return_std: float,
        *,
        distill_batch: CapDuelDistillBatch | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        flat_h = rollout.h_init[:, :, 0].reshape(N * A, cfg.gru_hidden)
        logprobs, entropies = [], []
        move_entropies, aim_entropies, binary_entropies = [], [], []
        aim_aux_losses, aim_aux_rmses, aim_aux_counts = [], [], []
        mode_aux_losses, mode_aux_accs, mode_aux_counts = [], [], []
        p_combat_parts = []
        target_aux_losses, target_aux_accs, target_aux_counts = [], [], []
        target_aux_metric_parts: dict[str, list[torch.Tensor]] = {
            "target_selection_label_entropy": [],
            "target_selection_same_target_fraction": [],
            "target_selection_fallback_rate": [],
        }
        h = flat_h
        for t in range(L):
            obs_t = rollout.actor_obs[:, :, t].reshape(N * A, cfg.obs_dim)
            features, h = self.model.actor_head_features(obs_t, h)
            mean, logits, target_selection_logits = self.model.policy_heads_from_features(
                obs_t, features
            )
            mode_logits = self.model.mode_logits_from_features(features)
            p_combat = self.model.combat_probability(mode_logits)
            if p_combat is not None:
                p_combat_parts.append(p_combat.mean())
            log_std = self.model.log_std
            logits = self.model.masked_binary_logits(obs_t, logits)
            target_logits = (
                self.model.actor_target_head(features)
                if self.model.actor_target_head is not None
                else None
            )
            target_logits = self.model._masked_target_logits(
                target_logits, self.model._target_mask(obs_t)
            )
            aim_pred = self.model.aim_aux_prediction_from_features(features)
            action_t = rollout.action[:, :, t].reshape(N * A, cfg.action_dim)
            logp, ent, move_ent, aim_ent, binary_ent = self._action_logprob_and_entropy(
                mean, log_std, logits, target_logits, action_t
            )
            logprobs.append(logp.view(N, A))
            entropies.append(ent.view(N, A))
            move_entropies.append(move_ent.view(N, A))
            aim_entropies.append(aim_ent.view(N, A))
            binary_entropies.append(binary_ent.view(N, A))
            flat_mask = rollout.agent_loss_mask[:, :, t].reshape(N * A)
            aim_loss, aim_rmse, aim_count = aim_aux_loss_and_rmse(
                aim_pred, obs_t, cfg, mask=flat_mask
            )
            aim_aux_losses.append(aim_loss)
            aim_aux_rmses.append(aim_rmse)
            aim_aux_counts.append(aim_count)
            mode_loss, mode_acc, mode_count = mode_aux_loss_and_accuracy(
                mode_logits, obs_t, cfg, mask=flat_mask
            )
            mode_aux_losses.append(mode_loss)
            mode_aux_accs.append(mode_acc)
            mode_aux_counts.append(mode_count)
            target_aux_loss, target_aux_acc, target_aux_count = (
                target_selection_aux_loss_and_accuracy(
                    target_selection_logits, obs_t, cfg, mask=flat_mask
                )
            )
            target_aux_losses.append(target_aux_loss)
            target_aux_accs.append(target_aux_acc)
            target_aux_counts.append(target_aux_count)
            for key, value in target_selection_aux_metrics(obs_t, cfg, mask=flat_mask).items():
                target_aux_metric_parts[key].append(value)
            done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
            h = h.view(N, A, cfg.gru_hidden)
            h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)
        new_logprob = torch.stack(logprobs, dim=2)
        entropy = torch.stack(entropies, dim=2)
        move_entropy = torch.stack(move_entropies, dim=2)
        aim_entropy = torch.stack(aim_entropies, dim=2)
        binary_entropy = torch.stack(binary_entropies, dim=2)
        other_entropy = entropy - move_entropy - aim_entropy - binary_entropy
        aim_aux_loss = torch.stack(aim_aux_losses).mean()
        aim_aux_count_total = torch.stack(aim_aux_counts).sum()
        if float(aim_aux_count_total.item()) > 0.0:
            aim_aux_rmse = torch.stack(aim_aux_rmses).mean()
        else:
            aim_aux_rmse = rollout.actor_obs.new_tensor(0.0)
        mode_aux_loss = torch.stack(mode_aux_losses).mean()
        mode_aux_count_total = torch.stack(mode_aux_counts).sum()
        if float(mode_aux_count_total.item()) > 0.0:
            mode_aux_acc = torch.stack(mode_aux_accs).mean()
        else:
            mode_aux_acc = rollout.actor_obs.new_tensor(0.0)
        mean_p_combat = (
            torch.stack(p_combat_parts).mean()
            if p_combat_parts
            else rollout.actor_obs.new_tensor(0.0)
        )
        target_aux_loss = torch.stack(target_aux_losses).mean()
        target_aux_count_total = torch.stack(target_aux_counts).sum()
        if float(target_aux_count_total.item()) > 0.0:
            target_aux_acc = torch.stack(target_aux_accs).mean()
        else:
            target_aux_acc = rollout.actor_obs.new_tensor(0.0)
        target_aux_metrics = {
            key: torch.stack(values).mean() if values else rollout.actor_obs.new_tensor(0.0)
            for key, values in target_aux_metric_parts.items()
        }
        valid_agent = rollout.agent_loss_mask.expand(N, A, L)
        if cfg.value_per_agent:
            value = (
                self.model.value(
                    rollout.critic_obs.permute(0, 2, 1, 3).reshape(N * L * A, cfg.critic_obs_dim)
                )
                .view(N, L, A)
                .permute(0, 2, 1)
            )
            advantage = rollout.advantages
        else:
            value = self.model.value(rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)).view(
                N, L
            )
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
        value_loss = _masked_mean(0.5 * torch.max(vl_unclipped, vl_clipped), value_mask)
        entropy_mean = _masked_mean(entropy, valid_agent)
        (
            entropy_bonus,
            move_entropy_mean,
            aim_entropy_mean,
            binary_entropy_mean,
            other_entropy_mean,
        ) = self._entropy_bonus(
            move_entropy=move_entropy,
            aim_entropy=aim_entropy,
            binary_entropy=binary_entropy,
            other_entropy=other_entropy,
            entropy=entropy,
            valid_agent=valid_agent,
        )
        total_loss = (
            policy_loss
            + cfg.value_coef * value_loss
            - entropy_bonus
            + cfg.aim_aux_coef * aim_aux_loss
            + (cfg.mode_aux_coef * mode_aux_loss if cfg.mode_gated_combat else 0.0)
            + cfg.target_selection_aux_coef * target_aux_loss
        )
        distill_metric_tensors: dict[str, torch.Tensor] = {}
        if distill_batch is not None:
            if self.cap_duel_distill_anchor is None:
                raise RuntimeError("distill_batch provided without configured anchor")
            distill_scaled_loss, distill_metric_tensors = (
                self.cap_duel_distill_anchor.loss_for_model(self.model, distill_batch)
            )
            total_loss = total_loss + distill_scaled_loss

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
        metrics = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_mean.item()),
            "entropy_move": float(move_entropy_mean.item()),
            "entropy_aim": float(aim_entropy_mean.item()),
            "entropy_binary": float(binary_entropy_mean.item()),
            "entropy_other": float(other_entropy_mean.item()),
            "entropy_bonus": float(entropy_bonus.item()),
            "approx_kl": float(approx_kl.item()),
            "clip_fraction": float(clip_fraction.item()),
            "total_loss": float(total_loss.item()),
            "aim_aux_loss": float(aim_aux_loss.item()),
            "aim_aux_rmse": float(aim_aux_rmse.item()),
            "aim_aux_count": float(aim_aux_count_total.item()),
            "mode_aux_loss": float(mode_aux_loss.item()),
            "mode_accuracy": float(mode_aux_acc.item()),
            "mode_aux_count": float(mode_aux_count_total.item()),
            "mean_p_combat": float(mean_p_combat.item()),
            "target_selection_aux_loss": float(target_aux_loss.item()),
            "target_selection_aux_accuracy": float(target_aux_acc.item()),
            "target_selection_aux_count": float(target_aux_count_total.item()),
            "target_selection_label_entropy": float(
                target_aux_metrics["target_selection_label_entropy"].item()
            ),
            "target_selection_same_target_fraction": float(
                target_aux_metrics["target_selection_same_target_fraction"].item()
            ),
            "target_selection_fallback_rate": float(
                target_aux_metrics["target_selection_fallback_rate"].item()
            ),
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "trunk_grad_norm": trunk_grad_norm,
            "lr": self.current_learning_rate,
        }
        for key, value in distill_metric_tensors.items():
            metrics[f"distill/{key}"] = float(value.item())
        return metrics


def make_mappo_config(config: dict) -> MappoConfig:
    runtime = resolve_runtime_spec(config)
    if runtime.learner.kind != "mappo":
        raise ValueError(
            f"MAPPO trainer requires learner.kind='mappo', got {runtime.learner.kind!r}"
        )
    if runtime.shapes.critic_obs_dim is None:
        raise ValueError("MAPPO trainer requires a centralized critic observation spec")
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    run_cfg = config.get("run", {})
    # ``run.device`` wins (matches other infra-level toggles); ``ppo.device``
    # is accepted as a fallback for symmetry with the rest of the PPO block.
    device = str(run_cfg.get("device", ppo_cfg.get("device", "cpu")))
    obs_encoder = str(model_cfg.get("obs_encoder", "flat"))
    n_agents = int(runtime.shapes.n_agents)
    raw_agent_loss_mask = ppo_cfg.get("agent_loss_mask", [1.0] * n_agents)
    agent_loss_mask = tuple(float(v) for v in raw_agent_loss_mask)
    if len(agent_loss_mask) != n_agents:
        raise ValueError(
            f"ppo.agent_loss_mask length must be {n_agents}, got {len(agent_loss_mask)}"
        )
    if any(v < 0.0 for v in agent_loss_mask):
        raise ValueError("ppo.agent_loss_mask values must be non-negative")
    if not any(v > 0.0 for v in agent_loss_mask):
        raise ValueError("ppo.agent_loss_mask must leave at least one active agent")
    _validate_mappo_hyperparameters(ppo_cfg)
    return MappoConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        n_agents=n_agents,
        agent_loss_mask=agent_loss_mask,
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=int(runtime.shapes.obs_dim),
        critic_obs_dim=int(runtime.shapes.critic_obs_dim),
        action_dim=int(runtime.shapes.action_dim),
        continuous_action_dim=int(runtime.shapes.continuous_action_dim),
        binary_action_dim=int(runtime.shapes.binary_action_dim),
        target_action_dim=int(runtime.shapes.target_action_dim),
        value_per_agent=bool(ppo_cfg.get("value_per_agent", False)),
        mask_fire_when_no_visible_enemy=bool(ppo_cfg.get("mask_fire_when_no_visible_enemy", False)),
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
        entropy_coef_move=(
            None
            if ppo_cfg.get("entropy_coef_move") is None
            else float(ppo_cfg["entropy_coef_move"])
        ),
        entropy_coef_aim=(
            None if ppo_cfg.get("entropy_coef_aim") is None else float(ppo_cfg["entropy_coef_aim"])
        ),
        entropy_coef_binary=(
            None
            if ppo_cfg.get("entropy_coef_binary") is None
            else float(ppo_cfg["entropy_coef_binary"])
        ),
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
        team_spirit_initial=float(ppo_cfg.get("team_spirit_initial", 0.0)),
        team_spirit_final=float(ppo_cfg.get("team_spirit_final", 0.0)),
        team_spirit_ramp_fraction=float(ppo_cfg.get("team_spirit_ramp_fraction", 0.3)),
        aim_aux_coef=float(ppo_cfg.get("aim_aux_coef", 0.0)),
        mode_gated_combat=bool(ppo_cfg.get("mode_gated_combat", False)),
        mode_aux_coef=float(ppo_cfg.get("mode_aux_coef", 0.3)),
        target_selection_dim=int(ppo_cfg.get("target_selection_dim", 0)),
        target_conditioned_combat=bool(ppo_cfg.get("target_conditioned_combat", False)),
        target_selection_aux_coef=float(ppo_cfg.get("target_selection_aux_coef", 0.0)),
        target_selection_aux_mode=str(ppo_cfg.get("target_selection_aux_mode", "nearest_visible")),
        target_selection_objective_proximity_coef=float(
            ppo_cfg.get("target_selection_objective_proximity_coef", 0.1)
        ),
        device=device,
    )


def _validate_mappo_hyperparameters(ppo_cfg: dict) -> None:
    gamma = float(ppo_cfg["gamma"])
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"ppo.gamma must satisfy 0 < gamma <= 1, got {gamma!r}")

    gae_lambda = float(ppo_cfg["gae_lambda"])
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"ppo.gae_lambda must satisfy 0 <= gae_lambda <= 1, got {gae_lambda!r}")

    clip_ratio = float(ppo_cfg["clip_ratio"])
    if clip_ratio <= 0.0:
        raise ValueError(f"ppo.clip_ratio must be > 0, got {clip_ratio!r}")

    value_clip_ratio = float(ppo_cfg["value_clip_ratio"])
    if value_clip_ratio <= 0.0:
        raise ValueError(f"ppo.value_clip_ratio must be > 0, got {value_clip_ratio!r}")

    for key in (
        "entropy_coef",
        "value_coef",
        "aim_aux_coef",
        "mode_aux_coef",
        "target_selection_aux_coef",
    ):
        value = float(ppo_cfg.get(key, 0.0))
        if value < 0.0:
            raise ValueError(f"ppo.{key} must be non-negative, got {value!r}")

    team_spirit_ramp_fraction = float(ppo_cfg.get("team_spirit_ramp_fraction", 0.3))
    if not (0.0 <= team_spirit_ramp_fraction <= 1.0):
        raise ValueError(
            "ppo.team_spirit_ramp_fraction must satisfy 0 <= team_spirit_ramp_fraction <= 1, "
            f"got {team_spirit_ramp_fraction!r}"
        )
