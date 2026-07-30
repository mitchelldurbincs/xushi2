from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym

    from train.cap_duel_distill import CapDuelDistillAnchor, CapDuelDistillBatch

from train.device import resolve_device
from train.losses import (
    _masked_mean,
    action_logprob_and_entropy_parts,
)
from train.mappo_advantage import compute_gae
from train.mappo_metrics import rollout_metrics
from train.mappo_model import (
    MappoActorCritic,
    MappoConfig,
    aim_aux_loss_and_rmse,
    compute_anchor_kl_coef,
    mode_aux_loss_and_accuracy,
    target_selection_aux_loss_and_accuracy,
    target_selection_aux_metrics,
)
from train.mappo_rollout import collect_rollout, step_loss_mask
from train.recurrent_common import (
    apply_global_seeds,
    get_optimizer_learning_rate,
    grad_group_norm,
    next_update_sampling_state,
    set_optimizer_learning_rate,
)
from train.runtime_specs import resolve_runtime_spec
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
        # Episode boundaries that are true MDP terminals, tracked apart from
        # `done` so GAE can bootstrap time-limit truncations correctly.
        self.terminated = torch.zeros(N, L, device=dev)
        # V(s_T) for the pre-reset state, filled only on truncated steps.
        self.truncated_value = torch.zeros_like(self.value)
        self.h_init = torch.zeros(N, A, L, cfg.gru_hidden, device=dev)
        self.last_done = torch.zeros(N, device=dev)
        self.last_terminated = torch.zeros(N, device=dev)
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
        # Frozen copy of the policy at PPO start, used by the anchor-KL
        # penalty (cfg.anchor_kl_coef). Set via init_anchor_from_current_model
        # by the orchestration layer after warm start + pretrain stages.
        self.anchor_model: MappoActorCritic | None = None
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

    # --- resume ---------------------------------------------------------
    #
    # Model weights alone are not enough to continue a run. Restoring from
    # weights only zeroes the Adam moments and restarts the LR schedule at
    # update 1, a large silent optimization discontinuity that looks like "the
    # run got worse after restart". These two methods carry the rest.

    def resume_state(self) -> dict[str, Any]:
        """Everything besides model weights needed to continue this run."""
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_idx": int(self._active_update_idx),
            "update_counter": int(self._update_counter),
            "hidden_state": self.h.detach().cpu(),
            "policy_sampling_generator_state": self.policy_sampling_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "seed": int(self.seed),
        }

    def load_resume_state(self, state: dict[str, Any]) -> int:
        """Restore optimizer/RNG/hidden state. Returns the completed update index.

        Shapes are checked rather than trusted: resuming into a differently
        shaped run would otherwise fail deep inside the first update, or worse,
        broadcast silently.
        """
        expected_h = (self.cfg.num_envs, self.cfg.n_agents, self.cfg.gru_hidden)
        hidden = state.get("hidden_state")
        if hidden is not None:
            if tuple(hidden.shape) != expected_h:
                raise ValueError(
                    f"resume hidden_state shape {tuple(hidden.shape)} does not match this "
                    f"run's {expected_h}; num_envs/n_agents/gru_hidden must match to resume"
                )
            self.h = hidden.to(self.device)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._update_counter = int(state.get("update_counter", 0))
        gen_state = state.get("policy_sampling_generator_state")
        if gen_state is not None:
            self.policy_sampling_generator.set_state(gen_state)
        torch_state = state.get("torch_rng_state")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        numpy_state = state.get("numpy_rng_state")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        completed = int(state.get("update_idx", 0))
        self.set_update_index(completed)
        return completed

    def set_learning_rate(self, lr: float) -> None:
        set_optimizer_learning_rate(self.optimizer, lr)

    def set_update_index(self, update_idx: int) -> None:
        self._active_update_idx = int(update_idx)

    def set_cap_duel_distill_anchor(self, anchor: CapDuelDistillAnchor | None) -> None:
        self.cap_duel_distill_anchor = anchor

    def supported_curriculum_setters(self) -> frozenset[str]:
        """Curriculum knobs the underlying envs can actually apply."""
        return self.vec_env.supported_curriculum_setters()

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

    def set_respawn_ticks(self, respawn_ticks: int) -> None:
        self.vec_env.set_respawn_ticks(int(respawn_ticks))

    def init_anchor_from_current_model(self) -> None:
        """Freeze a copy of the current policy as the anchor-KL reference."""
        anchor = copy.deepcopy(self.model)
        anchor.eval()
        for param in anchor.parameters():
            param.requires_grad_(False)
        self.anchor_model = anchor

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
        return rollout_metrics(self.cfg, rollout, model=self.model)

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
        critic_warmup_active = self._active_update_idx <= cfg.critic_warmup_updates
        anchor_coef = compute_anchor_kl_coef(
            update=self._active_update_idx,
            initial=cfg.anchor_kl_coef,
            anneal_updates=cfg.anchor_kl_anneal_updates,
        )
        anchor_active = (
            self.anchor_model is not None and anchor_coef > 0.0 and not critic_warmup_active
        )
        anchor_kls: list[torch.Tensor] = []
        h_anchor = flat_h
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
            if anchor_active:
                anchor = self.anchor_model
                assert anchor is not None
                with torch.no_grad():
                    anchor_features, h_anchor = anchor.actor_head_features(obs_t, h_anchor)
                    anchor_mean, anchor_logits, _anchor_sel = anchor.policy_heads_from_features(
                        obs_t, anchor_features
                    )
                    anchor_logits = anchor.masked_binary_logits(obs_t, anchor_logits)
                    anchor_target_logits = (
                        anchor.actor_target_head(anchor_features)
                        if anchor.actor_target_head is not None
                        else None
                    )
                    anchor_target_logits = anchor._masked_target_logits(
                        anchor_target_logits, anchor._target_mask(obs_t)
                    )
                anchor_kls.append(
                    _anchor_action_kl(
                        mean=mean,
                        log_std=log_std,
                        binary_logits=logits,
                        target_logits=target_logits,
                        anchor_mean=anchor_mean,
                        anchor_log_std=anchor.log_std,
                        anchor_binary_logits=anchor_logits,
                        anchor_target_logits=anchor_target_logits,
                    ).view(N, A)
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
            if anchor_active:
                h_anchor = h_anchor.view(N, A, cfg.gru_hidden)
                h_anchor = (h_anchor * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)
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
        anchor_kl_mean = rollout.actor_obs.new_tensor(0.0)
        if anchor_active and anchor_kls:
            anchor_kl_mean = _masked_mean(torch.stack(anchor_kls, dim=2), valid_agent)
        if critic_warmup_active:
            # Critic warmup: fit the value function to the (frozen) warm-start
            # policy's returns before any policy gradient is applied. The
            # actor terms are excluded from the loss, so actor/trunk params
            # receive no gradient during warmup.
            total_loss = cfg.value_coef * value_loss
        else:
            total_loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - entropy_bonus
                + cfg.aim_aux_coef * aim_aux_loss
                + (cfg.mode_aux_coef * mode_aux_loss if cfg.mode_gated_combat else 0.0)
                + cfg.target_selection_aux_coef * target_aux_loss
            )
            if anchor_active:
                total_loss = total_loss + anchor_coef * anchor_kl_mean
        distill_metric_tensors: dict[str, torch.Tensor] = {}
        if distill_batch is not None and not critic_warmup_active:
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
        # Only emit the stabilizer metrics when the features are configured so
        # legacy runs keep their existing W&B schema unchanged.
        if cfg.critic_warmup_updates > 0:
            metrics["critic_warmup_active"] = 1.0 if critic_warmup_active else 0.0
        if cfg.anchor_kl_coef > 0.0:
            metrics["anchor_kl"] = float(anchor_kl_mean.item())
            metrics["anchor_kl_coef"] = float(anchor_coef)
        for key, value in distill_metric_tensors.items():
            metrics[f"distill/{key}"] = float(value.item())
        return metrics


def _anchor_action_kl(
    *,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    binary_logits: torch.Tensor,
    target_logits: torch.Tensor | None,
    anchor_mean: torch.Tensor,
    anchor_log_std: torch.Tensor,
    anchor_binary_logits: torch.Tensor,
    anchor_target_logits: torch.Tensor | None,
) -> torch.Tensor:
    """Analytic per-sample KL(pi_current || pi_anchor) over the action heads.

    The continuous heads are tanh-squashed Gaussians; KL is invariant under
    the shared invertible tanh transform, so the pre-squash Normal KL is
    exact. Binary heads use the Bernoulli KL with probabilities clamped away
    from 0/1 so fire-masked (-inf) logits stay finite. The optional target
    head uses the categorical KL with the same clamping rationale.
    """
    kl = mean.new_zeros(mean.shape[0])
    if mean.shape[-1] > 0:
        var = (2.0 * log_std).exp()
        anchor_var = (2.0 * anchor_log_std).exp()
        gauss = (
            anchor_log_std
            - log_std
            + (var + (mean - anchor_mean) ** 2) / (2.0 * anchor_var)
            - 0.5
        )
        kl = kl + gauss.sum(-1)
    if binary_logits.shape[-1] > 0:
        eps = 1e-6
        p = torch.sigmoid(binary_logits).clamp(eps, 1.0 - eps)
        q = torch.sigmoid(anchor_binary_logits).clamp(eps, 1.0 - eps)
        bern = p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()
        kl = kl + bern.sum(-1)
    if target_logits is not None and anchor_target_logits is not None:
        logp = F.log_softmax(target_logits, dim=-1).clamp(min=-30.0)
        logq = F.log_softmax(anchor_target_logits, dim=-1).clamp(min=-30.0)
        kl = kl + (logp.exp() * (logp - logq)).sum(-1)
    return kl


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
        critic_warmup_updates=int(ppo_cfg.get("critic_warmup_updates", 0)),
        anchor_kl_coef=float(ppo_cfg.get("anchor_kl_coef", 0.0)),
        anchor_kl_anneal_updates=int(ppo_cfg.get("anchor_kl_anneal_updates", 0)),
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
        "anchor_kl_coef",
    ):
        value = float(ppo_cfg.get(key, 0.0))
        if value < 0.0:
            raise ValueError(f"ppo.{key} must be non-negative, got {value!r}")

    critic_warmup_updates = int(ppo_cfg.get("critic_warmup_updates", 0))
    if critic_warmup_updates < 0:
        raise ValueError(
            f"ppo.critic_warmup_updates must be non-negative, got {critic_warmup_updates!r}"
        )
    anchor_kl_anneal_updates = int(ppo_cfg.get("anchor_kl_anneal_updates", 0))
    if anchor_kl_anneal_updates < 0:
        raise ValueError(
            f"ppo.anchor_kl_anneal_updates must be non-negative, got {anchor_kl_anneal_updates!r}"
        )

    team_spirit_ramp_fraction = float(ppo_cfg.get("team_spirit_ramp_fraction", 0.3))
    if not (0.0 <= team_spirit_ramp_fraction <= 1.0):
        raise ValueError(
            "ppo.team_spirit_ramp_fraction must satisfy 0 <= team_spirit_ramp_fraction <= 1, "
            f"got {team_spirit_ramp_fraction!r}"
        )
