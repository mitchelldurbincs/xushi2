from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from train.mappo_model import MappoActorCritic, MappoConfig, _OWN_POSITION_SLICE
from train.phases import resolve_phase
from train.ppo_recurrent.losses import _masked_mean, action_logprob_and_entropy
from xushi2.entity_obs import entity_obs_self_position
from xushi2.obs_manifest import actor_field_slice
from xushi2.vector_env import make_xushi_vector_env

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

    def set_team_spirit(self, value: float) -> None:
        """Push team_spirit value to every wrapped env via the vector wrapper.

        Envs whose reward calculator is in scalar mode silently ignore the
        update (their ``set_team_spirit`` is a no-op stash); only Phase 4+
        per-agent envs actually reweight their per-step rewards."""
        self.vec_env.set_team_spirit(float(value))

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
        team_spirit_initial=float(ppo_cfg.get("team_spirit_initial", 0.0)),
        team_spirit_final=float(ppo_cfg.get("team_spirit_final", 0.0)),
        team_spirit_ramp_fraction=float(
            ppo_cfg.get("team_spirit_ramp_fraction", 0.3)
        ),
    )


