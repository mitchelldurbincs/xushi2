"""Phase-4 recurrent MAPPO trainer.

This is the first CTDE training path: a shared recurrent actor consumes
per-agent actor observations while a centralized critic consumes the
team-level critic observation supplied by ``Phase4MappoEnv``.
"""

from __future__ import annotations

import copy
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
    lr_schedule: str = "constant"
    lr_final_ratio: float = 1.0
    warmup_updates: int = 0
    value_normalization: bool = True
    torch_num_threads: int = 0


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
        self.actor_embed = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.embed_dim),
            nn.ReLU(),
        )
        self.actor_gru = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.actor_body = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.head_hidden),
            nn.ReLU(),
        )
        self.actor_mean_head = nn.Linear(cfg.head_hidden, cfg.continuous_action_dim)
        self.actor_binary_head = nn.Linear(cfg.head_hidden, cfg.binary_action_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.actor_embed(obs)
        h_next = self.actor_gru(emb, h)
        features = self.actor_body(h_next)
        mean = self.actor_mean_head(features)
        logits = self.actor_binary_head(features)
        return mean, self.log_std, logits, h_next

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def sample_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, logits, h_next = self.policy_outputs(obs, h)
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
        return torch.cat(pieces, dim=-1), logprob, h_next

    def greedy_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, _log_std, logits, h_next = self.policy_outputs(obs, h)
        action = torch.cat((torch.tanh(mean), (logits >= 0.0).to(obs.dtype)), dim=-1)
        return action, h_next


class MappoRollout:
    def __init__(self, cfg: MappoConfig) -> None:
        N, A, L = cfg.num_envs, cfg.n_agents, cfg.rollout_len
        self.actor_obs = torch.zeros(N, A, L, cfg.obs_dim)
        self.critic_obs = torch.zeros(N, L, cfg.critic_obs_dim)
        self.action = torch.zeros(N, A, L, cfg.action_dim)
        self.logprob = torch.zeros(N, A, L)
        self.reward = torch.zeros(N, A, L)
        self.value = torch.zeros(N, L)
        self.done = torch.zeros(N, L)
        self.h_init = torch.zeros(N, A, L, cfg.gru_hidden)
        self.advantages = torch.zeros(N, L)
        self.returns = torch.zeros(N, L)
        self.last_value = torch.zeros(N)
        self.last_done = torch.zeros(N)

    def compute_gae(self, cfg: MappoConfig) -> None:
        reward = self.reward.mean(dim=1)
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
        self.envs = [env_fn() for _ in range(cfg.num_envs)]
        obs = []
        for i, env in enumerate(self.envs):
            obs_i, _ = env.reset(seed=self.seed + i)
            obs.append(obs_i)
        self.last_obs = torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32)
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
            if name.startswith(("actor_body", "actor_mean_head", "actor_binary_head")):
                self._actor_params.append(p)
            elif name == "log_std":
                self._actor_params.append(p)
            elif name.startswith("critic"):
                self._critic_params.append(p)
            else:
                self._trunk_params.append(p)

    def close(self) -> None:
        for env in self.envs:
            env.close()

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
        out = np.zeros((self.cfg.num_envs, self.cfg.critic_obs_dim), dtype=np.float32)
        for i, env in enumerate(self.envs):
            env.build_critic_obs(out[i])
        return torch.as_tensor(out, dtype=torch.float32)

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
                value = self.model.value(critic_obs)
            action_3d = action.view(cfg.num_envs, cfg.n_agents, cfg.action_dim)
            action_np = action_3d.cpu().numpy()
            next_obs_parts = []
            reward_parts = []
            done_parts = []
            for e, env in enumerate(self.envs):
                next_obs_e, reward_e, terminated, truncated, _info = env.step(
                    action_np[e]
                )
                done = bool(terminated or truncated)
                reward_parts.append(reward_e)
                done_parts.append(done)
                if done:
                    reset_seed = (
                        self.seed
                        + 10_000 * (self._update_counter + 1)
                        + t * cfg.num_envs
                        + e
                    )
                    next_obs_e, _ = env.reset(seed=reset_seed)
                next_obs_parts.append(next_obs_e)
            done_np = np.asarray(done_parts, dtype=np.bool_)
            rollout.actor_obs[:, :, t] = obs
            rollout.critic_obs[:, t] = critic_obs
            rollout.action[:, :, t] = action_3d
            rollout.logprob[:, :, t] = logprob.view(cfg.num_envs, cfg.n_agents)
            rollout.reward[:, :, t] = torch.as_tensor(
                np.stack(reward_parts, axis=0), dtype=torch.float32
            )
            rollout.value[:, t] = value
            rollout.done[:, t] = torch.as_tensor(done_np, dtype=torch.float32)
            h = h_next.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
            rollout.h_init[:, :, t] = flat_h.view(cfg.num_envs, cfg.n_agents, cfg.gru_hidden)
            for e, done in enumerate(done_np):
                if bool(done):
                    h[e] = 0.0
            obs = torch.as_tensor(np.stack(next_obs_parts, axis=0), dtype=torch.float32)
        with torch.no_grad():
            rollout.last_value = self.model.value(self._critic_obs())
        rollout.last_done = rollout.done[:, -1].clone()
        self.last_obs = obs
        self.h = h
        return rollout

    def update(self, rollout: MappoRollout) -> dict[str, float]:
        cfg = self.cfg
        rollout.compute_gae(cfg)
        rollout_metrics = self._rollout_metrics(rollout)
        if cfg.value_normalization:
            ret_mean = float(rollout.returns.mean().item())
            ret_std = float(rollout.returns.std(unbiased=False).clamp(min=1e-6).item())
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
        move_mag = torch.linalg.vector_norm(action[:, :, :, 0:2], dim=-1)
        cont = action[:, :, :, : cfg.continuous_action_dim]
        binary = action[:, :, :, cfg.continuous_action_dim :]

        self_on_point_slice = actor_field_slice("self_on_point")
        own_pos = rollout.actor_obs[:, :, :, _OWN_POSITION_SLICE]
        distance_to_objective = torch.linalg.vector_norm(own_pos, dim=-1)
        self_on_point = rollout.actor_obs[:, :, :, self_on_point_slice]

        out = {
            "rollout_reward_mean": float(reward.mean().item()),
            "rollout_reward_std": float(reward.std(unbiased=False).item()),
            "rollout_reward_min": float(reward.min().item()),
            "rollout_reward_max": float(reward.max().item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "advantage_min": float(advantages.min().item()),
            "advantage_max": float(advantages.max().item()),
            "return_mean": float(returns.mean().item()),
            "return_std": float(returns.std(unbiased=False).item()),
            "action_move_mag_mean": float(move_mag.mean().item()),
            "action_cont_mean": float(cont.mean().item()),
            "action_cont_std": float(cont.std(unbiased=False).item()),
            "mean_distance_to_objective": float(distance_to_objective.mean().item()),
            "self_on_point_fraction": float(self_on_point.mean().item()),
        }
        if binary.numel() > 0:
            out["action_binary_mean"] = float(binary.mean().item())
        else:
            out["action_binary_mean"] = 0.0
        return out

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
            mean, log_std, logits, h = self.model.policy_outputs(obs_t, h)
            action_t = rollout.action[:, :, t].reshape(N * A, cfg.action_dim)
            logp, ent = action_logprob_and_entropy(mean, log_std, logits, action_t)
            logprobs.append(logp.view(N, A))
            entropies.append(ent.view(N, A))
            done_mask = rollout.done[:, t].view(N, 1, 1).expand(N, A, cfg.gru_hidden)
            h = h.view(N, A, cfg.gru_hidden)
            h = (h * (1.0 - done_mask)).reshape(N * A, cfg.gru_hidden)
        new_logprob = torch.stack(logprobs, dim=2)
        entropy = torch.stack(entropies, dim=2)
        value = self.model.value(
            rollout.critic_obs.reshape(N * L, cfg.critic_obs_dim)
        ).view(N, L)

        valid_team = torch.ones(N, L)
        valid_agent = valid_team[:, None, :].expand(N, A, L)
        advantage = rollout.advantages[:, None, :].expand(N, A, L)
        adv_mean = _masked_mean(advantage, valid_agent)
        adv_var = _masked_mean((advantage - adv_mean) ** 2, valid_agent)
        norm_adv = (advantage - adv_mean) / adv_var.clamp(min=1e-8).sqrt()

        ratio = (new_logprob - rollout.logprob).exp()
        pg1 = ratio * norm_adv
        pg2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * norm_adv
        policy_loss = _masked_mean(-torch.min(pg1, pg2), valid_agent)

        value_n = (value - return_mean) / return_std
        old_value_n = (rollout.value - return_mean) / return_std
        return_n = (rollout.returns - return_mean) / return_std
        value_clipped_n = old_value_n + torch.clamp(
            value_n - old_value_n, -cfg.value_clip_ratio, cfg.value_clip_ratio
        )
        vl_unclipped = (value_n - return_n) ** 2
        vl_clipped = (value_clipped_n - return_n) ** 2
        value_loss = _masked_mean(
            0.5 * torch.max(vl_unclipped, vl_clipped), valid_team
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
    if phase != 4:
        raise ValueError(f"MAPPO trainer only supports phase 4, got phase={phase!r}")
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    return MappoConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        n_agents=int(phase_spec["n_agents"]),
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=int(phase_spec["obs_dim"]),
        critic_obs_dim=int(phase_spec["critic_obs_dim"]),
        action_dim=int(phase_spec["action_dim"]),
        continuous_action_dim=int(phase_spec["continuous_action_dim"]),
        binary_action_dim=int(phase_spec["binary_action_dim"]),
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
    )


def _walk_to_objective_targets(obs: torch.Tensor, cfg: MappoConfig) -> torch.Tensor:
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
            mean, _log_std, logits, h = model.policy_outputs(obs_seq[t], h)
            pred_cont = torch.tanh(mean)
            target = target_seq[t]
            cont_losses.append(
                torch.nn.functional.mse_loss(
                    pred_cont, target[:, : cfg.continuous_action_dim]
                )
            )
            binary_losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target[:, cfg.continuous_action_dim :]
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
                f"[phase4/mappo] bc_pretrain step={step}/{steps} "
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


def train_phase4_from_config(config: dict) -> dict[str, float]:
    _, phase_spec = resolve_phase(config)
    env_fn, ckpt_env_cfg, seed_base = phase_spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    run_cfg = config.get("run", {})
    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 10))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    output_dir = Path(str(run_cfg.get("output_dir", "runs/phase4_mappo"))) / "mappo"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            )
            eval_stats = evaluate_mappo(
                trainer.model,
                env_fn,
                episodes=eval_episodes,
                seed=seed_base + 90_000,
            )
            print(
                f"[phase4/mappo] bc_eval mean_reward={eval_stats.mean_reward:+.3f} "
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
                    f"[phase4/mappo] update={update_idx}/{total_updates} "
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
                    f"[phase4/mappo] eval update={update_idx}/{total_updates} "
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
            if update_idx % checkpoint_every == 0 or update_idx == total_updates:
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict(),
                        "config": {
                            "phase": 4,
                            "env": ckpt_env_cfg,
                            "mappo": cfg.__dict__,
                        },
                    },
                    output_dir / f"ckpt_{update_idx:04d}.pt",
                )
    finally:
        trainer.close()
    torch.save(
        {
            "model_state_dict": (
                best_state if best_state is not None else trainer.model.state_dict()
            ),
            "config": {"phase": 4, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        output_dir / "ckpt_final.pt",
    )
    return {"mappo": best_eval if best_eval > float("-inf") else last_eval}
