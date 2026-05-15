from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym

from train.device import resolve_device
from train.mappo_advantage import compute_gae
from train.mappo_model import MappoActorCritic, MappoConfig
from train.phases import resolve_phase
from train.mappo_metrics import rollout_metrics
from train.mappo_rollout import collect_rollout, step_loss_mask
from train.mappo_update import update_full_rollout
from train.ppo_recurrent.losses import _masked_mean
from train.recurrent_common import (
    apply_global_seeds,
    get_optimizer_learning_rate,
    grad_group_norm,
    next_update_sampling_state,
    set_optimizer_learning_rate,
)
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
        self._sampling_rng_state = torch.get_rng_state()
        self._update_counter = 0
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

    def set_team_spirit(self, value: float) -> None:
        """Push team_spirit value to every wrapped env via the vector wrapper.

        Envs whose reward calculator is in scalar mode silently ignore the
        update (their ``set_team_spirit`` is a no-op stash); only Phase 4+
        per-agent envs actually reweight their per-step rewards."""
        self.vec_env.set_team_spirit(float(value))

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

        losses = []
        for _epoch in range(cfg.num_epochs):
            losses.append(self._update_full_rollout(rollout, ret_mean, ret_std))
        self._update_counter = next_update_sampling_state(
            self.seed, self._update_counter
        ).update_counter
        metrics = {k: float(np.mean([m[k] for m in losses])) for k in losses[0]}
        metrics.update(rollout_metrics)
        return metrics

    def _rollout_metrics(self, rollout: MappoRollout) -> dict[str, float]:
        return rollout_metrics(self.cfg, rollout)

    def _update_full_rollout(
        self, rollout: MappoRollout, return_mean: float, return_std: float
    ) -> dict[str, float]:
        return update_full_rollout(self, rollout, return_mean, return_std)


def make_mappo_config(config: dict) -> MappoConfig:
    phase, phase_spec = resolve_phase(config)
    if phase not in (4, 5, 6, 7, 8, 9, 10, 11):
        raise ValueError(f"MAPPO trainer only supports phases 4-11, got phase={phase!r}")
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    run_cfg = config.get("run", {})
    # ``run.device`` wins (matches other infra-level toggles); ``ppo.device``
    # is accepted as a fallback for symmetry with the rest of the PPO block.
    device = str(run_cfg.get("device", ppo_cfg.get("device", "cpu")))
    obs_encoder = str(model_cfg.get("obs_encoder", "flat"))
    n_agents = int(phase_spec["n_agents"])
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
        team_spirit_ramp_fraction=float(ppo_cfg.get("team_spirit_ramp_fraction", 0.3)),
        device=device,
    )
