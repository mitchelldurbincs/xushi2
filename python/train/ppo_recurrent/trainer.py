"""Recurrent PPO trainer (CleanRL-style) for xushi2 Phase-2.

See the package ``__init__`` docstring for the invariant contract this
class implements.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from train.models import ActorCritic, build_model
from train.rollout_buffer import RolloutBuffer

from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent import ppo_updater, rollout_collector


class PPOTrainer:
    """Recurrent PPO trainer satisfying the Task 5 invariant contract."""

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        config: PPOConfig,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.seed = int(seed)

        # Optional: pin PyTorch's intra-op thread count. With 16x small
        # tensors the BLAS-threading overhead dominates useful work; set
        # to 1 in configs that use ``AsyncVectorEnv`` so main-process
        # PyTorch stays out of the worker subprocesses' way.
        if config.torch_num_threads > 0:
            torch.set_num_threads(config.torch_num_threads)

        # --- Initial global RNG seeding. Applied early so env/space
        # construction is deterministic.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- Vectorized env. SyncVectorEnv calls the thunk once per env.
        # AsyncVectorEnv forks/spawns a subprocess per env — each subproc
        # runs its own sim step in parallel, which is the big win on
        # multi-core boxes. Seeding via ``reset(seed=seed)`` fans out as
        # ``[seed, seed+1, ..., seed+num_envs-1]`` per env in both.
        vec_cls: type = AsyncVectorEnv if config.vector_env == "async" else SyncVectorEnv
        self.envs = vec_cls(
            [env_fn for _ in range(config.num_envs)]
        )
        obs, _ = self.envs.reset(seed=self.seed)
        self._last_obs = torch.as_tensor(obs, dtype=torch.float32)

        # --- Re-seed torch AFTER env construction. Gymnasium's env and
        # space initialization consume an unspecified amount of torch RNG
        # state (notably inside ``Box.sample`` machinery and any
        # registration hook), so seeding only before env creation leaves
        # the RNG in a state that depends on gym internals. Re-seeding
        # here guarantees model init and action sampling are
        # deterministic regardless of how many env-related calls happened
        # upstream. Numpy seed re-applied too for symmetry in case any
        # future trainer code adds a numpy-RNG decision path.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- Model + optimizer.
        self.model: ActorCritic = build_model(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            continuous_action_dim=config.continuous_action_dim,
            binary_action_dim=config.binary_action_dim,
            use_recurrence=config.use_recurrence,
            embed_dim=config.embed_dim,
            gru_hidden=config.gru_hidden,
            head_hidden=config.head_hidden,
            action_log_std_init=config.action_log_std_init,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )
        self.set_learning_rate(config.learning_rate)

        # --- Per-trainer action-sampling RNG state. We use the GLOBAL
        # torch RNG for model sampling (since ``Normal.rsample`` has no
        # generator kwarg in older torch versions), but save/restore a
        # trainer-local snapshot around every sampling call so two
        # trainers do not interfere with each other's global-RNG advance.
        # Initial state is the snapshot of the global RNG AFTER all init
        # seeding is done — so two trainers with the same seed start at
        # byte-identical RNG state.
        self._sampling_rng_state: torch.Tensor = torch.get_rng_state()

        # Grouped param lists for per-head grad-norm instrumentation. The
        # shared trunk (embed + GRU/ff) gets gradient from both actor and
        # critic losses; the head groups are disjoint.
        self._actor_params: list[torch.nn.Parameter] = []
        self._critic_params: list[torch.nn.Parameter] = []
        self._trunk_params: list[torch.nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if name.startswith(("actor_body", "actor_mean_head", "actor_binary_head")) or name == "log_std":
                self._actor_params.append(p)
            elif name.startswith("critic_head"):
                self._critic_params.append(p)
            else:
                self._trunk_params.append(p)

        # --- Live hidden state carried across ``collect_rollout`` calls.
        self.h: torch.Tensor = self.model.init_hidden(config.num_envs)

        # NOTE: rollout-end bootstrap state (``last_value`` / ``last_done``)
        # is attached DIRECTLY to the buffer by ``collect_rollout`` so that
        # a ``deepcopy(rollout)`` carries it along and ``update(copy)``
        # produces identical results. See ``collect_rollout`` and
        # ``update`` for the read/write sites.

        # Counter used to deterministically seed the minibatch-shuffle
        # generator inside ``update``. Incremented per ``update`` call.
        self._update_counter: int = 0

        # NOTE: ``self._training_h_init_log`` is NOT created by default.
        # Tests opt in by setting it to ``[]`` before calling ``update``.
        # ``update`` guards with ``hasattr`` before appending.

    def set_learning_rate(self, lr: float) -> None:
        """Apply ``lr`` to every optimizer param group."""
        lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _group_grad_norm(params: list[torch.nn.Parameter]) -> float:
        total_sq = 0.0
        for p in params:
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum().item())
        return float(total_sq ** 0.5)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def _make_buffer(self) -> RolloutBuffer:
        cfg = self.config
        return RolloutBuffer(
            num_envs=cfg.num_envs,
            rollout_len=cfg.rollout_len,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            gru_hidden=cfg.gru_hidden,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device="cpu",
        )

    def collect_rollout(self) -> RolloutBuffer:
        """Roll out ``config.rollout_len`` ticks across the vector env."""
        return rollout_collector.collect_rollout(self)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, rollout: RolloutBuffer) -> dict:
        """Run PPO updates on ``rollout`` and return metrics."""
        return ppo_updater.update_ppo(self, rollout)

