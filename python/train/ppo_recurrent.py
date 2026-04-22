"""CleanRL-style recurrent PPO trainer for xushi2 Phase-2.

Implements the invariant contract defined by
``python/tests/test_ppo_recurrent_invariants.py``:

* Seed-deterministic rollouts (Test 1).
* Hidden-state zeroing on episode reset (Test 2).
* Identical per-segment ``h_init`` across all PPO epochs within an update
  (Test 3). Implemented by seeding the minibatch-shuffle generator once
  per update and reusing it across epochs.
* Feedforward-mode path that routes no gradient through ``h_init``
  (Test 4). The model handles this structurally; the trainer just feeds
  ``h_init`` in uniformly.
* ``valid_mask``-aware loss normalization — policy, value, and entropy
  terms all multiply by ``valid_mask`` and divide by ``valid_mask.sum()``
  (Test 5).

Design notes:
* CPU-only for Phase-2; the Phase-2 toy is tiny and the determinism tests
  assume CPU.
* Action log-probs are recomputed at training time from the stored
  squashed action by inverting tanh (atanh) with an eps-clamp.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Normal

from train.models import ActorCritic, build_model
from train.rollout_buffer import RolloutBuffer


# Numerical guard for atanh(action) reconstruction. Pulled out as a module
# constant so tests/callers can reason about it.
_ATANH_EPS = 1e-6
_LOG2 = math.log(2.0)


@dataclass
class PPOConfig:
    """Hyperparameters for :class:`PPOTrainer`."""

    num_envs: int
    rollout_len: int
    obs_dim: int
    action_dim: int
    embed_dim: int
    gru_hidden: int
    head_hidden: int
    action_log_std_init: float
    use_recurrence: bool
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


def _tanh_squashed_logprob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob of a tanh-squashed Gaussian evaluated at ``action``.

    Inverts ``action = tanh(u)`` via ``u = atanh(action)`` after clamping
    ``action`` into ``[-1 + eps, 1 - eps]`` to keep the inverse finite.
    Returns ``(logprob, pre_tanh_entropy)`` where ``pre_tanh_entropy`` is
    the summed entropy of the underlying Normal (we use this as a
    standard proxy for the full squashed-dist entropy).
    """
    action = action.clamp(-1.0 + _ATANH_EPS, 1.0 - _ATANH_EPS)
    # atanh(x) = 0.5 * (log1p(x) - log1p(-x))
    u = 0.5 * (torch.log1p(action) - torch.log1p(-action))
    std = log_std.exp()
    dist = Normal(mean, std)
    # Same tanh-correction formulation as models.sample_action.
    correction = 2.0 * (_LOG2 - u - F.softplus(-2.0 * u))
    logprob = dist.log_prob(u).sum(-1) - correction.sum(-1)
    entropy = dist.entropy().sum(-1)
    return logprob, entropy


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum ``values * mask`` and divide by ``mask.sum()`` (>=1 guard).

    Produces zero when the mask is entirely zero, which is the right
    behavior for empty padded batches.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


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

        # --- Initial global RNG seeding. Applied early so env/space
        # construction is deterministic.
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- Vectorized env. SyncVectorEnv calls the thunk once per env.
        # We seed via ``reset(seed=seed)`` which Gymnasium fans out as
        # ``[seed, seed+1, ..., seed+num_envs-1]`` per env.
        self.envs: SyncVectorEnv = SyncVectorEnv(
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
            use_recurrence=config.use_recurrence,
            embed_dim=config.embed_dim,
            gru_hidden=config.gru_hidden,
            head_hidden=config.head_hidden,
            action_log_std_init=config.action_log_std_init,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

        # --- Per-trainer action-sampling RNG state. We use the GLOBAL
        # torch RNG for model sampling (since ``Normal.rsample`` has no
        # generator kwarg in older torch versions), but save/restore a
        # trainer-local snapshot around every sampling call so two
        # trainers do not interfere with each other's global-RNG advance.
        # Initial state is the snapshot of the global RNG AFTER all init
        # seeding is done — so two trainers with the same seed start at
        # byte-identical RNG state.
        self._sampling_rng_state: torch.Tensor = torch.get_rng_state()

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
        """Roll out ``config.rollout_len`` ticks across the vector env.

        Writes per-tick ``(obs, action, logprob, reward, value, done,
        h_init)`` into a fresh :class:`RolloutBuffer`. Returns the
        buffer; the caller passes it back into :meth:`update`.
        """
        cfg = self.config
        buf = self._make_buffer()

        last_obs = self._last_obs
        h = self.h

        for tick in range(cfg.rollout_len):
            with torch.no_grad():
                # Sample action under a trainer-local RNG snapshot. We
                # swap in ``self._sampling_rng_state``, let the model do
                # its ``Normal.rsample``, then snapshot the advanced
                # state back. This isolates action sampling from any
                # other code advancing the global torch RNG.
                prev_rng = torch.get_rng_state()
                torch.set_rng_state(self._sampling_rng_state)
                try:
                    action, logprob, h_next = self.model.sample_action(last_obs, h)
                    self._sampling_rng_state = torch.get_rng_state()
                finally:
                    torch.set_rng_state(prev_rng)
                # A separate forward pass to pull the value estimate.
                # This is deterministic (no sampling), so we do not need
                # to swap RNG state here.
                _, _, value, _ = self.model.forward(last_obs, h)

            # h_init written to the buffer is the hidden state USED to
            # produce this tick's action/value — i.e. the pre-forward h.
            h_init_to_write = h

            # Step the vec env. SyncVectorEnv auto-resets any env that
            # terminates: ``next_obs`` for that env is the new reset obs,
            # and the final obs of the completed episode is stashed in
            # ``info``. We don't need the stashed final obs for PPO.
            action_np = action.detach().cpu().numpy()
            next_obs, reward, terminated, truncated, _info = self.envs.step(action_np)
            done_np = np.logical_or(terminated, truncated)

            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            done_t = torch.as_tensor(done_np, dtype=torch.float32)

            buf.add(
                tick=tick,
                obs=last_obs,
                action=action,
                logprob=logprob,
                reward=reward_t,
                value=value,
                done=done_t,
                h_init=h_init_to_write,
            )

            # Advance live hidden state BEFORE applying done-zeroing so
            # that envs that didn't reset carry their post-step h.
            h = h_next

            # Episode-reset bookkeeping: for each env that finished, zero
            # the live h row AND tell the buffer to write a zero h_init
            # on the next ``add`` for that env.
            if bool(done_np.any()):
                for e in range(cfg.num_envs):
                    if bool(done_np[e]):
                        h[e] = 0.0
                        buf.mark_reset(e)

            last_obs = torch.as_tensor(next_obs, dtype=torch.float32)

        # --- Bootstrap state for GAE at the end of the rollout. This is
        # the value estimate AT ``last_obs`` with the CURRENT (post-
        # advance, post-zero) live hidden state. Attach DIRECTLY to the
        # buffer so ``deepcopy(rollout)`` carries the state — ``update``
        # reads these off the rollout, not off the trainer, so a fresh
        # trainer that receives a cloned rollout can update identically.
        with torch.no_grad():
            _, _, last_value, _ = self.model.forward(last_obs, h)
        # ``buf.done[:, -1]`` is the done signal for the final tick; use
        # it as the bootstrap terminal flag. A True here means the env
        # just reset, so the bootstrap should be zeroed via GAE's
        # ``(1 - done)`` factor.
        buf.last_value = last_value
        buf.last_done = buf.done[:, -1].clone()

        # Persist post-rollout state for the next collect call.
        self._last_obs = last_obs
        self.h = h

        return buf

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, rollout: RolloutBuffer) -> dict:
        """Run ``num_epochs`` of PPO on ``rollout`` and return metrics.

        Computes GAE, then iterates minibatches yielded by
        :meth:`RolloutBuffer.iter_episode_minibatches`. The minibatch
        shuffle generator is seeded ONCE per ``update`` call, so the
        same permutation is visited on every epoch — a prerequisite for
        "same segment gets the same h_init on every epoch" (Test 3).
        """
        cfg = self.config

        # Compute GAE. We read the bootstrap (last_value, last_done) off
        # the rollout itself — populated by ``collect_rollout`` — so a
        # ``deepcopy(rollout)`` fed into a fresh trainer's ``update``
        # produces byte-identical results.
        last_value = getattr(rollout, "last_value", torch.zeros(cfg.num_envs))
        last_done = getattr(rollout, "last_done", torch.zeros(cfg.num_envs))
        rollout.compute_gae(last_values=last_value, last_dones=last_done)

        # Deterministic per-update minibatch generator. Re-seeded at the
        # start of each epoch so the same permutation is drawn every
        # time. Counter is incremented at the END of ``update`` so an
        # exception mid-update doesn't desynchronize repeated runs.
        mb_seed = self.seed * 1_000_003 + (self._update_counter + 1)

        # Metric accumulators. Weighted by ``valid_mask.sum()`` for a
        # true mean over valid samples.
        metrics_sum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "total_loss": 0.0,
        }
        total_valid = 0.0
        num_minibatches = 0

        for _epoch in range(cfg.num_epochs):
            gen = torch.Generator()
            gen.manual_seed(mb_seed)
            for batch in rollout.iter_episode_minibatches(
                minibatch_size=cfg.minibatch_size, generator=gen
            ):
                mb_stats, n_valid = self._ppo_minibatch_step(batch)
                if n_valid > 0:
                    for key, val in mb_stats.items():
                        metrics_sum[key] += val * n_valid
                    total_valid += n_valid
                num_minibatches += 1

        denom = max(total_valid, 1.0)
        metrics = {k: v / denom for k, v in metrics_sum.items()}
        metrics["num_minibatches"] = float(num_minibatches)
        metrics["total_valid"] = float(total_valid)
        # Commit the counter only on successful update.
        self._update_counter += 1
        return metrics

    def _ppo_minibatch_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, float], float]:
        """One PPO gradient step on a single padded minibatch.

        Returns per-valid-sample-averaged stats plus the valid-sample
        count so the caller can weight a multi-batch mean correctly.
        """
        cfg = self.config

        # --- Debug hook for Test 3 (h_init identical across epochs).
        # Append BEFORE the forward pass so the log reflects the input
        # the model actually saw, detached+cloned to survive any later
        # in-place surgery.
        if hasattr(self, "_training_h_init_log"):
            self._training_h_init_log.append(batch["h_init"].detach().clone())

        obs = batch["obs"]                 # (S, L, obs_dim)
        action = batch["action"]           # (S, L, action_dim)
        old_logprob = batch["old_logprob"] # (S, L)
        advantage = batch["advantage"]     # (S, L)
        return_ = batch["return_"]         # (S, L)
        old_value = batch["old_value"]     # (S, L)
        h_init = batch["h_init"]           # (S, H)
        valid_mask = batch["valid_mask"]   # (S, L)

        S, L = valid_mask.shape
        n_valid = float(valid_mask.sum().item())
        if n_valid <= 0.0:
            # Empty batch — skip but return zeros.
            return (
                {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0, "total_loss": 0.0},
                0.0,
            )

        # --- Segment-level BPTT. Unroll the forward pass across L,
        # carrying ``h`` within each segment. This matches how the
        # rollout produced the stored actions (modulo current policy
        # weights). In feedforward mode the model ignores h.
        h = h_init
        new_logprobs = []
        entropies = []
        values = []
        for t in range(L):
            mean_t, log_std_t, value_t, h = self.model.forward(obs[:, t], h)
            logp_t, ent_t = _tanh_squashed_logprob(
                mean_t, log_std_t, action[:, t]
            )
            new_logprobs.append(logp_t)
            entropies.append(ent_t)
            values.append(value_t)

        new_logprob = torch.stack(new_logprobs, dim=1)  # (S, L)
        entropy = torch.stack(entropies, dim=1)         # (S, L)
        value = torch.stack(values, dim=1)              # (S, L)

        # Advantage normalization over valid samples only. Standard PPO
        # trick; keeps the policy gradient scale stable.
        adv_mean = _masked_mean(advantage, valid_mask)
        adv_var = _masked_mean((advantage - adv_mean) ** 2, valid_mask)
        adv_std = adv_var.clamp(min=1e-8).sqrt()
        norm_adv = (advantage - adv_mean) / adv_std

        # --- Policy loss (clipped surrogate).
        ratio = (new_logprob - old_logprob).exp()
        pg1 = ratio * norm_adv
        pg2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * norm_adv
        policy_loss_per = -torch.min(pg1, pg2)
        policy_loss = _masked_mean(policy_loss_per, valid_mask)

        # --- Value loss (clipped).
        value_clipped = old_value + torch.clamp(
            value - old_value, -cfg.value_clip_ratio, cfg.value_clip_ratio
        )
        vl_unclipped = (value - return_) ** 2
        vl_clipped = (value_clipped - return_) ** 2
        value_loss_per = 0.5 * torch.max(vl_unclipped, vl_clipped)
        value_loss = _masked_mean(value_loss_per, valid_mask)

        # --- Entropy bonus (negated since we MINIMIZE loss).
        # NOTE: ``entropy`` is the pre-tanh Normal entropy, a standard
        # CleanRL-style proxy for the true squashed-distribution entropy.
        # Biased upward by a mean-dependent term; fine for Phase 2, may
        # matter if Phase 3 uses a larger ``entropy_coef``.
        entropy_mean = _masked_mean(entropy, valid_mask)
        entropy_loss = -entropy_mean

        total_loss = (
            policy_loss
            + cfg.value_coef * value_loss
            + cfg.entropy_coef * entropy_loss
        )

        # --- Backprop + grad clip + step.
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        # --- Diagnostics (masked).
        with torch.no_grad():
            approx_kl = _masked_mean(old_logprob - new_logprob, valid_mask)
            clip_frac_per = (
                (ratio - 1.0).abs() > cfg.clip_ratio
            ).float()
            clip_fraction = _masked_mean(clip_frac_per, valid_mask)

        return (
            {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_mean.item()),
                "approx_kl": float(approx_kl.item()),
                "clip_fraction": float(clip_fraction.item()),
                "total_loss": float(total_loss.item()),
            },
            n_valid,
        )


def evaluate_policy(
    model: ActorCritic,
    env_fn: Callable[[], gym.Env],
    num_episodes: int,
    seed: int,
) -> float:
    """Return mean episodic reward over ``num_episodes``."""
    model.eval()
    rewards: list[float] = []
    for i in range(int(num_episodes)):
        env = env_fn()
        obs, _ = env.reset(seed=seed + i)
        done = False
        ep_reward = 0.0
        h = model.init_hidden(batch_size=1)
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, h = model.sample_action(obs_t, h)
            obs, r, term, trunc, _ = env.step(action.squeeze(0).cpu().numpy())
            ep_reward += float(r)
            done = bool(term or trunc)
            if done:
                h.zero_()
        rewards.append(ep_reward)
        env.close()
    return float(np.mean(rewards)) if rewards else 0.0


def save_checkpoint(trainer: PPOTrainer, path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": trainer.model.state_dict(),
        "config": config,
    }
    torch.save(payload, path)


def _make_ppo_config(config: dict, *, use_recurrence: bool) -> PPOConfig:
    env_cfg = config.get("env", {})
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    return PPOConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=3,
        action_dim=2,
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
        action_log_std_init=float(model_cfg["action_log_std_init"]),
        use_recurrence=bool(use_recurrence),
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
    )


def _run_variant(config: dict, *, use_recurrence: bool, output_dir: Path) -> float:
    from envs.memory_toy import MemoryToyEnv

    env_cfg = config.get("env", {})
    run_cfg = config.get("run", {})
    ppo_cfg = _make_ppo_config(config, use_recurrence=use_recurrence)

    total_updates = int(run_cfg.get("total_updates", 10))
    eval_every = int(run_cfg.get("eval_every", 5))
    eval_episodes = int(run_cfg.get("eval_episodes", 20))
    checkpoint_every = int(run_cfg.get("checkpoint_every", 10))

    seed_base = int(env_cfg.get("seed_base", 0))
    variant_seed = seed_base + (0 if use_recurrence else 1_000_000)

    def env_fn() -> MemoryToyEnv:
        return MemoryToyEnv(
            episode_length=int(env_cfg.get("episode_length", 64)),
            cue_visible_ticks=int(env_cfg.get("cue_visible_ticks", 4)),
        )

    trainer = PPOTrainer(env_fn=env_fn, config=ppo_cfg, seed=variant_seed)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir))
    except Exception:
        writer = None

    last_eval = float("nan")
    for update_idx in range(1, total_updates + 1):
        rollout = trainer.collect_rollout()
        metrics = trainer.update(rollout)

        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, update_idx)

        if update_idx % eval_every == 0 or update_idx == total_updates:
            last_eval = evaluate_policy(
                trainer.model,
                env_fn,
                num_episodes=eval_episodes,
                seed=variant_seed + 100_000 + update_idx,
            )
            if writer is not None:
                writer.add_scalar("eval/mean_reward", last_eval, update_idx)

        if update_idx % checkpoint_every == 0 or update_idx == total_updates:
            ckpt_path = output_dir / f"ckpt_{update_idx:04d}.pt"
            save_checkpoint(
                trainer,
                ckpt_path,
                {
                    "env": env_cfg,
                    "model": {**config.get("model", {}), "use_recurrence": use_recurrence},
                    "ppo": config.get("ppo", {}),
                    "run": config.get("run", {}),
                },
            )

    final_ckpt = output_dir / "ckpt_final.pt"
    save_checkpoint(
        trainer,
        final_ckpt,
        {
            "env": env_cfg,
            "model": {**config.get("model", {}), "use_recurrence": use_recurrence},
            "ppo": config.get("ppo", {}),
            "run": config.get("run", {}),
        },
    )
    if writer is not None:
        writer.close()

    return last_eval


def train_from_config(config: dict) -> dict[str, float]:
    """Train recurrent and feedforward variants and return final eval means."""
    out_root = Path(str(config.get("run", {}).get("output_dir", "runs/phase2_memory_toy")))
    recurrent_dir = out_root / "recurrent"
    feedforward_dir = out_root / "feedforward"

    recurrent = _run_variant(config, use_recurrence=True, output_dir=recurrent_dir)
    feedforward = _run_variant(config, use_recurrence=False, output_dir=feedforward_dir)
    return {"recurrent": recurrent, "feedforward": feedforward}
