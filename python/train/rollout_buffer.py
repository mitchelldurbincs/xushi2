"""Rollout buffer + GAE for recurrent PPO.

Stores per-tick ``(obs, action, logprob, reward, value, done)`` plus
``h_init`` — the hidden state the forward pass used AT THIS TICK'S
SEGMENT START. A "segment" is a maximal contiguous run of ticks within
the rollout where ``h_init`` is constant: it starts at rollout tick 0
(or right after a ``done=True`` tick) and ends at the next ``done=True``
tick (or the rollout's last tick for a trailing partial segment).

Why store ``h_init`` redundantly for every tick in a segment? Because at
PPO update time we re-run the model forward pass from the segment's
initial hidden state (not from mid-segment). Storing ``h_init`` per tick
(shape ``(N, L, H)``) is the simplest way to make minibatch sampling
correct and fast. Memory cost is small: 16 envs * 256 ticks * 64 hidden
= ~1 MB.

Public API (consumed by ``train.ppo_recurrent`` — Task 6):

- ``__init__(num_envs, rollout_len, obs_dim, action_dim, gru_hidden, gamma, gae_lambda)``
- ``add(tick, obs, action, logprob, reward, value, done, h_init)`` —
  batched over envs at a single tick. Shapes:
      obs     (num_envs, obs_dim)
      action  (num_envs, action_dim)
      logprob (num_envs,)
      reward  (num_envs,)
      value   (num_envs,)
      done    (num_envs,)  float {0.0, 1.0}
      h_init  (num_envs, gru_hidden)
- ``mark_reset(env_idx)`` — after ``done=True`` at tick ``t`` for env
  ``env_idx``, the NEXT ``add`` for that env writes zero ``h_init``
  (overriding whatever the caller passes). The trainer's contract is:
  call ``mark_reset`` immediately after any ``add`` that contained a
  done, then on the next ``add`` the buffer zeroes the relevant rows.
- ``compute_gae(last_values, last_dones) -> (advantages, returns)`` —
  standard GAE with the ``(1 - done_t)`` boundary factor zeroing the
  bootstrap across episode boundaries. Also caches advantages/returns
  internally so ``iter_episode_minibatches`` can yield them.
- ``iter_episode_minibatches(minibatch_size, generator=None)`` — yields
  minibatches composed of whole segments (possibly including a trailing
  partial segment at rollout end). Each batch has shape ``(S, L_max,
  ...)`` with right-padding and a ``valid_mask``.

See ``docs/memory_toy.md`` §"Hidden-state management rules".
"""

from __future__ import annotations

from typing import Iterator

import torch


class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        rollout_len: int,
        obs_dim: int,
        action_dim: int,
        gru_hidden: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_envs = int(num_envs)
        self.rollout_len = int(rollout_len)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.gru_hidden = int(gru_hidden)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.device = torch.device(device)

        N, L, O, A, H = num_envs, rollout_len, obs_dim, action_dim, gru_hidden
        self.obs = torch.zeros(N, L, O, device=self.device)
        self.action = torch.zeros(N, L, A, device=self.device)
        self.logprob = torch.zeros(N, L, device=self.device)
        self.reward = torch.zeros(N, L, device=self.device)
        self.value = torch.zeros(N, L, device=self.device)
        self.done = torch.zeros(N, L, device=self.device)
        self.h_init = torch.zeros(N, L, H, device=self.device)

        # Filled by ``compute_gae``; consumed by ``iter_episode_minibatches``.
        self.advantages = torch.zeros(N, L, device=self.device)
        self.returns = torch.zeros(N, L, device=self.device)

        # ``reset_pending[e]`` = True means the next ``add`` on env ``e``
        # must write a zero ``h_init`` row regardless of what the caller
        # passes. Cleared once consumed.
        self._reset_pending = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Gate: ``iter_episode_minibatches`` refuses to run until
        # ``compute_gae`` has been called at least once, so the trainer
        # can't silently iterate over zero advantages from ``__init__``.
        self._gae_computed = False

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def add(
        self,
        tick: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        h_init: torch.Tensor,
    ) -> None:
        """Write one timestep for all envs at tick ``tick``."""
        if not (0 <= tick < self.rollout_len):
            raise IndexError(f"tick {tick} out of range [0, {self.rollout_len})")

        self.obs[:, tick] = obs
        self.action[:, tick] = action
        self.logprob[:, tick] = logprob
        self.reward[:, tick] = reward
        self.value[:, tick] = value
        self.done[:, tick] = done

        # Apply pending resets: rows flagged get zero h_init; others get
        # whatever the caller passed.
        if self._reset_pending.any():
            # Broadcast-safe zeroing via mask.
            mask = self._reset_pending.view(self.num_envs, 1).float()
            h_written = h_init * (1.0 - mask)  # zero where reset_pending
            self.h_init[:, tick] = h_written
            self._reset_pending.zero_()
        else:
            self.h_init[:, tick] = h_init

    def mark_reset(self, env_idx: int) -> None:
        """Signal that env ``env_idx`` was reset between the last add and
        the next add. The next ``add`` for that env will write zero
        ``h_init`` regardless of what the caller hands in."""
        if not (0 <= env_idx < self.num_envs):
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {self.num_envs})"
            )
        self._reset_pending[env_idx] = True

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------
    def compute_gae(
        self, last_values: torch.Tensor, last_dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages + returns for the whole buffer.

        Backward pass through time:
            delta_t = r_t + gamma * (1 - done_t) * V_{t+1} - V_t
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

        ``V_{t+1}`` at the last tick is ``last_values`` (bootstrap). The
        ``(1 - done_t)`` factor zeros both the bootstrap at step ``t``
        and the recursion from ``A_{t+1}`` across episode boundaries.
        """
        N, L = self.num_envs, self.rollout_len
        advantages = torch.zeros_like(self.value)
        last_gae = torch.zeros(N, device=self.device)

        for t in reversed(range(L)):
            if t == L - 1:
                next_value = last_values
                next_nonterminal = 1.0 - last_dones
            else:
                next_value = self.value[:, t + 1]
                # The boundary factor uses ``done_t``, NOT ``done_{t+1}``:
                # we gate on whether the episode ends AT step t, which
                # kills both the bootstrap into V_{t+1} and the carry of
                # A_{t+1} back into A_t.
                next_nonterminal = 1.0 - self.done[:, t]
            delta = (
                self.reward[:, t]
                + self.gamma * next_value * next_nonterminal
                - self.value[:, t]
            )
            last_gae = (
                delta
                + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            )
            advantages[:, t] = last_gae

        returns = advantages + self.value
        self.advantages = advantages
        self.returns = returns
        self._gae_computed = True
        return advantages, returns

    # ------------------------------------------------------------------
    # Minibatch iteration
    # ------------------------------------------------------------------
    def _segment_boundaries(self) -> list[tuple[int, int, int]]:
        """Return a list of ``(env_idx, start, end)`` tuples (inclusive
        end) for every segment in the rollout.

        A segment starts at tick 0 OR right after a ``done=True``; it
        ends at a ``done=True`` OR at ``rollout_len - 1`` for a trailing
        partial segment.
        """
        segments: list[tuple[int, int, int]] = []
        for e in range(self.num_envs):
            start = 0
            for t in range(self.rollout_len):
                if self.done[e, t].item() > 0.5:
                    segments.append((e, start, t))
                    start = t + 1
            if start < self.rollout_len:
                segments.append((e, start, self.rollout_len - 1))
        return segments

    def iter_episode_minibatches(
        self,
        minibatch_size: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield minibatches of whole segments.

        Each batch has shape ``(S, L_max, ...)`` where ``S ==
        minibatch_size`` (or fewer for the last trailing batch) and
        ``L_max`` is the longest segment in that batch. Shorter segments
        are right-padded with zeros; ``valid_mask`` marks the valid
        positions.

        Batch dict keys: ``obs`` (S, L, obs_dim), ``action`` (S, L,
        action_dim), ``old_logprob`` (S, L), ``advantage`` (S, L),
        ``return_`` (S, L), ``old_value`` (S, L), ``h_init`` (S,
        gru_hidden), ``valid_mask`` (S, L) [float].
        """
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be > 0, got {minibatch_size}")
        if not self._gae_computed:
            raise RuntimeError(
                "iter_episode_minibatches called before compute_gae. "
                "advantages/returns would be all zeros from __init__."
            )

        segments = self._segment_boundaries()
        if not segments:
            return

        # Shuffle segments for stochastic minibatching (standard PPO).
        num_segs = len(segments)
        if generator is not None:
            perm = torch.randperm(num_segs, generator=generator).tolist()
        else:
            perm = torch.randperm(num_segs).tolist()
        segments = [segments[i] for i in perm]

        for batch_start in range(0, num_segs, minibatch_size):
            batch = segments[batch_start : batch_start + minibatch_size]
            S = len(batch)
            lengths = [end - start + 1 for (_, start, end) in batch]
            L_max = max(lengths)

            obs = torch.zeros(S, L_max, self.obs_dim, device=self.device)
            action = torch.zeros(S, L_max, self.action_dim, device=self.device)
            old_logprob = torch.zeros(S, L_max, device=self.device)
            advantage = torch.zeros(S, L_max, device=self.device)
            return_ = torch.zeros(S, L_max, device=self.device)
            old_value = torch.zeros(S, L_max, device=self.device)
            h_init = torch.zeros(S, self.gru_hidden, device=self.device)
            valid_mask = torch.zeros(S, L_max, device=self.device)

            for i, (env_idx, start, end) in enumerate(batch):
                length = end - start + 1
                obs[i, :length] = self.obs[env_idx, start : end + 1]
                action[i, :length] = self.action[env_idx, start : end + 1]
                old_logprob[i, :length] = self.logprob[env_idx, start : end + 1]
                advantage[i, :length] = self.advantages[env_idx, start : end + 1]
                return_[i, :length] = self.returns[env_idx, start : end + 1]
                old_value[i, :length] = self.value[env_idx, start : end + 1]
                # All ticks in the segment share the same h_init; grab
                # from the segment's start tick.
                h_init[i] = self.h_init[env_idx, start]
                valid_mask[i, :length] = 1.0

            yield {
                "obs": obs,
                "action": action,
                "old_logprob": old_logprob,
                "advantage": advantage,
                "return_": return_,
                "old_value": old_value,
                "h_init": h_init,
                "valid_mask": valid_mask,
            }
