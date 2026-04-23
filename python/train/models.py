"""Actor-critic model for recurrent PPO tasks.

Supports two variants controlled by ``use_recurrence``:
  * Recurrent: ``embed -> GRUCell -> actor/critic heads`` (carries hidden state).
  * Feedforward: ``embed -> Linear+ReLU -> actor/critic heads`` (ignores ``h``).

Action handling is split into two groups:
  * continuous controls: tanh-squashed Gaussian in ``[-1, 1]``
  * binary controls: Bernoulli logits sampled into ``{0, 1}``

Phase 2 uses only the continuous path. Phase 3 uses both.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG2 = 0.6931471805599453


class PolicyOutput(tuple):
    __slots__ = ()

    @property
    def continuous_mean(self) -> torch.Tensor:
        return self[0]

    @property
    def continuous_log_std(self) -> torch.Tensor:
        return self[1]

    @property
    def binary_logits(self) -> torch.Tensor:
        return self[2]

    @property
    def value(self) -> torch.Tensor:
        return self[3]

    @property
    def h_next(self) -> torch.Tensor:
        return self[4]


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        continuous_action_dim: int,
        binary_action_dim: int,
        use_recurrence: bool,
        embed_dim: int,
        gru_hidden: int,
        head_hidden: int,
        action_log_std_init: float,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous_action_dim = continuous_action_dim
        self.binary_action_dim = binary_action_dim
        self.use_recurrence = use_recurrence
        self.gru_hidden = gru_hidden

        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
        )

        # Trunk: either GRUCell (recurrent) or a matching-shape Linear+ReLU
        # (feedforward). Note: GRUCell has ~6x the parameter count of the
        # feedforward trunk at equal hidden size. This is fine for the
        # Phase-2 memory-proof comparison because the feedforward failure
        # is structural (obs is (0,0,0) at the terminal tick, so no
        # policy can recover the target), not capacity-limited.
        if use_recurrence:
            self.gru = nn.GRUCell(embed_dim, gru_hidden)
            self.ff_trunk = None
        else:
            self.gru = None
            self.ff_trunk = nn.Sequential(
                nn.Linear(embed_dim, gru_hidden),
                nn.ReLU(),
            )

        # Actor heads.
        self.actor_body = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
        )
        self.actor_mean_head = nn.Linear(head_hidden, continuous_action_dim)
        self.actor_binary_head = (
            nn.Linear(head_hidden, binary_action_dim)
            if binary_action_dim > 0
            else None
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # State-independent learned log-std for continuous controls only.
        self.log_std = nn.Parameter(
            torch.ones(continuous_action_dim) * action_log_std_init
        )

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(batch_size, self.gru_hidden, device=p.device, dtype=p.dtype)

    def policy_outputs(self, obs: torch.Tensor, h: torch.Tensor) -> PolicyOutput:
        emb = self.embed(obs)

        if self.use_recurrence:
            h_next = self.gru(emb, h)
            trunk = h_next
        else:
            # Feedforward branch must NOT depend on h.
            trunk = self.ff_trunk(emb)
            h_next = torch.zeros_like(h)

        actor_features = self.actor_body(trunk)
        action_mean = self.actor_mean_head(actor_features)
        if self.actor_binary_head is None:
            binary_logits = trunk.new_zeros((trunk.shape[0], 0))
        else:
            binary_logits = self.actor_binary_head(actor_features)
        value = self.critic_head(trunk).squeeze(-1)
        return PolicyOutput((action_mean, self.log_std, binary_logits, value, h_next))

    def forward(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.policy_outputs(obs, h)
        return (
            outputs.continuous_mean,
            outputs.continuous_log_std,
            outputs.value,
            outputs.h_next,
        )

    def sample_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.policy_outputs(obs, h)

        pieces: list[torch.Tensor] = []
        logprob = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

        if self.continuous_action_dim > 0:
            std = outputs.continuous_log_std.exp()
            dist = torch.distributions.Normal(outputs.continuous_mean, std)
            u = dist.rsample()
            cont_action = torch.tanh(u)
            correction = 2.0 * (_LOG2 - u - F.softplus(-2.0 * u))
            logprob = logprob + dist.log_prob(u).sum(-1) - correction.sum(-1)
            pieces.append(cont_action)

        if self.binary_action_dim > 0:
            binary_dist = torch.distributions.Bernoulli(logits=outputs.binary_logits)
            binary_action = binary_dist.sample()
            logprob = logprob + binary_dist.log_prob(binary_action).sum(-1)
            pieces.append(binary_action)

        action = torch.cat(pieces, dim=-1) if pieces else obs.new_zeros((obs.shape[0], 0))
        return action, logprob, outputs.h_next

    def greedy_action(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.policy_outputs(obs, h)
        pieces: list[torch.Tensor] = []
        if self.continuous_action_dim > 0:
            pieces.append(torch.tanh(outputs.continuous_mean))
        if self.binary_action_dim > 0:
            pieces.append((outputs.binary_logits >= 0.0).to(dtype=obs.dtype))
        action = torch.cat(pieces, dim=-1) if pieces else obs.new_zeros((obs.shape[0], 0))
        return action, outputs.h_next


def build_model(
    obs_dim: int,
    action_dim: int,
    use_recurrence: bool,
    embed_dim: int,
    gru_hidden: int,
    head_hidden: int,
    action_log_std_init: float,
    continuous_action_dim: int | None = None,
    binary_action_dim: int = 0,
) -> ActorCritic:
    if continuous_action_dim is None:
        continuous_action_dim = action_dim
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous_action_dim=continuous_action_dim,
        binary_action_dim=binary_action_dim,
        use_recurrence=use_recurrence,
        embed_dim=embed_dim,
        gru_hidden=gru_hidden,
        head_hidden=head_hidden,
        action_log_std_init=action_log_std_init,
    )
