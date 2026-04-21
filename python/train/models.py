"""Actor-critic model for the memory-toy task.

Supports two variants controlled by ``use_recurrence``:
  * Recurrent: ``embed -> GRUCell -> actor/critic heads`` (carries hidden state).
  * Feedforward: ``embed -> Linear+ReLU -> actor/critic heads`` (ignores ``h``).

The ``log_std`` is a state-independent learned parameter of shape
``(action_dim,)``. ``sample_action`` applies a tanh squash with the standard
change-of-variables log-prob correction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        use_recurrence: bool,
        embed_dim: int,
        gru_hidden: int,
        head_hidden: int,
        action_log_std_init: float,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, action_dim),
        )

        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(gru_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # State-independent learned log-std.
        self.log_std = nn.Parameter(torch.ones(action_dim) * action_log_std_init)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        p = next(self.parameters())
        return torch.zeros(batch_size, self.gru_hidden, device=p.device, dtype=p.dtype)

    def forward(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embed(obs)

        if self.use_recurrence:
            h_next = self.gru(emb, h)
            trunk = h_next
        else:
            # Feedforward branch must NOT depend on h.
            trunk = self.ff_trunk(emb)
            h_next = torch.zeros_like(h)

        action_mean = self.actor_head(trunk)
        value = self.critic_head(trunk).squeeze(-1)
        return action_mean, self.log_std, value, h_next

    def sample_action(
        self, obs: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, log_std, _, h_next = self.forward(obs, h)
        std = log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        u = dist.rsample()
        action = torch.tanh(u)
        # Numerically stable tanh-squash log-prob correction (SAC-style):
        #   d/du tanh(u) = 1 - tanh(u)^2 = sech(u)^2
        #   log(sech(u)^2) = 2*log(2) - 2u - 2*softplus(-2u)
        # This avoids the underflow of log(1 - tanh(u)^2 + eps) near |u| large.
        correction = 2.0 * (
            torch.log(torch.tensor(2.0)) - u - torch.nn.functional.softplus(-2.0 * u)
        )
        logprob = dist.log_prob(u).sum(-1) - correction.sum(-1)
        return action, logprob, h_next


def build_model(
    obs_dim: int,
    action_dim: int,
    use_recurrence: bool,
    embed_dim: int,
    gru_hidden: int,
    head_hidden: int,
    action_log_std_init: float,
) -> ActorCritic:
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_recurrence=use_recurrence,
        embed_dim=embed_dim,
        gru_hidden=gru_hidden,
        head_hidden=head_hidden,
        action_log_std_init=action_log_std_init,
    )
