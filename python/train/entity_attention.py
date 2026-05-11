"""Entity-token attention encoder for Phase 5+ models."""

from __future__ import annotations

import torch
import torch.nn as nn


class EntityAttentionEncoder(nn.Module):
    """Pool variable-count entity tokens into one fixed-width feature vector.

    ``valid_mask`` uses True for real tokens and False for padding. Rows with no
    valid tokens return an all-zero feature instead of propagating NaNs from
    attention over an empty set.
    """

    def __init__(
        self,
        *,
        entity_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if entity_dim <= 0:
            raise ValueError("entity_dim must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.entity_dim = int(entity_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.output_dim = int(output_dim or embed_dim)

        self.token_embed = nn.Sequential(
            nn.Linear(entity_dim, embed_dim),
            nn.ReLU(),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.output_dim),
            nn.ReLU(),
        )

    def forward(
        self, tokens: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, entities, entity_dim)")
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape (batch, entities)")
        if tokens.shape[:2] != valid_mask.shape:
            raise ValueError("tokens and valid_mask batch/entity axes must match")
        if tokens.shape[2] != self.entity_dim:
            raise ValueError(f"tokens last dim must be {self.entity_dim}, got {tokens.shape[2]}")

        valid = valid_mask.to(dtype=torch.bool, device=tokens.device)
        empty = ~valid.any(dim=1)
        safe_valid = valid.clone()
        if bool(empty.any()):
            safe_valid[empty, 0] = True

        embedded = self.token_embed(tokens)
        if bool(empty.any()):
            embedded = embedded.clone()
            embedded[empty, 0] = 0.0

        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, weights = self.attn(
            query,
            embedded,
            embedded,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=True,
        )
        features = self.output(pooled.squeeze(1))
        weights = weights.squeeze(1).masked_fill(~valid, 0.0)
        if bool(empty.any()):
            features = features.masked_fill(empty[:, None], 0.0)
        return features, weights
