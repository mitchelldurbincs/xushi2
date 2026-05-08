from __future__ import annotations

import pytest
import torch

from train.entity_attention import EntityAttentionEncoder


def test_entity_attention_shapes_and_masked_weights() -> None:
    encoder = EntityAttentionEncoder(
        entity_dim=5,
        embed_dim=16,
        num_heads=4,
        output_dim=12,
    )
    tokens = torch.randn(3, 4, 5)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
            [True, True, True, True],
        ]
    )
    features, weights = encoder(tokens, mask)

    assert features.shape == (3, 12)
    assert weights.shape == (3, 4)
    assert torch.isfinite(features).all()
    assert torch.isfinite(weights).all()
    assert torch.all(weights[~mask] == 0.0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_entity_attention_empty_rows_return_zero_feature() -> None:
    encoder = EntityAttentionEncoder(entity_dim=3, embed_dim=8, num_heads=2)
    tokens = torch.randn(2, 5, 3)
    mask = torch.tensor(
        [
            [False, False, False, False, False],
            [True, False, False, False, False],
        ]
    )
    features, weights = encoder(tokens, mask)

    torch.testing.assert_close(features[0], torch.zeros_like(features[0]))
    torch.testing.assert_close(weights[0], torch.zeros_like(weights[0]))
    assert torch.isfinite(features[1]).all()


def test_entity_attention_is_permutation_invariant_with_matching_mask() -> None:
    torch.manual_seed(0)
    encoder = EntityAttentionEncoder(entity_dim=4, embed_dim=12, num_heads=3)
    tokens = torch.randn(1, 5, 4)
    mask = torch.tensor([[True, True, False, True, False]])
    perm = torch.tensor([3, 0, 4, 1, 2])

    features_a, _ = encoder(tokens, mask)
    features_b, _ = encoder(tokens[:, perm], mask[:, perm])

    torch.testing.assert_close(features_a, features_b, atol=1e-6, rtol=1e-6)


def test_entity_attention_rejects_bad_shapes() -> None:
    encoder = EntityAttentionEncoder(entity_dim=4, embed_dim=8, num_heads=2)
    with pytest.raises(ValueError, match="tokens must have shape"):
        encoder(torch.zeros(2, 4), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="batch/entity axes"):
        encoder(torch.zeros(2, 3, 4), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="last dim"):
        encoder(torch.zeros(2, 3, 5), torch.ones(2, 3, dtype=torch.bool))
