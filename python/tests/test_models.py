import torch
from train.models import build_model


def test_recurrent_forward_shapes():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.zeros(8, 3)
    h = model.init_hidden(batch_size=8)
    action_mean, log_std, value, h_next = model.forward(obs, h)
    assert action_mean.shape == (8, 2)
    assert log_std.shape == (2,)
    assert value.shape == (8,)
    assert h_next.shape == h.shape == (8, 64)


def test_feedforward_bypasses_gru():
    """With use_recurrence=False, forward output must not depend on h."""
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=False,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.randn(4, 3)
    h1 = torch.zeros(4, 64)
    h2 = torch.randn(4, 64)
    mean1, _, v1, _ = model.forward(obs, h1)
    mean2, _, v2, _ = model.forward(obs, h2)
    torch.testing.assert_close(mean1, mean2)
    torch.testing.assert_close(v1, v2)


def test_recurrent_uses_hidden_state():
    """With use_recurrence=True, forward output MUST depend on h."""
    torch.manual_seed(0)
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    obs = torch.randn(4, 3)
    h1 = torch.zeros(4, 64)
    h2 = torch.ones(4, 64)
    mean1, _, _, _ = model.forward(obs, h1)
    mean2, _, _, _ = model.forward(obs, h2)
    assert not torch.allclose(mean1, mean2)


def test_init_hidden_is_zeros():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=-1.0)
    h = model.init_hidden(batch_size=3)
    assert torch.all(h == 0)


def test_action_sampling_tanh_squash_bounds():
    model = build_model(obs_dim=3, action_dim=2, use_recurrence=True,
                        embed_dim=64, gru_hidden=64, head_hidden=64,
                        action_log_std_init=2.0)  # high std → actions could blow up
    obs = torch.zeros(1000, 3)
    h = model.init_hidden(1000)
    action, logp, _ = model.sample_action(obs, h)
    assert action.shape == (1000, 2)
    assert (action.abs() <= 1.0).all(), "tanh squash must bound actions in [-1, 1]"
    assert torch.isfinite(logp).all(), "logprob must be finite even with saturated actions"
