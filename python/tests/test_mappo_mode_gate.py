from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.mappo_model import MappoActorCritic, mode_aux_loss_and_accuracy
from train.mappo_rollout_trainer import make_mappo_config
from train.train import load_config
from xushi2.obs_manifest import actor_field_slice

_ENEMY_ALIVE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS = actor_field_slice("enemy_relative_position")


def _config(output_dir: Path, *, mode_gated_combat: bool = True) -> dict:
    return {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "opponent_bot": "noop",
            "learner_team": "A",
            "sim": {
                "round_length_seconds": 3,
                "fog_of_war_enabled": False,
                "randomize_map": False,
                "seed": 0,
                "action_repeat": 3,
            },
        },
        "model": {
            "use_recurrence": True,
            "embed_dim": 16,
            "gru_hidden": 8,
            "head_hidden": 16,
            "action_log_std_init": -1.0,
        },
        "ppo": {
            "num_envs": 1,
            "rollout_len": 4,
            "num_epochs": 1,
            "minibatch_size": 1,
            "learning_rate": 3.0e-4,
            "value_normalization": True,
            "vector_env": "sync",
            "torch_num_threads": 1,
            "clip_ratio": 0.2,
            "value_clip_ratio": 0.2,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "mode_gated_combat": mode_gated_combat,
            "mode_aux_coef": 0.3,
        },
        "run": {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": str(output_dir),
        },
    }


def test_model_forward_produces_mode_logits_shape(tmp_path: Path) -> None:
    cfg = make_mappo_config(_config(tmp_path))
    model = MappoActorCritic(cfg)
    obs = torch.zeros(3, cfg.obs_dim)
    h = model.init_hidden(3)

    features, _h = model.actor_head_features(obs, h)
    mode_logits = model.mode_logits_from_features(features)

    assert mode_logits is not None
    assert mode_logits.shape == (3, 2)


def test_fire_gate_reduces_fire_probability(tmp_path: Path) -> None:
    cfg = make_mappo_config(_config(tmp_path))
    model = MappoActorCritic(cfg)
    raw_logits = torch.tensor([[0.0], [2.0], [-2.0]])
    mode_logits = torch.tensor([[4.0, -4.0], [0.0, 0.0], [-4.0, 4.0]])

    gated_logits = model.gated_binary_logits(raw_logits, mode_logits)
    p_fire_raw = torch.sigmoid(raw_logits[:, 0])
    p_fire_actual = torch.sigmoid(gated_logits[:, 0])

    assert torch.all(p_fire_actual <= p_fire_raw + 1.0e-6)


def test_mode_loss_finite_and_positive_when_logits_mismatch_target(
    tmp_path: Path,
) -> None:
    cfg = make_mappo_config(_config(tmp_path))
    obs = torch.zeros(2, cfg.obs_dim)
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([[0.2, 0.0], [0.1, 0.0]])
    logits = torch.tensor([[4.0, -4.0], [4.0, -4.0]])
    labels = torch.ones(2, dtype=torch.long)

    loss, acc, count = mode_aux_loss_and_accuracy(logits, obs, cfg, labels=labels)

    assert count.item() == 2
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert acc.item() == pytest.approx(0.0)


def test_config_loading_sets_mode_gate_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        """
phase: 4
env:
  seed_base: 0
  opponent_bot: noop
  learner_team: A
  sim:
    round_length_seconds: 3
    fog_of_war_enabled: false
    randomize_map: false
    seed: 0
    action_repeat: 3
model:
  use_recurrence: true
  embed_dim: 16
  gru_hidden: 8
  head_hidden: 16
  action_log_std_init: -1.0
ppo:
  num_envs: 1
  rollout_len: 4
  num_epochs: 1
  minibatch_size: 1
  learning_rate: 3.0e-4
  value_normalization: true
  vector_env: sync
  torch_num_threads: 1
  clip_ratio: 0.2
  value_clip_ratio: 0.2
  gamma: 0.997
  gae_lambda: 0.95
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
run:
  total_updates: 1
  eval_every: 1
  eval_episodes: 1
  checkpoint_every: 1
  log_every: 1
  output_dir: runs/test
""",
        encoding="utf-8",
    )

    loaded = load_config(path)
    cfg = make_mappo_config(loaded)

    assert loaded["ppo"]["mode_gated_combat"] is False
    assert loaded["ppo"]["mode_aux_coef"] == pytest.approx(0.3)
    assert cfg.mode_gated_combat is False
    assert cfg.mode_aux_coef == pytest.approx(0.3)
