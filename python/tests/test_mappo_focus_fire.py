from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.mappo_model import (
    MappoActorCritic,
    target_selection_aux_loss_and_accuracy,
    target_selection_aux_metrics,
    target_selection_aux_targets,
)
from train.mappo_rollout_trainer import make_mappo_config
from xushi2.obs_manifest import actor_field_slice

_OWN_POSITION = actor_field_slice("own_position")
_ENEMY_ALIVE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS = actor_field_slice("enemy_relative_position")
_ENEMY_HP = actor_field_slice("enemy_hp")


def _cfg(output_dir: Path) -> dict:
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
            "target_selection_dim": 4,
            "target_conditioned_combat": True,
            "target_selection_aux_coef": 0.5,
            "target_selection_aux_mode": "team_focus_low_hp",
            "target_selection_objective_proximity_coef": 0.1,
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


def _obs(cfg) -> torch.Tensor:
    return torch.zeros(3, cfg.obs_dim)


def test_team_focus_low_hp_labels_shared_visible_low_hp_enemy(tmp_path: Path) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.0, 0.0], [0.4, 0.0], [-0.4, 0.0]])
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([[1.0, 0.0], [1.1, 0.0], [1.2, 0.0]])
    obs[:, _ENEMY_HP] = torch.tensor([[0.8], [0.2], [0.5]])

    labels, valid = target_selection_aux_targets(obs, cfg)

    assert valid.tolist() == [True, True, True]
    assert labels.tolist() == [1, 1, 1]


def test_same_target_fraction_high_when_agents_see_same_enemy(tmp_path: Path) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]])
    enemy_pos = torch.tensor([0.7, 0.0])
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = enemy_pos - obs[:, _OWN_POSITION]
    obs[:, _ENEMY_HP] = 0.25

    metrics = target_selection_aux_metrics(obs, cfg)

    assert metrics["target_selection_same_target_fraction"].item() == pytest.approx(1.0)
    assert metrics["target_selection_label_entropy"].item() == pytest.approx(0.0)


def test_focus_fire_fallback_rate_low_for_shared_enemy(tmp_path: Path) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]])
    enemy_pos = torch.tensor([0.7, 0.0])
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = enemy_pos - obs[:, _OWN_POSITION]
    obs[:, _ENEMY_HP] = 0.25

    metrics = target_selection_aux_metrics(obs, cfg)

    assert metrics["target_selection_fallback_rate"].item() == pytest.approx(0.0)


def test_agents_without_visible_enemy_get_no_target_label(tmp_path: Path) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    obs = _obs(cfg)
    obs[0, _ENEMY_ALIVE] = 1.0
    obs[0, _ENEMY_REL_POS] = torch.tensor([1.0, 0.0])
    obs[0, _ENEMY_HP] = 0.2

    labels, valid = target_selection_aux_targets(obs, cfg)

    assert valid.tolist() == [True, True, True]
    assert labels.tolist() == [0, 3, 3]


def test_focus_fire_model_forward_target_logits_shape(tmp_path: Path) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    model = MappoActorCritic(cfg)
    obs = _obs(cfg)
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([1.0, 0.0])
    h = model.init_hidden(3)

    features, _h = model.actor_head_features(obs, h)
    _mean, _logits, target_logits = model.policy_heads_from_features(obs, features)

    assert target_logits is not None
    assert target_logits.shape == (3, 4)


def test_focus_fire_aux_loss_finite_and_nonzero_when_labels_mismatch(
    tmp_path: Path,
) -> None:
    cfg = make_mappo_config(_cfg(tmp_path))
    obs = _obs(cfg)
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    obs[:, _ENEMY_HP] = torch.tensor([[0.9], [0.1], [0.8]])
    logits = torch.full((3, 4), -4.0)
    logits[:, 0] = 4.0

    loss, acc, count = target_selection_aux_loss_and_accuracy(logits, obs, cfg)

    assert count.item() == 3
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert acc.item() == pytest.approx(0.0)
