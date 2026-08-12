"""Tests for the PPO warm-start stabilizers: critic-only warmup updates and
the annealed KL(pi || pi_anchor) penalty toward the frozen PPO-start policy."""

from __future__ import annotations

import copy

import pytest
import torch

from envs.phase4_mappo import Phase4MappoEnv
from train.mappo import compute_anchor_kl_coef
from train.mappo_rollout_trainer import (
    MappoTrainer,
    _anchor_action_kl,
    make_mappo_config,
)


def _tiny_cfg(**ppo_overrides) -> dict:
    ppo = {
        "num_envs": 2,
        "rollout_len": 8,
        "num_epochs": 1,
        "learning_rate": 3.0e-4,
        "value_normalization": True,
        "vector_env": "sync",
        "torch_num_threads": 1,
        "lr_schedule": "constant",
        "lr_final_ratio": 1.0,
        "warmup_updates": 0,
        "clip_ratio": 0.2,
        "value_clip_ratio": 0.2,
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    ppo.update(ppo_overrides)
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
                "mechanics": {
                    "revolver_damage_centi_hp": 7500,
                    "revolver_fire_cooldown_ticks": 15,
                    "revolver_hitbox_radius": 0.75,
                    "respawn_ticks": 240,
                },
            },
        },
        "model": {
            "embed_dim": 16,
            "gru_hidden": 8,
            "head_hidden": 16,
            "action_log_std_init": -1.0,
        },
        "ppo": ppo,
        "run": {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": "runs/test_critic_warmup_anchor",
        },
    }


def _make_trainer(**ppo_overrides) -> MappoTrainer:
    config = _tiny_cfg(**ppo_overrides)
    cfg = make_mappo_config(config)
    return MappoTrainer(
        lambda: Phase4MappoEnv(
            config["env"]["sim"],
            opponent_bot="noop",
            learner_team="A",
            reward_cfg={},
        ),
        cfg,
        seed=0,
    )


def _named_param_snapshot(model: torch.nn.Module, prefix_excluded: str) -> dict:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not name.startswith(prefix_excluded)
    }


# --- schedule / config parsing ------------------------------------------


def test_anchor_kl_coef_anneals_linearly_to_zero():
    assert compute_anchor_kl_coef(update=0, initial=1.0, anneal_updates=100) == pytest.approx(1.0)
    assert compute_anchor_kl_coef(update=50, initial=1.0, anneal_updates=100) == pytest.approx(0.5)
    assert compute_anchor_kl_coef(update=100, initial=1.0, anneal_updates=100) == pytest.approx(0.0)


def test_anchor_kl_coef_noanneal_holds_constant():
    assert compute_anchor_kl_coef(update=999, initial=0.7, anneal_updates=0) == pytest.approx(0.7)


def test_make_mappo_config_parses_warm_start_stabilizers():
    cfg = make_mappo_config(
        _tiny_cfg(critic_warmup_updates=25, anchor_kl_coef=1.0, anchor_kl_anneal_updates=250)
    )
    assert cfg.critic_warmup_updates == 25
    assert cfg.anchor_kl_coef == pytest.approx(1.0)
    assert cfg.anchor_kl_anneal_updates == 250


def test_make_mappo_config_defaults_keep_stabilizers_off():
    cfg = make_mappo_config(_tiny_cfg())
    assert cfg.critic_warmup_updates == 0
    assert cfg.anchor_kl_coef == pytest.approx(0.0)
    assert cfg.anchor_kl_anneal_updates == 0


def test_make_mappo_config_rejects_negative_stabilizer_values():
    with pytest.raises(ValueError):
        make_mappo_config(_tiny_cfg(critic_warmup_updates=-1))
    with pytest.raises(ValueError):
        make_mappo_config(_tiny_cfg(anchor_kl_coef=-0.1))
    with pytest.raises(ValueError):
        make_mappo_config(_tiny_cfg(anchor_kl_anneal_updates=-1))


# --- _anchor_action_kl math ----------------------------------------------


def test_anchor_action_kl_zero_for_identical_distributions():
    mean = torch.randn(4, 3)
    log_std = torch.full((3,), -0.5)
    logits = torch.randn(4, 3)
    kl = _anchor_action_kl(
        mean=mean,
        log_std=log_std,
        binary_logits=logits,
        target_logits=None,
        anchor_mean=mean.clone(),
        anchor_log_std=log_std.clone(),
        anchor_binary_logits=logits.clone(),
        anchor_target_logits=None,
    )
    assert kl.shape == (4,)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)


def test_anchor_action_kl_positive_and_matches_gaussian_closed_form():
    # Single continuous dim, no binary divergence: KL(N(mu,s) || N(0,s)) =
    # mu^2 / (2 s^2).
    mean = torch.tensor([[1.0]])
    log_std = torch.tensor([0.0])
    logits = torch.zeros(1, 1)
    kl = _anchor_action_kl(
        mean=mean,
        log_std=log_std,
        binary_logits=logits,
        target_logits=None,
        anchor_mean=torch.zeros(1, 1),
        anchor_log_std=torch.tensor([0.0]),
        anchor_binary_logits=logits.clone(),
        anchor_target_logits=None,
    )
    assert kl.item() == pytest.approx(0.5, abs=1e-6)


def test_anchor_action_kl_finite_with_masked_binary_logits():
    # Fire-masked logits are -inf on both sides; the clamped Bernoulli KL
    # must stay finite.
    mean = torch.zeros(2, 3)
    log_std = torch.zeros(3)
    logits = torch.tensor([[0.3, -float("inf"), 0.1], [0.0, 0.0, -float("inf")]])
    kl = _anchor_action_kl(
        mean=mean,
        log_std=log_std,
        binary_logits=logits,
        target_logits=None,
        anchor_mean=mean.clone(),
        anchor_log_std=log_std.clone(),
        anchor_binary_logits=logits.clone(),
        anchor_target_logits=None,
    )
    assert torch.isfinite(kl).all()


# --- critic warmup -------------------------------------------------------


def test_critic_warmup_freezes_actor_and_trains_critic():
    trainer = _make_trainer(critic_warmup_updates=1)
    try:
        actor_before = _named_param_snapshot(trainer.model, prefix_excluded="critic")
        critic_before = {
            name: param.detach().clone()
            for name, param in trainer.model.named_parameters()
            if name.startswith("critic")
        }
        trainer.set_update_index(1)
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["critic_warmup_active"] == pytest.approx(1.0)
        for name, before in actor_before.items():
            after = dict(trainer.model.named_parameters())[name]
            assert torch.equal(before, after), f"actor/trunk param {name} moved during warmup"
        critic_moved = any(
            not torch.equal(before, dict(trainer.model.named_parameters())[name])
            for name, before in critic_before.items()
        )
        assert critic_moved, "critic params did not move during warmup"
    finally:
        trainer.close()


def test_actor_trains_again_after_critic_warmup_window():
    trainer = _make_trainer(critic_warmup_updates=1)
    try:
        trainer.set_update_index(2)
        actor_before = _named_param_snapshot(trainer.model, prefix_excluded="critic")
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["critic_warmup_active"] == pytest.approx(0.0)
        actor_moved = any(
            not torch.equal(before, dict(trainer.model.named_parameters())[name])
            for name, before in actor_before.items()
        )
        assert actor_moved, "actor params did not move after warmup ended"
    finally:
        trainer.close()


# --- anchor KL through the trainer ---------------------------------------


def test_anchor_kl_metric_is_zero_when_anchor_equals_policy():
    trainer = _make_trainer(anchor_kl_coef=1.0)
    try:
        trainer.init_anchor_from_current_model()
        trainer.set_update_index(1)
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["anchor_kl_coef"] == pytest.approx(1.0)
        assert metrics["anchor_kl"] == pytest.approx(0.0, abs=1e-6)
    finally:
        trainer.close()


def test_anchor_kl_metric_positive_after_policy_perturbation():
    trainer = _make_trainer(anchor_kl_coef=1.0)
    try:
        trainer.init_anchor_from_current_model()
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.add_(0.05 * torch.randn_like(param))
        trainer.set_update_index(1)
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["anchor_kl"] > 0.0
    finally:
        trainer.close()


def test_anchor_model_is_frozen_and_detached_copy():
    trainer = _make_trainer(anchor_kl_coef=1.0)
    try:
        trainer.init_anchor_from_current_model()
        anchor = trainer.anchor_model
        assert anchor is not None
        assert not anchor.training
        assert all(not p.requires_grad for p in anchor.parameters())
        anchor_state = copy.deepcopy(anchor.state_dict())
        with torch.no_grad():
            for param in trainer.model.parameters():
                param.add_(1.0)
        for name, tensor in anchor.state_dict().items():
            assert torch.equal(tensor, anchor_state[name]), (
                f"anchor param {name} changed when the live model moved"
            )
    finally:
        trainer.close()
