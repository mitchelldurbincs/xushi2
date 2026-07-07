"""Tests for PPO warm-start stabilization: critic-only warmup, reference-policy
anchor, and the explained-variance diagnostic.

These target the 2026-06 conversion_v1 collapse mode (a stale warm-start critic
destroying the policy in the first ~25-50 updates) documented in
docs/reports/2026-07-07-phase4-getting-unstuck-review.md.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.reference_policy_anchor import (
    ReferencePolicyAnchor,
    reference_anchor_step_losses,
)


def _phase4_cfg(
    tmp_path: Path,
    *,
    critic_warmup_updates: int = 0,
    reference_anchor: dict | None = None,
) -> dict:
    ppo: dict = {
        "num_envs": 2,
        "rollout_len": 4,
        "num_epochs": 1,
        "minibatch_size": 1,
        "learning_rate": 1.0e-3,
        "value_normalization": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
    }
    if critic_warmup_updates:
        ppo["critic_warmup_updates"] = critic_warmup_updates
    if reference_anchor is not None:
        ppo["reference_anchor"] = reference_anchor
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
            "use_recurrence": True,
            "embed_dim": 16,
            "gru_hidden": 8,
            "head_hidden": 16,
            "action_log_std_init": -1.0,
        },
        "ppo": ppo,
        "run": {
            "total_updates": 10,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": str(tmp_path),
        },
    }


def _make_trainer(cfg_dict: dict) -> tuple[MappoTrainer, object]:
    cfg = make_mappo_config(cfg_dict)
    env_sim = cfg_dict["env"]["sim"]

    def env_fn():
        import envs

        return envs.Phase4MappoEnv(env_sim, opponent_bot="noop", learner_team="A", reward_cfg={})

    return MappoTrainer(env_fn, cfg, seed=0), cfg


def _save_teacher_checkpoint(path: Path, cfg_dict: dict) -> None:
    cfg = make_mappo_config(cfg_dict)
    model = MappoActorCritic(cfg)
    torch.save(
        {"config": {"mappo": asdict(cfg)}, "model_state_dict": model.state_dict()},
        path,
    )


# --- config parsing ---------------------------------------------------------


def test_make_mappo_config_parses_stabilization_fields(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_cfg(
            tmp_path,
            critic_warmup_updates=5,
            reference_anchor={
                "checkpoint": "some/teacher.pt",
                "coef": 0.05,
                "anneal_updates": 100,
                "aim_coef": 2.0,
            },
        )
    )
    assert cfg.critic_warmup_updates == 5
    assert cfg.reference_anchor_coef == pytest.approx(0.05)
    assert cfg.reference_anchor_anneal_updates == 100
    assert cfg.reference_anchor_aim_coef == pytest.approx(2.0)
    assert cfg.reference_anchor_checkpoint == "some/teacher.pt"


def test_reference_anchor_coef_without_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint is required"):
        make_mappo_config(_phase4_cfg(tmp_path, reference_anchor={"coef": 0.05}))


def test_stabilization_defaults_off(tmp_path: Path) -> None:
    cfg = make_mappo_config(_phase4_cfg(tmp_path))
    assert cfg.critic_warmup_updates == 0
    assert cfg.reference_anchor_coef == 0.0
    assert cfg.reference_anchor_checkpoint is None


# --- reference anchor unit behavior ----------------------------------------


def _tiny_anchor(tmp_path: Path, **kwargs) -> ReferencePolicyAnchor:
    cfg = make_mappo_config(_phase4_cfg(tmp_path))
    teacher = MappoActorCritic(cfg)
    defaults = dict(coef=0.05, anneal_updates=100, aim_coef=1.0, fire_coef=1.0, move_coef=1.0)
    defaults.update(kwargs)
    return ReferencePolicyAnchor(teacher=teacher, **defaults)


def test_reference_anchor_coef_anneals_linearly_to_zero(tmp_path: Path) -> None:
    anchor = _tiny_anchor(tmp_path, coef=0.05, anneal_updates=100)
    assert anchor.coef_for_update(0) == pytest.approx(0.05)
    assert anchor.coef_for_update(50) == pytest.approx(0.025)
    assert anchor.coef_for_update(100) == 0.0
    assert anchor.coef_for_update(200) == 0.0


def test_reference_anchor_no_anneal_holds_constant(tmp_path: Path) -> None:
    anchor = _tiny_anchor(tmp_path, coef=0.05, anneal_updates=0)
    assert anchor.coef_for_update(0) == pytest.approx(0.05)
    assert anchor.coef_for_update(1000) == pytest.approx(0.05)


def test_reference_anchor_freezes_teacher(tmp_path: Path) -> None:
    anchor = _tiny_anchor(tmp_path)
    assert all(not p.requires_grad for p in anchor.teacher.parameters())


def test_reference_anchor_step_losses_zero_drift_on_continuous() -> None:
    cont = torch.randn(4, 3)
    fire_logits = torch.randn(4, 3)
    fire_prob = torch.sigmoid(fire_logits[:, 0])
    aim, move, fire = reference_anchor_step_losses(cont, fire_logits, cont, fire_prob)
    # Identical continuous actions -> zero aim/move penalty.
    assert torch.allclose(aim, torch.zeros(4), atol=1e-6)
    assert torch.allclose(move, torch.zeros(4), atol=1e-6)
    # Fire BCE against the matching soft label is finite and non-negative, and
    # its gradient at the match point is zero (sigmoid(logit) - target = 0).
    assert torch.all(fire >= 0.0)
    logit = fire_logits[:, 0].clone().requires_grad_(True)
    padded = torch.zeros(4, 3)
    padded[:, 0] = logit
    _a, _m, f = reference_anchor_step_losses(cont, padded, cont, torch.sigmoid(logit).detach())
    f.sum().backward()
    assert torch.allclose(logit.grad, torch.zeros(4), atol=1e-5)


# --- critic warmup end to end ----------------------------------------------


def test_critic_warmup_freezes_actor_and_trains_critic(tmp_path: Path) -> None:
    trainer, _cfg = _make_trainer(_phase4_cfg(tmp_path, critic_warmup_updates=2))
    try:
        model = trainer.model
        actor_w = model.actor_mean_head.weight.detach().clone()
        log_std = model.log_std.detach().clone()
        critic_w = model.critic[0].weight.detach().clone()

        trainer.set_update_index(1)  # inside the warmup window
        m1 = trainer.update(trainer.collect_rollout())

        assert m1["critic_warmup"] == 1.0
        assert m1["actor_grad_norm"] == 0.0
        assert "explained_variance" in m1
        assert m1["explained_variance"] <= 1.0 + 1e-6
        # Actor parameters must be byte-for-byte unchanged during warmup.
        assert torch.equal(model.actor_mean_head.weight, actor_w)
        assert torch.equal(model.log_std, log_std)
        # The critic must have moved (value baseline recalibrating).
        assert not torch.equal(model.critic[0].weight, critic_w)

        actor_w2 = model.actor_mean_head.weight.detach().clone()
        trainer.set_update_index(3)  # past the warmup window
        m3 = trainer.update(trainer.collect_rollout())

        assert m3["critic_warmup"] == 0.0
        assert not torch.equal(model.actor_mean_head.weight, actor_w2)
    finally:
        trainer.close()


def test_no_warmup_updates_actor_immediately(tmp_path: Path) -> None:
    trainer, _cfg = _make_trainer(_phase4_cfg(tmp_path))  # critic_warmup=0
    try:
        model = trainer.model
        actor_w = model.actor_mean_head.weight.detach().clone()
        trainer.set_update_index(1)
        m1 = trainer.update(trainer.collect_rollout())
        assert m1["critic_warmup"] == 0.0
        assert not torch.equal(model.actor_mean_head.weight, actor_w)
    finally:
        trainer.close()


# --- reference anchor end to end -------------------------------------------


def test_reference_anchor_active_in_update(tmp_path: Path) -> None:
    teacher_ckpt = tmp_path / "teacher.pt"
    _save_teacher_checkpoint(teacher_ckpt, _phase4_cfg(tmp_path))
    trainer, _cfg = _make_trainer(
        _phase4_cfg(
            tmp_path,
            reference_anchor={
                "checkpoint": str(teacher_ckpt),
                "coef": 0.1,
                "anneal_updates": 100,
            },
        )
    )
    try:
        assert trainer.reference_anchor is not None
        assert all(not p.requires_grad for p in trainer.reference_anchor.teacher.parameters())
        trainer.set_update_index(1)
        m = trainer.update(trainer.collect_rollout())
        # coef at update 1 = 0.1 * (1 - 1/100)
        assert m["reference_anchor_coef"] == pytest.approx(0.099, abs=1e-4)
        assert "reference_anchor_loss" in m
        assert m["reference_anchor_loss"] == m["reference_anchor_loss"]  # finite (not NaN)
    finally:
        trainer.close()
