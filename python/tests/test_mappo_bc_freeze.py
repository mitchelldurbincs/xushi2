from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import pytest

from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv
from train.mappo_bc_pretrain import (
    _collect_walk_bc_sequence,
    _walk_and_shoot_to_objective_targets,
    bc_pretrain_walk_and_shoot_to_objective,
    load_bc_aim_target_model,
)
from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import make_mappo_config
from xushi2.obs_manifest import actor_field_slice


def _aim_only_cfg() -> dict:
    return {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "mini_game": "aim_only",
            "mini_game_config": {"episode_decisions": 4},
            "sim": {
                "seed": 0,
                "round_length_seconds": 3,
                "fog_of_war_enabled": False,
                "randomize_map": False,
                "action_repeat": 3,
                "mechanics": {
                    "revolver_damage_centi_hp": 1000,
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
        "ppo": {
            "num_envs": 1,
            "rollout_len": 4,
            "num_epochs": 1,
            "minibatch_size": 1,
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
        },
    }


def test_bc_freeze_actor_aim_masks_shared_trunk_and_aim_row() -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    torch.manual_seed(0)
    model = MappoActorCritic(cfg)

    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    bc_pretrain_walk_and_shoot_to_objective(
        model,
        lambda: Phase4AimOnlyMappoEnv(episode_decisions=4),
        cfg,
        steps=3,
        batch_size=12,
        learning_rate=1.0e-2,
        seed=0,
        log_label="test",
        freeze_actor_aim=True,
    )

    after = dict(model.named_parameters())
    assert torch.allclose(after["actor_embed.0.weight"], before["actor_embed.0.weight"])
    assert torch.allclose(after["actor_gru.weight_ih"], before["actor_gru.weight_ih"])
    assert torch.allclose(after["actor_body.0.weight"], before["actor_body.0.weight"])
    assert torch.allclose(after["actor_mean_head.weight"][2], before["actor_mean_head.weight"][2])
    assert torch.allclose(after["actor_mean_head.bias"][2], before["actor_mean_head.bias"][2])

    move_delta = (after["actor_mean_head.weight"][:2] - before["actor_mean_head.weight"][:2]).abs()
    binary_delta = (after["actor_binary_head.weight"] - before["actor_binary_head.weight"]).abs()
    assert float(move_delta.max().item()) > 0.0
    assert float(binary_delta.max().item()) > 0.0


def test_bc_without_freeze_updates_actor_trunk_and_aim_row() -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    torch.manual_seed(0)
    model = MappoActorCritic(cfg)

    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    bc_pretrain_walk_and_shoot_to_objective(
        model,
        lambda: Phase4AimOnlyMappoEnv(episode_decisions=4),
        cfg,
        steps=3,
        batch_size=12,
        learning_rate=1.0e-2,
        seed=0,
        log_label="test",
    )

    after = dict(model.named_parameters())
    assert not torch.allclose(after["actor_embed.0.weight"], before["actor_embed.0.weight"])
    assert not torch.allclose(
        after["actor_mean_head.weight"][2], before["actor_mean_head.weight"][2]
    )


def _constant_aim_checkpoint(path: Path, aim: float) -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    model = MappoActorCritic(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.actor_mean_head.bias[2] = torch.atanh(torch.tensor(aim))
    torch.save(
        {
            "config": {"mappo": asdict(cfg)},
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def test_load_bc_aim_target_model_freezes_checkpoint(tmp_path: Path) -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    ckpt = tmp_path / "aim_target.pt"
    _constant_aim_checkpoint(ckpt, aim=0.25)

    model = load_bc_aim_target_model(ckpt, cfg)

    assert not model.training
    assert all(not param.requires_grad for param in model.parameters())


def test_walk_and_shoot_bc_can_replace_only_aim_from_checkpoint(tmp_path: Path) -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    ckpt = tmp_path / "aim_target.pt"
    _constant_aim_checkpoint(ckpt, aim=0.25)
    aim_target_model = load_bc_aim_target_model(ckpt, cfg)

    obs_seq, target_seq = _collect_walk_bc_sequence(
        lambda: Phase4AimOnlyMappoEnv(episode_decisions=4),
        cfg,
        batch_size=12,
        seed=0,
        target_fn=_walk_and_shoot_to_objective_targets,
        aim_target_model=aim_target_model,
    )
    crude_targets = _walk_and_shoot_to_objective_targets(obs_seq[0], cfg)

    assert torch.allclose(target_seq[0, :, 2], torch.full((3,), 0.25), atol=1.0e-6)
    assert torch.allclose(target_seq[0, :, :2], crude_targets[:, :2])
    assert torch.allclose(target_seq[0, :, 3:], crude_targets[:, 3:])


def test_walk_and_shoot_bc_aim_target_is_relative_to_current_aim() -> None:
    cfg = make_mappo_config(_aim_only_cfg())
    obs = torch.zeros(2, cfg.obs_dim, dtype=torch.float32)
    enemy_alive = actor_field_slice("enemy_alive")
    enemy_rel = actor_field_slice("enemy_relative_position")
    own_aim = actor_field_slice("own_aim_unit")

    obs[:, enemy_alive] = 1.0
    obs[:, enemy_rel] = torch.tensor([0.0, 1.0])
    obs[0, own_aim] = torch.tensor([0.0, 1.0])  # current aim angle 0 rad.
    obs[1, own_aim] = torch.tensor([1.0, 0.0])  # already aimed at +pi/2.

    target = _walk_and_shoot_to_objective_targets(obs, cfg)

    assert target[0, 2] > 0.9
    assert target[1, 2] == pytest.approx(0.0, abs=1.0e-6)
