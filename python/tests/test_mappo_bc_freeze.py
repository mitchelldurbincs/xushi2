from __future__ import annotations

import torch

from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv
from train.mappo_bc_pretrain import bc_pretrain_walk_and_shoot_to_objective
from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import make_mappo_config


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
