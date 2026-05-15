from __future__ import annotations

from pathlib import Path

import pytest
import torch

from train.mappo_model import (
    MappoActorCritic,
    aim_aux_loss_and_rmse,
    aim_aux_targets,
    target_selection_aux_loss_and_accuracy,
    target_selection_aux_targets,
    wrapped_angle_error,
)
from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.ppo_recurrent.losses import (
    action_logprob_and_entropy,
    action_logprob_and_entropy_parts,
)


def _phase4_smoke_cfg(
    output_dir: Path,
    *,
    aim_aux_coef: float = 0.0,
    entropy_coef_move: float | None = None,
    entropy_coef_aim: float | None = None,
    entropy_coef_binary: float | None = None,
    mask_fire_when_no_visible_enemy: bool = False,
    target_conditioned_combat: bool = False,
    target_selection_aux_coef: float = 0.0,
) -> dict:
    ppo = {
        "num_envs": 2,
        "rollout_len": 8,
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
        "aim_aux_coef": aim_aux_coef,
        "mask_fire_when_no_visible_enemy": mask_fire_when_no_visible_enemy,
        "target_conditioned_combat": target_conditioned_combat,
        "target_selection_dim": 3 if target_conditioned_combat else 0,
        "target_selection_aux_coef": target_selection_aux_coef,
        "target_selection_aux_mode": "nearest_visible",
    }
    if entropy_coef_move is not None:
        ppo["entropy_coef_move"] = entropy_coef_move
    if entropy_coef_aim is not None:
        ppo["entropy_coef_aim"] = entropy_coef_aim
    if entropy_coef_binary is not None:
        ppo["entropy_coef_binary"] = entropy_coef_binary
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
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": str(output_dir),
        },
    }


def test_aim_aux_targets_extract_visible_enemy_angle() -> None:
    cfg = make_mappo_config(_phase4_smoke_cfg(Path("unused"), aim_aux_coef=1.0))
    obs = torch.zeros(2, cfg.obs_dim)
    obs[0, 10] = 1.0
    obs[0, 12:14] = torch.tensor([1.0, 1.0])
    obs[1, 10] = 0.0
    obs[1, 12:14] = torch.tensor([1.0, 0.0])

    target, mask = aim_aux_targets(obs, cfg)

    assert target[0].item() == pytest.approx(0.785398, abs=1.0e-5)
    assert mask.tolist() == [True, False]


def test_wrapped_angle_error_uses_shortest_signed_distance() -> None:
    pred = torch.tensor([3.13])
    target = torch.tensor([-3.13])

    err = wrapped_angle_error(pred, target)

    assert abs(err.item()) < 0.03


def test_aux_aim_head_is_opt_in_and_reports_loss(tmp_path: Path) -> None:
    cfg = make_mappo_config(_phase4_smoke_cfg(tmp_path, aim_aux_coef=1.0))
    model = MappoActorCritic(cfg)
    obs = torch.zeros(3, cfg.obs_dim)
    obs[:, 10] = 1.0
    obs[:, 12] = 1.0
    h = model.init_hidden(3)

    features, _h_next = model.actor_head_features(obs, h)
    pred = model.aim_aux_prediction_from_features(features)
    loss, rmse, count = aim_aux_loss_and_rmse(pred, obs, cfg)

    assert pred is not None
    assert count.item() == 3
    assert loss.item() >= 0.0
    assert rmse.item() >= 0.0


def test_aux_aim_head_can_warm_start_from_checkpoint_without_head(tmp_path: Path) -> None:
    base_cfg = make_mappo_config(_phase4_smoke_cfg(tmp_path / "base", aim_aux_coef=0.0))
    aux_cfg = make_mappo_config(_phase4_smoke_cfg(tmp_path / "aux", aim_aux_coef=1.0))
    base_model = MappoActorCritic(base_cfg)
    aux_model = MappoActorCritic(aux_cfg)

    result = aux_model.load_state_dict(base_model.state_dict(), strict=False)

    assert set(result.missing_keys) == {
        "actor_aim_aux_head.weight",
        "actor_aim_aux_head.bias",
    }
    assert result.unexpected_keys == []


def test_target_selection_aux_targets_choose_nearest_visible_enemy(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_smoke_cfg(tmp_path, target_conditioned_combat=True, target_selection_aux_coef=0.5)
    )
    obs = torch.zeros(3, cfg.obs_dim)
    obs[:, 6:8] = torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    obs[:, 10] = 1.0
    obs[:, 12:14] = torch.tensor([[3.0, 0.0], [1.0, 0.0], [4.0, 0.0]])

    labels, valid = target_selection_aux_targets(obs, cfg)

    assert valid.tolist() == [True, True, True]
    assert labels.tolist() == [1, 1, 1]


def test_target_conditioned_policy_adds_internal_target_head(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_smoke_cfg(tmp_path, target_conditioned_combat=True, target_selection_aux_coef=0.5)
    )
    model = MappoActorCritic(cfg)
    obs = torch.zeros(3, cfg.obs_dim)
    obs[:, 10] = 1.0
    obs[:, 12] = 1.0
    h = model.init_hidden(3)

    features, _h = model.actor_head_features(obs, h)
    mean, logits, target_logits = model.policy_heads_from_features(obs, features)
    loss, acc, count = target_selection_aux_loss_and_accuracy(target_logits, obs, cfg)

    assert model.actor_target_selection_head is not None
    assert mean.shape == (3, cfg.continuous_action_dim)
    assert logits.shape == (3, cfg.binary_action_dim)
    assert target_logits is not None
    assert target_logits.shape == (3, 3)
    assert count.item() == 3
    assert loss.item() >= 0.0
    assert 0.0 <= acc.item() <= 1.0


def test_mappo_update_logs_aux_aim_metrics(tmp_path: Path) -> None:
    cfg = make_mappo_config(_phase4_smoke_cfg(tmp_path, aim_aux_coef=1.0))
    trainer = MappoTrainer(lambda: __import__("envs").Phase4MappoEnv(
        _phase4_smoke_cfg(tmp_path)["env"]["sim"],
        opponent_bot="noop",
        learner_team="A",
        reward_cfg={},
    ), cfg, seed=0)
    try:
        metrics = trainer.update(trainer.collect_rollout())
    finally:
        trainer.close()

    assert "aim_aux_loss" in metrics
    assert "aim_aux_rmse" in metrics
    assert "aim_aux_count" in metrics


def test_entropy_parts_sum_to_existing_entropy() -> None:
    mean = torch.zeros(4, 3)
    log_std = torch.full((3,), -0.5)
    binary_logits = torch.zeros(4, 3)
    action = torch.zeros(4, 6)

    logp, entropy = action_logprob_and_entropy(mean, log_std, binary_logits, action)
    logp_parts, move_entropy, aim_entropy, binary_entropy = action_logprob_and_entropy_parts(
        mean,
        log_std,
        binary_logits,
        action,
    )

    assert torch.allclose(logp_parts, logp)
    assert torch.allclose(move_entropy + aim_entropy + binary_entropy, entropy)
    assert move_entropy.mean().item() > aim_entropy.mean().item()
    assert binary_entropy.mean().item() > 0.0


def test_make_mappo_config_parses_per_action_entropy_coefficients(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_smoke_cfg(
            tmp_path,
            entropy_coef_move=0.02,
            entropy_coef_aim=0.15,
            entropy_coef_binary=0.05,
        )
    )

    assert cfg.entropy_coef == pytest.approx(0.01)
    assert cfg.entropy_coef_move == pytest.approx(0.02)
    assert cfg.entropy_coef_aim == pytest.approx(0.15)
    assert cfg.entropy_coef_binary == pytest.approx(0.05)


def test_mappo_update_logs_per_action_entropy_bonus(tmp_path: Path) -> None:
    config = _phase4_smoke_cfg(
        tmp_path,
        entropy_coef_move=0.02,
        entropy_coef_aim=0.15,
        entropy_coef_binary=0.05,
    )
    cfg = make_mappo_config(config)
    trainer = MappoTrainer(
        lambda: __import__("envs").Phase4MappoEnv(
            config["env"]["sim"],
            opponent_bot="noop",
            learner_team="A",
            reward_cfg={},
        ),
        cfg,
        seed=0,
    )
    try:
        metrics = trainer.update(trainer.collect_rollout())
    finally:
        trainer.close()

    assert metrics["entropy_move"] > 0.0
    assert metrics["entropy_aim"] > 0.0
    assert metrics["entropy_binary"] > 0.0
    expected_bonus = (
        0.02 * metrics["entropy_move"]
        + 0.15 * metrics["entropy_aim"]
        + 0.05 * metrics["entropy_binary"]
    )
    assert metrics["entropy_bonus"] == pytest.approx(expected_bonus)


def test_make_mappo_config_parses_invalid_fire_mask_flag(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_smoke_cfg(tmp_path, mask_fire_when_no_visible_enemy=True)
    )

    assert cfg.mask_fire_when_no_visible_enemy is True


def test_invalid_fire_mask_suppresses_primary_fire_for_hidden_enemy(tmp_path: Path) -> None:
    cfg = make_mappo_config(
        _phase4_smoke_cfg(tmp_path, mask_fire_when_no_visible_enemy=True)
    )
    model = MappoActorCritic(cfg)
    with torch.no_grad():
        model.actor_binary_head.weight.zero_()
        model.actor_binary_head.bias.fill_(10.0)

    obs = torch.zeros(2, cfg.obs_dim)
    obs[1, 10] = 1.0
    obs[1, 12] = 1.0
    h = model.init_hidden(2)

    action, _h_next = model.greedy_action(obs, h)
    primary_fire = action[:, cfg.continuous_action_dim]

    assert primary_fire.tolist() == [0.0, 1.0]


def test_mappo_update_logs_invalid_fire_mask_fraction(tmp_path: Path) -> None:
    config = _phase4_smoke_cfg(tmp_path, mask_fire_when_no_visible_enemy=True)
    cfg = make_mappo_config(config)
    trainer = MappoTrainer(
        lambda: __import__("envs").Phase4MappoEnv(
            config["env"]["sim"],
            opponent_bot="noop",
            learner_team="A",
            reward_cfg={},
        ),
        cfg,
        seed=0,
    )
    try:
        metrics = trainer.update(trainer.collect_rollout())
    finally:
        trainer.close()

    assert "fire_valid_fraction" in metrics
    assert 0.0 <= metrics["fire_valid_fraction"] <= 1.0
