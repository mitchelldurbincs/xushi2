from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from train.composition_rehearsal import (
    build_mappo_env_fn_with_overrides,
    composition_rehearsal_losses,
    composition_rehearsal_pretrain,
    load_frozen_mappo_teacher,
)
from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv
from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import make_mappo_config
from train.train import load_config
from xushi2.obs_manifest import ACTOR_PHASE1_DIM


class ConstantObsMappoEnv(gym.Env):
    n_agents = 3
    actor_obs_dim = ACTOR_PHASE1_DIM
    critic_obs_dim = 135
    action_dim = 6

    def __init__(self, *, episode_decisions: int = 2) -> None:
        super().__init__()
        self.episode_decisions = int(episode_decisions)
        self._tick = 0
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3, ACTOR_PHASE1_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3, 6),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._tick = 0
        return np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32), {}

    def step(self, action):
        del action
        self._tick += 1
        obs = np.zeros((3, ACTOR_PHASE1_DIM), dtype=np.float32)
        return obs, np.zeros(3, dtype=np.float32), False, self._tick >= self.episode_decisions, {}

    def close(self) -> None:
        return None


def _cfg_dict() -> dict:
    return {
        "phase": 4,
        "env": {
            "seed_base": 0,
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
            "embed_dim": 8,
            "gru_hidden": 8,
            "head_hidden": 8,
            "action_log_std_init": -1.0,
        },
        "ppo": {
            "num_envs": 1,
            "rollout_len": 2,
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


def _zero_model() -> MappoActorCritic:
    model = MappoActorCritic(make_mappo_config(_cfg_dict()))
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model


def _constant_teacher(
    *,
    move_x: float,
    move_y: float,
    aim: float,
    fire_logit: float,
) -> MappoActorCritic:
    model = _zero_model()
    with torch.no_grad():
        model.actor_mean_head.bias[0] = torch.atanh(torch.tensor(move_x))
        model.actor_mean_head.bias[1] = torch.atanh(torch.tensor(move_y))
        model.actor_mean_head.bias[2] = torch.atanh(torch.tensor(aim))
        model.actor_binary_head.bias[0] = fire_logit
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def test_row_scoped_losses_only_touch_intended_action_heads() -> None:
    student = _zero_model()
    obs = torch.zeros(2, 3, student.cfg.obs_dim)
    objective_cont = torch.zeros(2, 3, student.cfg.continuous_action_dim)
    objective_cont[:, :, 0] = 0.4
    objective_cont[:, :, 1] = -0.3
    objective_cont[:, :, 2] = 0.9
    combat_cont = torch.zeros(2, 3, student.cfg.continuous_action_dim)
    combat_cont[:, :, 0] = -0.9
    combat_cont[:, :, 1] = 0.9
    combat_cont[:, :, 2] = 0.25
    combat_binary = torch.zeros(2, 3, student.cfg.binary_action_dim)
    combat_binary[:, :, 0] = 1.0
    combat_binary[:, :, 1:] = 1.0

    loss, _parts = composition_rehearsal_losses(
        student,
        obs,
        objective_cont,
        obs,
        combat_cont,
        combat_binary,
    )
    loss.backward()

    mean_grad = student.actor_mean_head.bias.grad
    binary_grad = student.actor_binary_head.bias.grad
    assert mean_grad is not None
    assert binary_grad is not None
    assert float(mean_grad[0].abs().item()) > 0.0
    assert float(mean_grad[1].abs().item()) > 0.0
    assert float(mean_grad[2].abs().item()) > 0.0
    assert float(binary_grad[0].abs().item()) > 0.0
    assert torch.allclose(binary_grad[1:], torch.zeros_like(binary_grad[1:]))


def test_frozen_teachers_do_not_receive_gradients() -> None:
    student = _zero_model()
    objective_teacher = _constant_teacher(move_x=0.4, move_y=-0.4, aim=0.0, fire_logit=0.0)
    combat_teacher = _constant_teacher(move_x=0.0, move_y=0.0, aim=0.3, fire_logit=4.0)

    composition_rehearsal_pretrain(
        student,
        objective_teacher,
        combat_teacher,
        lambda: ConstantObsMappoEnv(),
        lambda: ConstantObsMappoEnv(),
        {
            "steps": 1,
            "objective_batch_size": 6,
            "combat_batch_size": 6,
            "learning_rate": 1.0e-2,
            "seed": 0,
            "log_label": "test",
        },
    )

    assert all(param.grad is None for param in objective_teacher.parameters())
    assert all(param.grad is None for param in combat_teacher.parameters())
    assert all(not param.requires_grad for param in objective_teacher.parameters())
    assert all(not param.requires_grad for param in combat_teacher.parameters())


def test_student_preserves_both_mock_skills_after_short_rehearsal() -> None:
    student = _zero_model()
    objective_teacher = _constant_teacher(move_x=0.5, move_y=-0.5, aim=0.0, fire_logit=0.0)
    combat_teacher = _constant_teacher(move_x=0.0, move_y=0.0, aim=0.35, fire_logit=5.0)

    composition_rehearsal_pretrain(
        student,
        objective_teacher,
        combat_teacher,
        lambda: ConstantObsMappoEnv(),
        lambda: ConstantObsMappoEnv(),
        {
            "steps": 40,
            "objective_batch_size": 6,
            "combat_batch_size": 6,
            "learning_rate": 5.0e-2,
            "seed": 0,
            "log_label": "test",
        },
    )

    obs = torch.zeros(3, student.cfg.obs_dim)
    h = student.init_hidden(3)
    with torch.no_grad():
        features, _h = student.actor_head_features(obs, h)
        mean, logits, _target_selection_logits = student.policy_heads_from_features(obs, features)
        cont = torch.tanh(mean).mean(dim=0)
        fire = torch.sigmoid(logits[:, 0]).mean()
    assert cont[0] > 0.35
    assert cont[1] < -0.35
    assert cont[2] > 0.20
    assert fire > 0.80


def test_load_frozen_mappo_teacher_freezes_checkpoint(tmp_path: Path) -> None:
    teacher = _constant_teacher(move_x=0.1, move_y=-0.1, aim=0.2, fire_logit=1.0)
    ckpt = tmp_path / "teacher.pt"
    torch.save(
        {"config": {"mappo": asdict(teacher.cfg)}, "model_state_dict": teacher.state_dict()},
        ckpt,
    )

    loaded = load_frozen_mappo_teacher(ckpt)

    assert not loaded.training
    assert all(not param.requires_grad for param in loaded.parameters())


def test_config_loading_adds_composition_defaults_and_probe_config_loads() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/configs/phase4/probe/phase4_mappo_composition_rehearsal_v1.yaml"
    )
    config = load_config(config_path)
    run_cfg = config["run"]

    assert run_cfg["composition_pretrain"] is True
    assert run_cfg["composition_pretrain_steps"] == 1000
    assert run_cfg["composition_objective_batch_size"] == 256
    assert run_cfg["composition_combat_batch_size"] == 256
    assert run_cfg["composition_gate"]["hit_fire_gate"] == 0.02
    assert run_cfg["composition_objective_env"]["opponent_bot"] == "weak_basic_v2"
    assert run_cfg["composition_combat_env"]["mini_game"] == "combat_1v1"


@pytest.mark.parametrize(
    "config_name,expected_steps",
    [
        ("phase4_mappo_composition_rehearsal_cap_duel_v2.yaml", 2000),
        ("phase4_mappo_composition_rehearsal_cap_duel_v2_4000.yaml", 4000),
    ],
)
def test_cap_duel_v2_combat_env_path_honors_knobs_and_has_finite_bc_loss(
    config_name: str,
    expected_steps: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "experiments/configs/phase4/probe/"
        / config_name
    )
    config = load_config(config_path)
    run_cfg = config["run"]
    assert run_cfg["composition_pretrain_steps"] == expected_steps
    env_fn = build_mappo_env_fn_with_overrides(
        config["env"],
        run_cfg["composition_combat_env"],
    )

    env = env_fn()
    try:
        assert isinstance(env, Phase4CapDuelMappoEnv)
        assert env.episode_decisions == 96
        assert env.point_radius == 0.18
        assert env.enemy_hp == 3
        assert env.score_ticks_to_clear == 12
        assert env.enemy_recontest_delay == 12
        assert env.hit_tolerance == 0.12
        assert env.hit_reward == 1.0
        assert env.kill_bonus == 4.0
        assert env.score_per_tick == 0.1
        assert env.off_point_penalty == 0.0
        assert env.time_penalty_per_decision == 0.0
        assert env._hit_push == 0.0
        assert env.spawn_distance == 0.4
        assert env.respawn_at_spawn_position is True

        obs, _info = env.reset(seed=123)
        assert obs.shape == (3, ACTOR_PHASE1_DIM)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
    finally:
        env.close()

    student = _zero_model()
    objective_teacher = _constant_teacher(move_x=0.2, move_y=-0.1, aim=0.0, fire_logit=0.0)
    combat_teacher = _constant_teacher(move_x=0.0, move_y=0.0, aim=0.2, fire_logit=3.0)
    metrics = composition_rehearsal_pretrain(
        student,
        objective_teacher,
        combat_teacher,
        lambda: ConstantObsMappoEnv(),
        env_fn,
        {
            "steps": 1,
            "objective_batch_size": 3,
            "combat_batch_size": 3,
            "learning_rate": 1.0e-2,
            "seed": 123,
            "log_label": "test",
        },
    )

    assert metrics
    assert all(np.isfinite(float(value)) for value in metrics.values())
