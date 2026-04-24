from __future__ import annotations

import numpy as np
import pytest

from train.ppo_recurrent import PPOConfig, PPOTrainer
from train.ppo_recurrent.orchestration import make_env_fn


pytest.importorskip("xushi2.xushi2_cpp")

from envs.phase3_ranger import Phase3RangerEnv


def _sim_cfg() -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 10,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def _phase3_config() -> PPOConfig:
    return PPOConfig(
        num_envs=1,
        rollout_len=4,
        obs_dim=31,
        action_dim=6,
        continuous_action_dim=3,
        embed_dim=16,
        gru_hidden=8,
        head_hidden=8,
        action_log_std_init=-1.0,
        use_recurrence=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        num_epochs=1,
        minibatch_size=1,
        binary_action_dim=3,
    )


def _env_fn() -> Phase3RangerEnv:
    return Phase3RangerEnv(_sim_cfg(), opponent_bot="basic", learner_team="A")


def test_phase3_action_adapter_produces_valid_action_dict() -> None:
    action = np.array([0.25, -0.5, 1.5, 0.2, 0.5, 0.9], dtype=np.float32)
    out = Phase3RangerEnv._action_to_dict(action)
    assert out["move_x"] == pytest.approx(0.25)
    assert out["move_y"] == pytest.approx(-0.5)
    assert out["aim_delta"] == pytest.approx(np.pi / 4.0)
    assert out["primary_fire"] == 0
    assert out["ability_1"] == 1
    assert out["ability_2"] == 1


def test_phase3_trainer_collects_real_env_rollout() -> None:
    trainer = PPOTrainer(env_fn=_env_fn, config=_phase3_config(), seed=123)
    try:
        rollout = trainer.collect_rollout()
        assert rollout.obs.shape == (1, 4, 31)
        assert rollout.action.shape == (1, 4, 6)
        assert np.isfinite(rollout.obs.numpy()).all()
        assert np.isfinite(rollout.action.numpy()).all()
        binary_actions = rollout.action[:, :, 3:].numpy()
        assert np.all((binary_actions == 0.0) | (binary_actions == 1.0))
    finally:
        trainer.envs.close()


def test_phase3_trainer_updates_on_real_env_rollout() -> None:
    trainer = PPOTrainer(env_fn=_env_fn, config=_phase3_config(), seed=321)
    try:
        rollout = trainer.collect_rollout()
        metrics = trainer.update(rollout)
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
        assert np.isfinite(metrics["entropy"])
    finally:
        trainer.envs.close()


def test_phase3_make_env_fn_preserves_reward_config() -> None:
    cfg = {
        "phase": 3,
        "env": {
            "seed_base": 123,
            "opponent_bot": "noop",
            "learner_team": "A",
            "reward": {"score_per_second": 0.1},
            "sim": _sim_cfg(),
        },
    }

    _env_fn, env_meta, seed_base = make_env_fn(cfg)

    assert seed_base == 123
    assert env_meta["opponent_bot"] == "noop"
    assert env_meta["reward"] == {"score_per_second": 0.1}
