from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train.full_env_rehearsal import (
    _collect_scripted_batch,
    _cpp_bot_targets,
    full_env_rehearsal_pretrain,
    full_env_rehearsal_loss,
    multi_enemy_visible_targets,
    run_full_env_rehearsal_gate,
    scripted_objective_focus_fire_targets,
)
from train.mappo_model import MappoActorCritic, MappoConfig
from envs.phase4_mappo import Phase4MappoEnv
from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv
from xushi2.entity_obs import ENTITY_TOKEN_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.grid_obs import GRID_CHANNELS, GRID_SIZE, MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM, actor_field_slice

_OWN_AIM = actor_field_slice("own_aim_unit")
_OWN_POSITION = actor_field_slice("own_position")
_SELF_ON_POINT = actor_field_slice("self_on_point")
_ENEMY_ALIVE = actor_field_slice("enemy_alive")
_ENEMY_REL_POS = actor_field_slice("enemy_relative_position")
_ENEMY_HP = actor_field_slice("enemy_hp")
_SELF_TOKEN = 0
_FIRST_ENEMY_TOKEN = 1
_OBJECTIVE_TOKEN = 4
_ENTITY_POSITION = slice(8, 10)
_ENTITY_AIM = slice(12, 14)
_ENTITY_AUX = 17


def _cfg(*, target_conditioned: bool = True) -> MappoConfig:
    return MappoConfig(
        num_envs=1,
        n_agents=3,
        rollout_len=4,
        obs_dim=ACTOR_PHASE1_DIM,
        critic_obs_dim=CRITIC_DIM,
        action_dim=6,
        continuous_action_dim=3,
        binary_action_dim=3,
        embed_dim=16,
        gru_hidden=8,
        head_hidden=16,
        action_log_std_init=-1.0,
        gamma=0.997,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=3.0e-4,
        num_epochs=1,
        minibatch_size=1,
        target_selection_dim=4 if target_conditioned else 0,
        target_conditioned_combat=target_conditioned,
        target_selection_aux_coef=0.5 if target_conditioned else 0.0,
        target_selection_aux_mode="team_focus_low_hp",
    )


def _multi_enemy_cfg() -> MappoConfig:
    return MappoConfig(
        num_envs=1,
        n_agents=3,
        rollout_len=4,
        obs_dim=MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
        critic_obs_dim=CRITIC_DIM,
        action_dim=6,
        continuous_action_dim=3,
        binary_action_dim=3,
        embed_dim=16,
        gru_hidden=8,
        head_hidden=16,
        action_log_std_init=-1.0,
        gamma=0.997,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=3.0e-4,
        num_epochs=1,
        minibatch_size=1,
        obs_encoder="entity_attention_grid",
        entity_token_count=MULTI_ENEMY_TOKEN_COUNT,
        entity_token_dim=ENTITY_TOKEN_DIM,
        entity_num_heads=1,
        grid_channels=GRID_CHANNELS,
        grid_size=GRID_SIZE,
    )


def _multi_enemy_obs(cfg: MappoConfig) -> torch.Tensor:
    obs = torch.zeros(3, cfg.obs_dim)
    token_width = cfg.entity_token_count * cfg.entity_token_dim
    tokens = obs[:, :token_width].view(3, cfg.entity_token_count, cfg.entity_token_dim)
    mask = obs[:, token_width : token_width + cfg.entity_token_count]
    mask[:, _SELF_TOKEN] = 1.0
    mask[:, _OBJECTIVE_TOKEN] = 1.0
    tokens[:, _SELF_TOKEN, _ENTITY_AIM] = torch.tensor([0.0, 1.0])
    tokens[:, _OBJECTIVE_TOKEN, _ENTITY_POSITION] = torch.tensor([0.5, 0.0])
    return obs


def _obs(cfg: MappoConfig) -> torch.Tensor:
    obs = torch.zeros(3, cfg.obs_dim)
    obs[:, _OWN_AIM] = torch.tensor([0.0, 1.0])
    return obs


def _sim_cfg() -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 2,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 1000,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def test_scripted_teacher_actions_are_finite_and_bounded() -> None:
    cfg = _cfg()
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.5, 0.0], [0.0, -0.5], [0.0, 0.0]])
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    cont, binary = scripted_objective_focus_fire_targets(obs, cfg)

    assert cont.shape == (3, 3)
    assert binary.shape == (3, 3)
    assert torch.isfinite(cont).all()
    assert torch.isfinite(binary).all()
    assert float(cont.min()) >= -1.0
    assert float(cont.max()) <= 1.0
    assert set(binary[:, 0].tolist()) == {1.0}


def test_scripted_teacher_aim_delta_uses_sin_cos_convention() -> None:
    cfg = _cfg()
    obs = _obs(cfg)
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[0, _OWN_AIM] = torch.tensor([0.0, 1.0])  # angle 0, facing +y.
    obs[0, _ENEMY_REL_POS] = torch.tensor([1.0, 0.0])  # target at +x, +pi/2.
    obs[1, _OWN_AIM] = torch.tensor([1.0, 0.0])  # already facing +x.
    obs[1, _ENEMY_REL_POS] = torch.tensor([1.0, 0.0])
    obs[2, _OWN_AIM] = torch.tensor([0.0, 1.0])
    obs[2, _ENEMY_REL_POS] = torch.tensor([0.0, 1.0])  # target straight ahead.

    cont, _binary = scripted_objective_focus_fire_targets(obs, cfg)

    assert cont[0, 2].item() == pytest.approx(1.0)
    assert cont[1, 2].item() == pytest.approx(0.0)
    assert cont[2, 2].item() == pytest.approx(0.0)


def test_hidden_enemy_rows_emit_no_fire_and_no_target() -> None:
    cfg = _cfg()
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.5, 0.0], [0.2, 0.0], [-0.2, 0.0]])
    obs[0, _ENEMY_ALIVE] = 1.0
    obs[0, _ENEMY_REL_POS] = torch.tensor([1.0, 0.0])
    obs[0, _ENEMY_HP] = 0.2

    _cont, binary = scripted_objective_focus_fire_targets(obs, cfg)
    model = MappoActorCritic(cfg)
    loss, parts = full_env_rehearsal_loss(
        model,
        obs,
        {"target_selection_coef": 0.5},
    )

    assert binary[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert torch.isfinite(loss)
    assert parts["target_selection_count"].item() == pytest.approx(3.0)


def test_scripted_teacher_uses_actor_obs_only() -> None:
    params = list(inspect.signature(scripted_objective_focus_fire_targets).parameters)
    assert params == ["actor_obs", "cfg"]


def test_one_full_env_rehearsal_update_has_finite_losses() -> None:
    cfg = _cfg()
    obs = _obs(cfg)
    obs[:, _OWN_POSITION] = torch.tensor([[0.4, 0.0], [0.1, -0.2], [-0.3, 0.1]])
    obs[:, _ENEMY_ALIVE] = 1.0
    obs[:, _ENEMY_REL_POS] = torch.tensor([[1.0, 0.0], [0.5, 0.2], [0.2, -0.4]])
    obs[:, _ENEMY_HP] = torch.tensor([[0.8], [0.2], [0.5]])
    model = MappoActorCritic(cfg)

    loss, parts = full_env_rehearsal_loss(
        model,
        obs,
        {
            "movement_coef": 1.0,
            "aim_coef": 1.0,
            "fire_coef": 1.0,
            "target_selection_coef": 0.5,
        },
    )

    assert torch.isfinite(loss)
    for value in parts.values():
        assert torch.isfinite(value)


def test_cpp_teacher_targets_are_finite_and_bounded() -> None:
    cfg = _cfg(target_conditioned=False)
    env = Phase4MappoEnv(_sim_cfg(), opponent_bot="noop")
    try:
        env.reset(seed=0)
        cont, binary = _cpp_bot_targets(env, cfg, "basic")
    finally:
        env.close()

    assert cont.shape == (3, cfg.continuous_action_dim)
    assert binary.shape == (3, cfg.binary_action_dim)
    assert torch.isfinite(cont).all()
    assert torch.isfinite(binary).all()
    assert float(cont.min()) >= -1.0
    assert float(cont.max()) <= 1.0
    assert float(binary.min()) >= 0.0
    assert float(binary.max()) <= 1.0


def test_multi_enemy_visible_teacher_uses_masks_and_nearest_visible_enemy() -> None:
    cfg = _multi_enemy_cfg()
    obs = _multi_enemy_obs(cfg)
    token_width = cfg.entity_token_count * cfg.entity_token_dim
    tokens = obs[:, :token_width].view(3, cfg.entity_token_count, cfg.entity_token_dim)
    mask = obs[:, token_width : token_width + cfg.entity_token_count]
    mask[0, _FIRST_ENEMY_TOKEN] = 1.0
    mask[0, _FIRST_ENEMY_TOKEN + 1] = 1.0
    tokens[0, _FIRST_ENEMY_TOKEN, _ENTITY_POSITION] = torch.tensor([0.8, 0.0])
    tokens[0, _FIRST_ENEMY_TOKEN + 1, _ENTITY_POSITION] = torch.tensor([0.0, 0.2])
    mask[1, _FIRST_ENEMY_TOKEN] = 0.0
    tokens[1, _FIRST_ENEMY_TOKEN, _ENTITY_POSITION] = torch.tensor([0.1, 0.0])

    cont, binary = multi_enemy_visible_targets(obs, cfg)

    assert cont.shape == (3, cfg.continuous_action_dim)
    assert binary.shape == (3, cfg.binary_action_dim)
    assert binary[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert cont[0, 2].item() == pytest.approx(1.0)
    assert torch.isfinite(cont).all()


def test_collect_scripted_batch_can_use_multi_enemy_visible_teacher() -> None:
    cfg = _multi_enemy_cfg()

    batch, target_cont, target_binary = _collect_scripted_batch(
        lambda: Phase4MultiEnemyMappoEnv(_sim_cfg(), opponent_bot="noop"),
        cfg,
        batch_size=6,
        seed=0,
        teacher="multi_enemy_visible",
    )

    assert batch.shape == (6, cfg.obs_dim)
    assert target_cont.shape == (6, cfg.continuous_action_dim)
    assert target_binary.shape == (6, cfg.binary_action_dim)
    assert torch.isfinite(batch).all()
    assert torch.isfinite(target_cont).all()
    assert torch.isfinite(target_binary).all()


def test_collect_scripted_batch_can_use_cpp_teacher() -> None:
    cfg = _cfg(target_conditioned=False)

    batch, target_cont, target_binary = _collect_scripted_batch(
        lambda: Phase4MappoEnv(_sim_cfg(), opponent_bot="noop"),
        cfg,
        batch_size=6,
        seed=0,
        teacher="cpp_basic",
    )

    assert batch.shape == (6, cfg.obs_dim)
    assert target_cont.shape == (6, cfg.continuous_action_dim)
    assert target_binary.shape == (6, cfg.binary_action_dim)
    assert torch.isfinite(batch).all()
    assert torch.isfinite(target_cont).all()
    assert torch.isfinite(target_binary).all()


def test_cpp_teacher_rehearsal_pretrain_one_step_is_finite() -> None:
    cfg = _cfg(target_conditioned=False)
    model = MappoActorCritic(cfg)

    metrics = full_env_rehearsal_pretrain(
        model,
        lambda: Phase4MappoEnv(_sim_cfg(), opponent_bot="noop"),
        {
            "steps": 1,
            "batch_size": 6,
            "learning_rate": 1.0e-4,
            "seed": 0,
            "teacher": "cpp_basic",
            "target_selection_coef": 0.0,
        },
    )

    assert metrics["loss"] >= 0.0
    assert metrics["move_loss"] >= 0.0
    assert metrics["aim_loss"] >= 0.0
    assert metrics["fire_loss"] >= 0.0


def test_full_env_rehearsal_gate_writes_not_reached_and_blocks_ppo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(target_conditioned=False)
    model = MappoActorCritic(cfg)

    def fake_eval(*_args, **_kwargs):
        return SimpleNamespace(
            team_a_hit_fire=0.0,
            team_a_visible_fire_rate=0.0,
            mean_majority_on_point_seconds_a=0.0,
            mean_final_tick=1800.0,
            losses=50,
            wins=0,
            mean_team_a_score=0.0,
            mean_team_b_score=37.0,
        )

    monkeypatch.setattr("train.full_env_rehearsal.evaluate_mappo", fake_eval)
    gate = run_full_env_rehearsal_gate(
        model,
        lambda: None,
        gate={
            "episodes": 50,
            "min_team_a_hit_fire": 0.04,
            "min_objective_on_point": 0.25,
            "max_losses": 49,
        },
        output_dir=tmp_path,
        seed=0,
        checkpoint_path=tmp_path / "ckpt_full_env_rehearsal.pt",
    )

    assert gate.status == "NOT_REACHED"
    assert not gate.passed
    assert (tmp_path / "full_env_rehearsal_gate.json").exists()
    assert gate.thresholds["min_mean_score_a"] == pytest.approx(0.0)
