from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from envs.phase4_selfplay_mappo import Phase4CurrentSelfplayMappoEnv
from train.mappo import make_mappo_config, train_phase4_from_config
from train.mappo_bc_pretrain import (
    _collect_walk_bc_sequence,
    _walk_and_shoot_to_objective_targets,
)
from train.phases import resolve_phase
from xushi2.obs_manifest import ACTOR_PHASE1_DIM, CRITIC_DIM
from xushi2.obs_manifest import actor_field_slice
from _paths import config_path


def _sim_cfg() -> dict:
    return {
        "round_length_seconds": 5,
        "action_repeat": 3,
        "seed": 123,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "mechanics": {
            "revolver_damage_centi_hp": 1000,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def test_phase4_current_selfplay_env_exposes_six_flat_agents() -> None:
    env = Phase4CurrentSelfplayMappoEnv(
        _sim_cfg(),
        reward_cfg={"distance_shaping_coef": 0.005},
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (6, ACTOR_PHASE1_DIM)
        assert obs.dtype == np.float32
        assert info["match_type"] == "current"
        assert info["learner_team"] == "both"
        assert info["loss_mask"].tolist() == [1.0] * 6

        action = np.zeros((6, 6), dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == (6, ACTOR_PHASE1_DIM)
        assert reward.shape == (6,)
        assert reward.dtype == np.float32
        assert np.allclose(reward[:3], reward[0])
        assert np.allclose(reward[3:], reward[3])
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["loss_mask"].tolist() == [1.0] * 6
        assert "objective_metrics" in step_info
        assert "combat_metrics" in step_info
    finally:
        env.close()


def test_phase4_anchor_mix_masks_team_b_and_uses_bot_actions() -> None:
    env = Phase4CurrentSelfplayMappoEnv(
        _sim_cfg(),
        reward_cfg={"distance_shaping_coef": 0.005},
        self_play_schedule={
            "weights": {"current": 0.0, "snapshot": 0.0, "anchor": 1.0},
            "anchor_bot": "noop",
        },
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (6, ACTOR_PHASE1_DIM)
        assert info["match_type"] == "anchor"
        assert info["learner_team"] == "A"
        assert info["anchor_bot"] == "noop"
        assert info["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

        _obs, _reward, _term, _trunc, step_info = env.step(
            np.ones((6, 6), dtype=np.float32)
        )
        assert step_info["opponent_actions"].shape == (3, 6)
        assert np.allclose(step_info["opponent_actions"], 0.0)
        assert step_info["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    finally:
        env.close()


def test_phase4_current_selfplay_builds_per_agent_team_critic_obs() -> None:
    env = Phase4CurrentSelfplayMappoEnv(_sim_cfg())
    try:
        env.reset(seed=11)
        critic_obs = np.zeros(6 * CRITIC_DIM, dtype=np.float32)
        env.build_critic_obs(critic_obs)
        views = critic_obs.reshape(6, CRITIC_DIM)
        assert np.isfinite(views).all()
        assert np.allclose(views[0], views[1])
        assert np.allclose(views[0], views[2])
        assert np.allclose(views[3], views[4])
        assert np.allclose(views[3], views[5])
        assert not np.allclose(views[0], views[3])
    finally:
        env.close()


def test_phase4_current_selfplay_config_uses_six_agent_value_per_agent_mappo() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 4
    assert spec["label"] == "phase4_selfplay"
    assert cfg.n_agents == 6
    assert cfg.obs_dim == ACTOR_PHASE1_DIM
    assert cfg.critic_obs_dim == CRITIC_DIM
    assert cfg.value_per_agent is True
    assert ckpt_env_cfg["self_play"]["enabled"] is True
    assert ckpt_env_cfg["match_type"] == "current"


def test_phase4_selfplay_config_preserves_anchor_schedule() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["env"] = dict(config["env"])
    config["env"]["self_play_schedule"] = {
        "weights": {"current": 0.25, "snapshot": 0.0, "anchor": 0.75},
        "anchor_bot": "weak_basic_v2",
    }

    _phase, spec = resolve_phase(config)
    env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    assert ckpt_env_cfg["self_play_schedule"]["weights"]["current"] == 0.25
    assert ckpt_env_cfg["self_play_schedule"]["anchor_bot"] == "weak_basic_v2"
    env = env_fn()
    try:
        _obs, info = env.reset(seed=8)
        assert info["schedule"] == "current:0.25,anchor:0.75"
    finally:
        env.close()


def test_phase4_current_selfplay_walk_and_shoot_bc_targets_all_six_agents() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    _phase, spec = resolve_phase(config)
    env_fn, _ckpt_env_cfg, seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)

    obs_seq, target_seq = _collect_walk_bc_sequence(
        env_fn,
        cfg,
        batch_size=12,
        seed=seed + 123,
        target_fn=_walk_and_shoot_to_objective_targets,
    )

    assert obs_seq.shape == (2, 6, ACTOR_PHASE1_DIM)
    assert target_seq.shape == (2, 6, 6)
    assert torch.isfinite(obs_seq).all()
    assert torch.isfinite(target_seq).all()

    own_position_slice = actor_field_slice("own_position")
    own_pos = obs_seq[0, :, own_position_slice]
    move = target_seq[0, :, :2]
    direction_to_objective = -own_pos
    direction_norm = torch.linalg.vector_norm(direction_to_objective, dim=-1)
    move_norm = torch.linalg.vector_norm(move, dim=-1)
    aligned = (move * direction_to_objective).sum(dim=-1) / (
        direction_norm * move_norm
    ).clamp(min=1.0e-6)

    assert torch.all(move_norm > 0.9)
    assert torch.all(aligned > 0.99)


def test_phase4_current_selfplay_bc_actions_move_both_teams_toward_objective() -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    _phase, spec = resolve_phase(config)
    env_fn, _ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    own_position_slice = actor_field_slice("own_position")

    env = env_fn()
    try:
        obs, _info = env.reset(seed=123)
        start_distance = np.linalg.norm(obs[:, own_position_slice], axis=1)
        for _ in range(60):
            target = _walk_and_shoot_to_objective_targets(
                torch.as_tensor(obs, dtype=torch.float32), cfg
            )
            obs, _reward, terminated, truncated, _info = env.step(target.numpy())
            if terminated or truncated:
                break
        end_distance = np.linalg.norm(obs[:, own_position_slice], axis=1)
    finally:
        env.close()

    assert end_distance[:3].mean() < start_distance[:3].mean() - 0.2
    assert end_distance[3:].mean() < start_distance[3:].mean() - 0.2


def test_phase4_current_selfplay_train_runs_one_update(tmp_path: Path) -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase4_current_selfplay")
    result = train_phase4_from_config(config)
    assert np.isfinite(float(result["mappo"]))
    assert (
        tmp_path / "phase4_current_selfplay" / "mappo" / "ckpt_final.pt"
    ).exists()
