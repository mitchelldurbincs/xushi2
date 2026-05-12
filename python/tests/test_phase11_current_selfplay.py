from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv
from train.mappo import MappoActorCritic, make_mappo_config, train_phase4_from_config
from train.phases import resolve_phase
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.obs_manifest import CRITIC_DIM
from _paths import config_path


def _env() -> Phase11CurrentSelfplayMappoEnv:
    return Phase11CurrentSelfplayMappoEnv(
        {
            "round_length_seconds": 5,
            "action_repeat": 3,
            "seed": 123,
            "fog_of_war_enabled": False,
            "mechanics": {
                "revolver_damage_centi_hp": 7500,
                "revolver_fire_cooldown_ticks": 15,
                "revolver_hitbox_radius": 0.75,
                "respawn_ticks": 240,
            },
        },
        reward_cfg={"distance_shaping_coef": 0.01},
        map_randomization={"span_jitter": 2.0, "min_span": 45.0, "max_span": 55.0},
    )


def _write_phase8_snapshot(path: Path) -> None:
    with open(
        config_path("phase8_random_map_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 8,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        path,
    )


def test_phase11_current_selfplay_env_exposes_six_agent_shapes() -> None:
    env = _env()
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (6, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert obs.dtype == np.float32
        assert info["match_type"] == "current"
        assert info["learner_team"] == "both"
        assert info["loss_mask"].tolist() == [1.0] * 6
        assert "map_layout_hash" in info

        action = np.zeros((6, 6), dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == (6, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert reward.shape == (6,)
        assert reward.dtype == np.float32
        assert np.allclose(reward[:3], reward[0])
        assert np.allclose(reward[3:], reward[3])
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["match_type"] == "current"
    finally:
        env.close()


def test_phase11_anchor_match_masks_team_b_and_uses_bot_actions() -> None:
    base_env = _env()
    try:
        sim_cfg = dict(base_env._base_sim_cfg)
    finally:
        base_env.close()
    env = Phase11CurrentSelfplayMappoEnv(
        sim_cfg,
        reward_cfg={"distance_shaping_coef": 0.01},
        map_randomization={"span_jitter": 2.0, "min_span": 45.0, "max_span": 55.0},
        self_play_schedule={
            "weights": {"current": 0.0, "snapshot": 0.0, "anchor": 1.0},
            "anchor_bot": "noop",
        },
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (6, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert info["match_type"] == "anchor"
        assert info["learner_team"] == "A"
        assert info["anchor_bot"] == "noop"
        assert info["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

        _obs, _reward, _term, _trunc, step_info = env.step(np.ones((6, 6), dtype=np.float32))
        assert step_info["opponent_actions"].shape == (3, 6)
        assert np.allclose(step_info["opponent_actions"], 0.0)
    finally:
        env.close()


def test_phase11_snapshot_match_masks_team_b_and_loads_snapshot(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "phase8_snapshot.pt"
    _write_phase8_snapshot(snapshot_path)
    base_env = _env()
    try:
        sim_cfg = dict(base_env._base_sim_cfg)
    finally:
        base_env.close()
    env = Phase11CurrentSelfplayMappoEnv(
        sim_cfg,
        reward_cfg={"distance_shaping_coef": 0.01},
        map_randomization={"span_jitter": 2.0, "min_span": 45.0, "max_span": 55.0},
        self_play_schedule={"weights": {"current": 0.0, "snapshot": 1.0, "anchor": 0.0}},
        snapshot_league={
            "latest": [str(snapshot_path)],
            "weights": {"latest": 1.0},
        },
    )
    try:
        _obs, info = env.reset(seed=7)
        assert info["match_type"] == "snapshot"
        assert info["snapshot_path"] == str(snapshot_path)
        assert info["snapshot_group"] == "latest"
        assert info["loss_mask"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

        _obs, _reward, _term, _trunc, step_info = env.step(np.zeros((6, 6), dtype=np.float32))
        assert step_info["opponent_actions"].shape == (3, 6)
    finally:
        env.close()


def test_phase11_current_selfplay_builds_per_agent_team_critic_obs() -> None:
    env = _env()
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


def test_phase11_config_enables_six_agent_value_per_agent_mappo() -> None:
    with open(
        config_path("phase11/probe/phase11_current_selfplay_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    cfg = make_mappo_config(config)
    assert phase == 11
    assert spec["label"] == "phase11"
    assert cfg.n_agents == 6
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.critic_obs_dim == CRITIC_DIM
    assert cfg.value_per_agent is True
    assert config["run"]["matrix_eval"]["current_selfplay"] is True


def test_phase11_current_selfplay_train_runs_one_update(tmp_path) -> None:
    with open(
        config_path("phase11/probe/phase11_current_selfplay_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase11_current")
    result = train_phase4_from_config(config)
    assert np.isfinite(float(result["mappo"]))
    assert (tmp_path / "phase11_current" / "mappo" / "ckpt_final.pt").exists()
    matrix_path = tmp_path / "phase11_current" / "mappo" / "matrix_eval.json"
    assert matrix_path.exists()


def test_phase11_anchor_league_train_runs_one_update(tmp_path) -> None:
    with open(
        config_path("phase11/probe/phase11_current_selfplay_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["env"] = dict(config["env"])
    config["env"]["self_play_schedule"] = {
        "weights": {"current": 0.0, "snapshot": 0.0, "anchor": 1.0},
        "anchor_bot": "noop",
    }
    config["run"] = dict(config["run"])
    config["run"].pop("matrix_eval", None)
    config["run"]["output_dir"] = str(tmp_path / "phase11_anchor")
    result = train_phase4_from_config(config)
    assert np.isfinite(float(result["mappo"]))
    assert (tmp_path / "phase11_anchor" / "mappo" / "ckpt_final.pt").exists()
