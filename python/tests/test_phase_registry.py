from pathlib import Path

import yaml
import torch

from train.mappo import (
    MappoTrainer,
    _walk_to_objective_targets,
    evaluate_mappo,
    make_mappo_config,
    train_phase4_from_config,
)
from train.phases import PHASE_REGISTRY, resolve_phase
from xushi2.entity_obs import (
    ENTITY_OBS_DIM,
    ENTITY_TOKEN_COUNT,
    ENTITY_TOKEN_DIM,
    MULTI_ENEMY_TOKEN_COUNT,
)
from xushi2.grid_obs import (
    ENTITY_GRID_OBS_DIM,
    GRID_CHANNELS,
    GRID_SIZE,
    MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
)

PHASE10_TARGET_OBS_DIM = MULTI_ENEMY_ENTITY_GRID_OBS_DIM + MULTI_ENEMY_TOKEN_COUNT


def test_phase_registry_entries_have_required_shapes() -> None:
    for phase, spec in PHASE_REGISTRY.items():
        assert "label" in spec
        assert "training_variants" in spec
        variants = spec["training_variants"]
        assert isinstance(variants, tuple)
        if variants:
            for key in (
                "obs_dim",
                "action_dim",
                "continuous_action_dim",
                "binary_action_dim",
                "env_bundle",
            ):
                assert key in spec, f"phase={phase} missing {key}"
            assert callable(spec["env_bundle"])
        else:
            assert "seed_deriver" in spec
            assert callable(spec["seed_deriver"])


def test_phase4_registry_declares_mappo_shapes() -> None:
    phase, spec = resolve_phase({"phase": 4})
    assert phase == 4
    assert spec["label"] == "phase4"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == 31
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase5_registry_declares_entity_attention_shapes() -> None:
    phase, spec = resolve_phase({"phase": 5})
    assert phase == 5
    assert spec["label"] == "phase5"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == ENTITY_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase6_registry_declares_entity_grid_shapes() -> None:
    phase, spec = resolve_phase({"phase": 6})
    assert phase == 6
    assert spec["label"] == "phase6"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase7_registry_declares_partial_obs_shapes() -> None:
    phase, spec = resolve_phase({"phase": 7})
    assert phase == 7
    assert spec["label"] == "phase7"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase8_registry_declares_random_map_shapes() -> None:
    phase, spec = resolve_phase({"phase": 8})
    assert phase == 8
    assert spec["label"] == "phase8"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase9_registry_declares_snapshot_shapes() -> None:
    phase, spec = resolve_phase({"phase": 9})
    assert phase == 9
    assert spec["label"] == "phase9"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase10_registry_declares_target_slot_shapes() -> None:
    phase, spec = resolve_phase({"phase": 10})
    assert phase == 10
    assert spec["label"] == "phase10"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == PHASE10_TARGET_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 7
    assert spec["target_action_dim"] == MULTI_ENEMY_TOKEN_COUNT


def test_phase11_registry_declares_current_selfplay_shapes() -> None:
    phase, spec = resolve_phase({"phase": 11})
    assert phase == 11
    assert spec["label"] == "phase11"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 6
    assert spec["action_dim"] == 6


def test_phase4_smoke_config_builds_mappo_config() -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.n_agents == 3
    assert cfg.obs_dim == 31
    assert cfg.critic_obs_dim == 135
    assert cfg.vector_env == "sync"


def test_phase4_config_can_select_async_vector_backend() -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    config["ppo"] = dict(config["ppo"])
    config["ppo"]["vector_env"] = "async"
    cfg = make_mappo_config(config)
    assert cfg.vector_env == "async"


def test_phase4_basic_config_builds_mappo_config() -> None:
    with open(
        "../experiments/configs/phase4_mappo_basic.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 8
    assert cfg.n_agents == 3
    assert cfg.rollout_len == 128
    assert cfg.critic_obs_dim == 135


def test_phase4_noop_probe_config_builds_mappo_config() -> None:
    with open(
        "../experiments/configs/phase4_mappo_noop_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 8
    assert cfg.n_agents == 3
    assert cfg.lr_schedule == "constant"
    assert cfg.entropy_coef == 0.001
    assert config["env"]["opponent_bot"] == "noop"
    assert config["env"]["reward"]["distance_shaping_coef"] == 0.05
    assert config["env"]["reward"]["on_point_shaping_coef"] == 0.02
    assert config["run"]["bc_pretrain_steps"] == 500


def test_phase4_objective_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase4_mappo_objective_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.gru_hidden == 32
    assert cfg.learning_rate == 1.0e-5
    assert config["run"]["bc_pretrain_steps"] == 200
    assert config["run"]["total_updates"] == 1


def test_phase5_entity_attention_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase5_entity_attention_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == ENTITY_OBS_DIM
    assert cfg.obs_encoder == "entity_attention"
    assert cfg.entity_token_count == ENTITY_TOKEN_COUNT
    assert cfg.entity_token_dim == ENTITY_TOKEN_DIM
    assert cfg.entity_num_heads == 2
    assert config["run"]["bc_pretrain_steps"] == 200
    assert config["run"]["total_updates"] == 1


def test_phase6_entity_grid_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase6_entity_grid_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.entity_token_count == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.obs_encoder == "entity_attention_grid"
    assert cfg.entity_token_count == ENTITY_TOKEN_COUNT
    assert cfg.entity_token_dim == ENTITY_TOKEN_DIM
    assert cfg.grid_channels == GRID_CHANNELS
    assert cfg.grid_size == GRID_SIZE
    assert config["run"]["bc_pretrain_steps"] == 200
    assert config["run"]["total_updates"] == 1


def test_phase7_team_fog_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase7_team_fog_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 7
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.entity_token_count == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.obs_encoder == "entity_attention_grid"
    assert ckpt_env_cfg["fog_mode"] == "team_shared"
    assert ckpt_env_cfg["visible_radius"] == 0.65
    assert config["run"]["bc_pretrain_steps"] == 500
    assert config["run"]["bc_batch_size"] == 900
    assert config["run"]["total_updates"] == 1
    assert config["run"]["eval_gate"]["min_episodes"] == 2
    assert config["run"]["eval_gate"]["min_win_rate"] == 1.0
    assert config["run"]["eval_gate"]["max_draw_rate"] == 0.0
    assert config["run"]["eval_gate"]["min_mean_reward"] == 10.0
    assert config["run"]["eval_gate"]["min_mean_score_a"] == 7.0
    assert config["run"]["eval_gate"]["max_mean_score_b"] == 0.0


def test_phase7_per_agent_fog_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase7_per_agent_fog_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 7
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.entity_token_count == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.obs_encoder == "entity_attention_grid"
    assert ckpt_env_cfg["fog_mode"] == "per_agent"
    assert ckpt_env_cfg["visible_radius"] == 0.65
    assert config["run"]["bc_pretrain_steps"] == 500
    assert config["run"]["bc_batch_size"] == 900
    assert config["run"]["total_updates"] == 1
    assert config["run"]["eval_gate"]["min_episodes"] == 2
    assert config["run"]["eval_gate"]["min_win_rate"] == 1.0
    assert config["run"]["eval_gate"]["max_draw_rate"] == 0.0
    assert config["run"]["eval_gate"]["min_mean_reward"] == 10.0
    assert config["run"]["eval_gate"]["min_mean_score_a"] == 7.0
    assert config["run"]["eval_gate"]["max_mean_score_b"] == 0.0


def test_phase8_random_map_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase8_random_map_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 8
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.entity_token_count == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.obs_encoder == "entity_attention_grid"
    assert ckpt_env_cfg["fog_mode"] == "team_shared"
    assert ckpt_env_cfg["map_randomization"]["span_jitter"] == 5.0
    assert ckpt_env_cfg["map_randomization"]["cover_count_per_side"] == 2
    assert ckpt_env_cfg["map_randomization"]["cover_jitter"] == 1.0
    assert ckpt_env_cfg["map_randomization"]["cover_radius"] == 1.0
    assert ckpt_env_cfg["sim"]["randomize_map"] is True
    assert config["run"]["bc_pretrain_steps"] == 500
    assert config["run"]["bc_batch_size"] == 900
    assert config["run"]["total_updates"] == 1


def test_phase9_snapshot_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase9_snapshot_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 9
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert cfg.entity_token_count == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.obs_encoder == "entity_attention_grid"
    assert ckpt_env_cfg["opponent_bot"] == "snapshot"
    assert len(ckpt_env_cfg["snapshot_paths"]) == 1
    assert ckpt_env_cfg["snapshot_league"]["weights"]["latest"] == 0.7
    assert ckpt_env_cfg["snapshot_league"]["weights"]["historical"] == 0.2
    assert ckpt_env_cfg["snapshot_league"]["weights"]["anchor"] == 0.1
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["current"] == 0.7
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["snapshot"] == 0.2
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["anchor"] == 0.1
    assert config["run"]["snapshot_retention"]["max_latest"] == 2
    assert config["run"]["snapshot_retention"]["preserve_best"] == 1
    assert config["run"]["matrix_eval"]["episodes"] == 1
    assert config["run"]["matrix_eval"]["anchor_bots"] == ["noop"]
    assert len(config["run"]["matrix_eval"]["opponent_checkpoints"]) == 1
    assert config["run"]["matrix_eval"]["gate"]["min_rows"] == 2
    assert config["run"]["matrix_eval"]["gate"]["min_win_rate"]["bot"] == 1.0
    assert config["run"]["bc_pretrain_steps"] == 500
    assert config["run"]["bc_batch_size"] == 900
    assert config["run"]["total_updates"] == 1


def test_phase10_target_slot_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase10_target_slot_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 10
    assert cfg.action_dim == 7
    assert cfg.target_action_dim == MULTI_ENEMY_TOKEN_COUNT
    assert cfg.num_envs == 2
    assert cfg.rollout_len == 32
    assert cfg.obs_dim == PHASE10_TARGET_OBS_DIM
    assert cfg.obs_encoder == "entity_attention_grid"
    assert ckpt_env_cfg["opponent_bot"] == "noop"
    assert ckpt_env_cfg["sim"]["hero_kinds"] == [
        "Vanguard",
        "Ranger",
        "Mender",
        "Vanguard",
        "Ranger",
        "Mender",
    ]
    assert ckpt_env_cfg["map_randomization"]["span_jitter"] == 5.0
    assert config["run"]["bc_pretrain_steps"] == 500
    assert config["run"]["bc_batch_size"] == 900
    assert config["run"]["total_updates"] == 1


def test_phase11_mixed_league_probe_config_is_compact() -> None:
    with open(
        "../experiments/configs/phase11_mixed_league_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    assert phase == 11
    assert cfg.n_agents == 6
    assert cfg.value_per_agent is True
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["current"] == 0.34
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["snapshot"] == 0.33
    assert ckpt_env_cfg["self_play_schedule"]["weights"]["anchor"] == 0.33
    assert ckpt_env_cfg["snapshot_league"]["weights"]["latest"] == 0.7
    assert config["run"]["matrix_eval"]["current_selfplay"] is True
    assert config["run"]["matrix_eval"]["anchor_bots"] == ["noop"]
    assert len(config["run"]["matrix_eval"]["opponent_checkpoints"]) == 1
    assert config["run"]["total_updates"] == 1


def test_phase4_walk_bc_target_points_toward_objective() -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    obs = torch.tensor(
        [
            [0.0] * 31,
            [0.0] * 31,
        ],
        dtype=torch.float32,
    )
    obs[0, 5:7] = torch.tensor([0.0, -0.8])
    obs[1, 5:7] = torch.tensor([0.6, 0.0])
    target = _walk_to_objective_targets(obs, cfg)
    assert target[0, 0].item() == 0.0
    assert target[0, 1].item() == 1.0
    assert target[1, 0].item() == -1.0
    assert target[1, 1].item() == 0.0


def test_phase4_mappo_smoke_train_runs_one_update(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4")
    result = train_phase4_from_config(config)
    assert set(result) == {"mappo"}
    assert (tmp_path / "phase4" / "mappo" / "ckpt_final.pt").exists()


def test_phase4_mappo_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase4_mappo_objective_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase4_objective")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase4_objective" / "mappo" / "ckpt_final.pt").exists()


def test_phase5_entity_attention_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase5_entity_attention_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase5_entity")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase5_entity" / "mappo" / "ckpt_final.pt").exists()


def test_phase6_entity_grid_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase6_entity_grid_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase6_grid")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase6_grid" / "mappo" / "ckpt_final.pt").exists()


def test_phase7_team_fog_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase7_team_fog_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase7_fog")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase7_fog" / "mappo" / "ckpt_final.pt").exists()


def test_phase7_per_agent_fog_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase7_per_agent_fog_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase7_per_agent_fog")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase7_per_agent_fog" / "mappo" / "ckpt_final.pt").exists()


def test_phase8_random_map_bc_eval_can_be_best_result(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase8_random_map_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["output_dir"] = str(tmp_path / "phase8_random_map")
    result = train_phase4_from_config(config)
    assert result["mappo"] > 10.0
    assert (tmp_path / "phase8_random_map" / "mappo" / "ckpt_final.pt").exists()


def test_phase4_mappo_eval_reports_diagnostics() -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml", "r", encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    assert phase == 4
    env_fn, _ckpt_env_cfg, seed_base = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    trainer = MappoTrainer(env_fn, cfg, seed=seed_base)
    try:
        stats = evaluate_mappo(trainer.model, env_fn, episodes=1, seed=seed_base + 1)
    finally:
        trainer.close()
    assert stats.episodes == 1
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.terminated + stats.truncated == 1
    assert stats.mean_final_tick > 0
