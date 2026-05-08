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
