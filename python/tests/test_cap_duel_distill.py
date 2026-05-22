from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from train.cap_duel_distill import (
    build_cap_duel_distill_anchor,
    configure_cap_duel_distill_anchor,
    run_cap_duel_distill_diagnostics,
)
from train.mappo_model import MappoActorCritic
from train.mappo_rollout_trainer import MappoTrainer
from train.mappo_runtime_context import build_runtime_context
from train.train import load_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config(relative: str) -> dict:
    return load_config(_repo_root() / "experiments/configs" / relative)


def _small_phase4_config(tmp_path: Path, *, enabled: bool) -> dict:
    cfg = _load_config("phase4/smoke/phase4_mappo_smoke.yaml")
    cfg["wandb"] = {"enabled": False}
    cfg["ppo"]["num_envs"] = 1
    cfg["ppo"]["rollout_len"] = 2
    cfg["ppo"]["num_epochs"] = 1
    cfg["run"]["total_updates"] = 1
    cfg["run"]["eval_every"] = 1
    cfg["run"]["eval_episodes"] = 1
    cfg["run"]["checkpoint_every"] = 1
    cfg["run"]["output_dir"] = str(tmp_path / ("enabled" if enabled else "disabled"))
    cfg["run"]["cap_duel_distill"] = {"enabled": enabled}
    return cfg


def _write_teacher_checkpoint(path: Path, cfg) -> Path:
    teacher = MappoActorCritic(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": {"mappo": asdict(cfg)}, "model_state_dict": teacher.state_dict()},
        path,
    )
    return path


def _small_distill_cfg(teacher_path: Path) -> dict:
    return {
        "enabled": True,
        "teacher_checkpoint": str(teacher_path),
        "env": {
            "mini_game": "cap_duel",
            "mini_game_config": {
                "episode_decisions": 4,
                "enemy_hp": 1,
                "point_radius": 0.18,
                "score_ticks_to_clear": 2,
                "enemy_recontest_delay": 2,
                "hit_tolerance": 0.12,
                "hit_reward": 1.0,
                "kill_bonus": 4.0,
                "score_per_tick": 0.1,
                "off_point_penalty": 0.0,
                "time_penalty_per_decision": 0.0,
                "knockback_magnitude": 0.0,
                "spawn_distance": 0.4,
                "respawn_at_spawn_position": True,
            },
        },
        "batch_size": 2,
        "every_updates": 1,
        "coef": 0.05,
        "aim_coef": 1.0,
        "fire_coef": 1.0,
        "move_coef": 0.0,
    }


def test_config_defaults_keep_cap_duel_distill_disabled() -> None:
    cfg = _load_config("phase4/smoke/phase4_mappo_smoke.yaml")

    assert cfg["run"]["cap_duel_distill"] == {"enabled": False}


def test_cap_duel_distill_disabled_path_adds_no_metrics(tmp_path: Path) -> None:
    config = _small_phase4_config(tmp_path, enabled=False)
    context = build_runtime_context(config)
    trainer = MappoTrainer(context.env_fn, context.cfg, seed=context.seed_base)
    try:
        assert configure_cap_duel_distill_anchor(context, trainer) is False
        trainer.set_update_index(1)
        metrics = trainer.update(trainer.collect_rollout())
    finally:
        trainer.close()

    assert not any(key.startswith("distill/") for key in metrics)


def test_enabled_anchor_loads_frozen_teacher_and_ppo_update_logs_distill_metrics(
    tmp_path: Path,
) -> None:
    config = _small_phase4_config(tmp_path, enabled=True)
    context = build_runtime_context(config)
    teacher_path = _write_teacher_checkpoint(tmp_path / "teacher.pt", context.cfg)
    config["run"]["cap_duel_distill"] = _small_distill_cfg(teacher_path)
    context = build_runtime_context(config)
    trainer = MappoTrainer(context.env_fn, context.cfg, seed=context.seed_base)
    try:
        assert configure_cap_duel_distill_anchor(context, trainer) is True
        assert trainer.cap_duel_distill_anchor is not None
        teacher = trainer.cap_duel_distill_anchor.teacher
        assert all(not param.requires_grad for param in teacher.parameters())

        trainer.set_update_index(1)
        metrics = trainer.update(trainer.collect_rollout())
        assert metrics["distill/updates"] == 1.0
        assert np.isfinite(metrics["distill/loss"])
        assert np.isfinite(metrics["distill/aim_loss"])
        assert np.isfinite(metrics["distill/fire_loss"])
        assert metrics["distill/active_samples"] >= 2.0
        assert all(param.grad is None for param in teacher.parameters())
    finally:
        trainer.close()


def test_cap_duel_distill_diagnostics_emit_required_fields(tmp_path: Path) -> None:
    config = _small_phase4_config(tmp_path, enabled=True)
    context = build_runtime_context(config)
    teacher_path = _write_teacher_checkpoint(tmp_path / "teacher.pt", context.cfg)
    distill_cfg = _small_distill_cfg(teacher_path)
    anchor = build_cap_duel_distill_anchor(
        base_env_cfg=context.ckpt_env_cfg,
        student_cfg=context.cfg,
        distill_cfg=distill_cfg,
        seed=context.seed_base,
    )
    assert anchor is not None
    model = MappoActorCritic(context.cfg)

    diagnostics = run_cap_duel_distill_diagnostics(
        model,
        anchor=anchor,
        objective_env_fn=context.env_fn,
        full_env_fn=context.env_fn,
        episodes=1,
        seed=context.seed_base,
    )

    expected = {
        "objective_on_point",
        "objective_losses",
        "cap_duel_kills",
        "full_hit_fire",
        "full_aim_error",
        "aim_mse",
        "fire_bce",
        "teacher_fire_prob",
        "student_fire_prob",
    }
    assert expected.issubset(diagnostics)
    assert all(np.isfinite(float(value)) for value in diagnostics.values())


def test_probe_config_mirrors_cap_duel_v2_block_and_enables_anchor() -> None:
    root = _repo_root()
    probe = yaml.safe_load(
        (
            root
            / "experiments/configs/phase4/probe/"
            "phase4_mappo_cap_duel_distill_anchor_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    cap_duel_v2 = yaml.safe_load(
        (
            root
            / "experiments/configs/phase4/probe/"
            "phase4_mappo_cap_duel_selfplay_v2.yaml"
        ).read_text(encoding="utf-8")
    )

    distill = probe["run"]["cap_duel_distill"]
    assert distill["enabled"] is True
    assert distill["teacher_checkpoint"] == (
        "runs/phase4_mappo_cap_duel_selfplay_v2/mappo/ckpt_final.pt"
    )
    assert probe["run"]["composition_pretrain"] is False
    assert probe["run"]["bc_pretrain_steps"] == 0
    assert distill["env"]["mini_game"] == "cap_duel"
    assert distill["env"]["mini_game_config"] == cap_duel_v2["env"]["mini_game_config"]
