"""Test that the MAPPO trainer warm-starts from a checkpoint when
``run.init_from_checkpoint`` is set in the config.

Used by the Phase 4 cap-training escalation to seed a basic-opponent run
from a noop-trained policy. See
``docs/plans/2026-05-08-phase4-cap-training-escalation.md`` Task 8.
"""

from __future__ import annotations

from pathlib import Path

import torch

from train.mappo import train_phase4_from_config


def _phase4_smoke_cfg(output_dir: Path, **run_overrides) -> dict:
    """Minimal Phase 4 config that runs a few updates on a small env so
    the test completes quickly."""
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
        "ppo": {
            "num_envs": 2,
            "rollout_len": 16,
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
        "run": {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": str(output_dir),
            **run_overrides,
        },
    }


def _set_seed(cfg: dict, seed: int) -> dict:
    cfg["env"]["seed_base"] = seed
    cfg["env"]["sim"]["seed"] = seed
    return cfg


def test_mappo_warm_starts_from_init_checkpoint(tmp_path: Path) -> None:
    """With different seeds for stage 1 vs stage 2, warm-started stage 2
    should still land near stage 1's weights — proving the load path
    overrode the random init."""
    # Stage 1: seed=0, train one update, save checkpoint.
    stage1_dir = tmp_path / "stage1"
    train_phase4_from_config(_set_seed(_phase4_smoke_cfg(stage1_dir), 0))
    ckpt_path = stage1_dir / "mappo" / "ckpt_final.pt"
    assert ckpt_path.exists(), "stage 1 must produce a checkpoint"

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sample_param_name = "actor_embed.0.weight"
    expected_value = raw_ckpt["model_state_dict"][sample_param_name].clone()

    # Stage 2: seed=12345, warm-start from stage 1. Without warm-start,
    # this would init the model with completely different random weights
    # (different seed) — divergence would be large. With warm-start,
    # weights should match stage 1 modulo one PPO update of drift.
    stage2_dir = tmp_path / "stage2"
    cfg2 = _set_seed(
        _phase4_smoke_cfg(stage2_dir, init_from_checkpoint=str(ckpt_path)),
        12345,
    )
    train_phase4_from_config(cfg2)

    stage2_ckpt = torch.load(
        stage2_dir / "mappo" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    actual = stage2_ckpt["model_state_dict"][sample_param_name]
    diff = (actual - expected_value).abs().mean().item()
    assert diff < 0.05, (
        f"warm-started stage2 weights diverged too far from stage1 "
        f"(diff={diff:.4f}); expected close-to-zero — warm-start failed "
        f"to override the differently-seeded random init?"
    )


def test_mappo_without_warm_start_with_different_seed_diverges(
    tmp_path: Path,
) -> None:
    """Negative control for the warm-start test above: without
    ``init_from_checkpoint``, a seed=0 run and seed=12345 run should have
    very different weights. If this test fails, the warm-start positive
    test is invalid (would pass even without the load path)."""
    seed0_dir = tmp_path / "seed0"
    train_phase4_from_config(_set_seed(_phase4_smoke_cfg(seed0_dir), 0))
    seed0_ckpt = torch.load(
        seed0_dir / "mappo" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    seed_other_dir = tmp_path / "seed12345"
    train_phase4_from_config(_set_seed(_phase4_smoke_cfg(seed_other_dir), 12345))
    seed_other_ckpt = torch.load(
        seed_other_dir / "mappo" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=False,
    )

    sample = "actor_embed.0.weight"
    diff = (
        (seed0_ckpt["model_state_dict"][sample] - seed_other_ckpt["model_state_dict"][sample])
        .abs()
        .mean()
        .item()
    )
    # Two different seeds should produce noticeably different weights.
    # If this is < 0.05, the positive test's threshold is too loose.
    assert diff > 0.05, (
        f"different-seed runs should diverge but didn't (diff={diff:.4f}); "
        f"loosen the warm-start positive test threshold"
    )
