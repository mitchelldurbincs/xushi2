"""Warm-start (init-from-checkpoint) tests for the recurrent PPO orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from train import ppo_recurrent as ppo_mod
from train.models import build_model
from train.ppo_recurrent import EvaluationStats
from train.ppo_recurrent import orchestration as ppo_orch


def _save_phase2_checkpoint(path: Path, *, obs_dim: int = 3) -> dict[str, Any]:
    """Build and save a real ActorCritic checkpoint for warm-start tests."""
    model_cfg = {
        "obs_dim": obs_dim,
        "action_dim": 2,
        "continuous_action_dim": 2,
        "binary_action_dim": 0,
        "use_recurrence": True,
        "embed_dim": 8,
        "gru_hidden": 8,
        "head_hidden": 8,
        "action_log_std_init": -1.0,
    }
    model = build_model(**model_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"phase": 2, "model": model_cfg},
        },
        path,
    )
    return model_cfg


def test_load_init_checkpoint_round_trips_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "src.pt"
    saved_cfg = _save_phase2_checkpoint(ckpt)

    target = build_model(**saved_cfg)
    # Perturb target weights so the load actually changes something.
    with torch.no_grad():
        for p in target.parameters():
            p.add_(1.0)

    ppo_orch._load_init_checkpoint(target, ckpt, saved_cfg)

    expected_state = torch.load(ckpt, weights_only=False)["model_state_dict"]
    actual_state = target.state_dict()
    assert set(expected_state.keys()) == set(actual_state.keys())
    for key, exp in expected_state.items():
        assert torch.equal(actual_state[key], exp), f"{key} mismatch after load"


def test_load_init_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "src.pt"
    _save_phase2_checkpoint(ckpt, obs_dim=3)

    mismatched_cfg = {
        "obs_dim": 31,  # Phase 3 obs_dim, intentionally different.
        "action_dim": 2,
        "continuous_action_dim": 2,
        "binary_action_dim": 0,
        "use_recurrence": True,
        "embed_dim": 8,
        "gru_hidden": 8,
        "head_hidden": 8,
        "action_log_std_init": -1.0,
    }
    target = build_model(**mismatched_cfg)

    with pytest.raises(ValueError, match="obs_dim"):
        ppo_orch._load_init_checkpoint(target, ckpt, mismatched_cfg)


def test_load_init_checkpoint_rejects_missing_topology_keys(tmp_path: Path) -> None:
    ckpt = tmp_path / "src_missing.pt"
    saved_cfg = _save_phase2_checkpoint(ckpt)
    state = torch.load(ckpt, weights_only=False)
    del state["config"]["model"]["gru_hidden"]
    torch.save(state, ckpt)

    target = build_model(**saved_cfg)
    with pytest.raises(ValueError, match="missing topology keys"):
        ppo_orch._load_init_checkpoint(target, ckpt, saved_cfg)


def test_load_init_checkpoint_accepts_legacy_config_without_schema_version(
    tmp_path: Path,
) -> None:
    ckpt = tmp_path / "legacy.pt"
    saved_cfg = _save_phase2_checkpoint(ckpt)
    state = torch.load(ckpt, weights_only=False)
    # Simulate old format: no explicit schema_version.
    state["config"].pop("schema_version", None)
    torch.save(state, ckpt)

    target = build_model(**saved_cfg)
    ppo_orch._load_init_checkpoint(target, ckpt, saved_cfg)


# --- Orchestration wiring tests (mirror test_phase2_early_stop.py structure).


class _DummyModel:
    def __init__(self) -> None:
        self.update_idx = 0

    def state_dict(self) -> dict[str, int]:
        return {"update_idx": self.update_idx}


class _DummyTrainer:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.model = _DummyModel()

    def set_learning_rate(self, *_: Any, **__: Any) -> None:
        return None

    def collect_rollout(self) -> None:
        return None

    def update(self, _rollout: None) -> dict[str, float]:
        self.model.update_idx += 1
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "trunk_grad_norm": 0.0,
            "terminal_adv_std": 0.0,
            "mean_log_std": 0.0,
            "lr": 0.0,
        }


def _base_phase2_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "env": {"episode_length": 8, "cue_visible_ticks": 4, "seed_base": 123},
        "model": {
            "embed_dim": 8,
            "gru_hidden": 8,
            "head_hidden": 8,
            "action_log_std_init": -1.0,
        },
        "ppo": {
            "num_envs": 1,
            "rollout_len": 4,
            "num_epochs": 1,
            "minibatch_size": 1,
            "learning_rate": 3e-4,
            "clip_ratio": 0.2,
            "value_clip_ratio": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "run": {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1000,
            "log_every": 0,
            "output_dir": str(tmp_path / "runs"),
        },
    }


def _eval_stats(mean_reward: float) -> EvaluationStats:
    return EvaluationStats(
        mean_reward=mean_reward,
        episodes=1,
        wins=0,
        losses=0,
        draws=0,
        terminated=0,
        truncated=0,
        mean_final_tick=0.0,
        mean_team_a_score=0.0,
        mean_team_b_score=0.0,
        mean_team_a_kills=0.0,
        mean_team_b_kills=0.0,
    )


def test_run_variant_invokes_warm_start_when_configured(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _base_phase2_config(tmp_path)
    cfg["run"]["init_from_checkpoint"] = str(tmp_path / "init.pt")

    calls: list[tuple[Any, str, dict[str, Any]]] = []

    def _fake_load(model: Any, path: Any, expected_model_cfg: dict[str, Any]) -> None:
        calls.append((model, str(path), dict(expected_model_cfg)))

    monkeypatch.setattr(ppo_orch, "PPOTrainer", _DummyTrainer)
    monkeypatch.setattr(ppo_orch, "_load_init_checkpoint", _fake_load)
    monkeypatch.setattr(
        ppo_orch, "evaluate_policy_stats", lambda *a, **k: _eval_stats(0.1)
    )

    ppo_mod._run_variant(cfg, use_recurrence=True, output_dir=tmp_path / "out")

    assert len(calls) == 1
    _, ckpt_arg, expected_cfg_arg = calls[0]
    assert ckpt_arg == str(tmp_path / "init.pt")
    # The expected_model_cfg passed to the loader must reflect the new
    # config's topology (obs_dim/action_dim resolved from the phase task spec).
    assert expected_cfg_arg["embed_dim"] == 8
    assert expected_cfg_arg["use_recurrence"] is True


def test_run_variant_skips_warm_start_when_unset(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = _base_phase2_config(tmp_path)
    # No "init_from_checkpoint" key.

    calls: list[Any] = []

    def _fake_load(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(ppo_orch, "PPOTrainer", _DummyTrainer)
    monkeypatch.setattr(ppo_orch, "_load_init_checkpoint", _fake_load)
    monkeypatch.setattr(
        ppo_orch, "evaluate_policy_stats", lambda *a, **k: _eval_stats(0.1)
    )

    ppo_mod._run_variant(cfg, use_recurrence=True, output_dir=tmp_path / "out")

    assert calls == []
