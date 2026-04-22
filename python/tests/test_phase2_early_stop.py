"""Phase-2 training early-stop behavior tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from train import ppo_recurrent as ppo_mod


class _DummyModel:
    def __init__(self) -> None:
        self.update_idx = 0

    def state_dict(self) -> dict[str, int]:
        return {"update_idx": self.update_idx}


class _DummyTrainer:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.model = _DummyModel()

    def collect_rollout(self) -> None:
        return None

    def update(self, _rollout: None) -> dict[str, float]:
        self.model.update_idx += 1
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }


def _base_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "env": {
            "episode_length": 8,
            "cue_visible_ticks": 4,
            "seed_base": 123,
        },
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
            "total_updates": 20,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1000,
            "log_every": 0,
            "output_dir": str(tmp_path / "runs"),
        },
    }


def test_early_stop_from_stagnation_keeps_best_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    cfg = _base_config(tmp_path)
    cfg["run"].update({
        "early_stop_patience_evals": 2,
        "early_stop_min_delta": 0.01,
        "max_regression_from_best": 1.0,
    })

    evals = iter([0.10, 0.105, 0.106, 0.106])
    monkeypatch.setattr(ppo_mod, "PPOTrainer", _DummyTrainer)
    monkeypatch.setattr(ppo_mod, "evaluate_policy", lambda *args, **kwargs: next(evals))

    out_dir = tmp_path / "stagnation"
    best_eval = ppo_mod._run_variant(cfg, use_recurrence=True, output_dir=out_dir)

    ckpt = torch.load(out_dir / "ckpt_final.pt")
    assert best_eval == 0.10
    assert ckpt["model_state_dict"]["update_idx"] == 1
    assert "early-stop: eval improvement stagnated past patience" in capsys.readouterr().out


def test_early_stop_from_regression_keeps_best_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    cfg = _base_config(tmp_path)
    cfg["run"].update({
        "early_stop_patience_evals": 100,
        "early_stop_min_delta": 0.0,
        "max_regression_from_best": 0.05,
    })

    evals = iter([0.50, 0.20])
    monkeypatch.setattr(ppo_mod, "PPOTrainer", _DummyTrainer)
    monkeypatch.setattr(ppo_mod, "evaluate_policy", lambda *args, **kwargs: next(evals))

    out_dir = tmp_path / "regression"
    best_eval = ppo_mod._run_variant(cfg, use_recurrence=False, output_dir=out_dir)

    ckpt = torch.load(out_dir / "ckpt_final.pt")
    assert best_eval == 0.50
    assert ckpt["model_state_dict"]["update_idx"] == 1
    assert "early-stop: eval regression exceeded max_regression_from_best" in capsys.readouterr().out
