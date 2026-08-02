"""Regression tests for the 2026-08-02 campaign review fixes: transfer
summary snapshot inclusion + gate semantics, log_std optimizer ownership,
snapshot RNG seed flow, and async propagation of the new opponent setters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from train.mappo_post_training import transfer_rows_and_gate
from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.snapshot_policy import SnapshotPolicy
from xushi2.vector_env import XushiAsyncVectorEnv

_CKPT = Path(__file__).resolve().parents[2] / (
    "data/checkpoints/phase4_v5_upd300_stochastic_600t_converter.pt"
)


def _sim_cfg() -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 3,
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


def _tiny_config(**ppo_overrides) -> dict:
    ppo = {
        "num_envs": 1,
        "rollout_len": 4,
        "num_epochs": 1,
        "minibatch_size": 4,
        "learning_rate": 1.0e-4,
        "value_normalization": True,
        "vector_env": "sync",
        "torch_num_threads": 1,
        "clip_ratio": 0.2,
        "value_clip_ratio": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    ppo.update(ppo_overrides)
    return {
        "phase": 4,
        "env": {
            "seed_base": 0,
            "opponent_bot": "noop",
            "learner_team": "A",
            "sim": _sim_cfg(),
        },
        "model": {
            "use_recurrence": True,
            "embed_dim": 16,
            "gru_hidden": 8,
            "head_hidden": 16,
            "action_log_std_init": -1.0,
        },
        "ppo": ppo,
        "run": {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "log_every": 1,
            "output_dir": "runs/test_review_fixes",
        },
    }


def _make_trainer(**ppo_overrides) -> MappoTrainer:
    config = _tiny_config(**ppo_overrides)
    cfg = make_mappo_config(config)
    return MappoTrainer(
        lambda: Phase4MappoEnv(
            config["env"]["sim"],
            opponent_bot="noop",
            learner_team="A",
            reward_cfg={},
        ),
        cfg,
        seed=0,
    )


# --- transfer summary: snapshot inclusion + gate semantics ---------------


def _row(opponent: str, opponent_type: str, cap_gain: float = 0.0) -> dict:
    return {
        "opponent": opponent,
        "opponent_type": opponent_type,
        "mean_cap_progress_gain_ticks": cap_gain,
    }


def test_transfer_rows_include_listed_snapshots():
    rows = [
        _row("weak_basic_v2", "bot"),
        _row("champ.pt", "snapshot"),
        _row("unlisted.pt", "snapshot"),
    ]
    selected, _status = transfer_rows_and_gate(rows, ("weak_basic_v2", "champ.pt"))
    assert [r["opponent"] for r in selected] == ["weak_basic_v2", "champ.pt"]


def test_transfer_gate_ungated_without_noop():
    rows = [_row("weak_basic_v2", "bot")]
    _selected, status = transfer_rows_and_gate(rows, ("weak_basic_v2",))
    assert status == "ungated"


def test_transfer_gate_noop_semantics_preserved():
    assert transfer_rows_and_gate([_row("noop", "bot", 5.0)], ("noop",))[1] == "pass"
    assert (
        transfer_rows_and_gate([_row("noop", "bot", 0.0)], ("noop",))[1]
        == "evidence_insufficient"
    )


# --- log_std optimizer ownership -----------------------------------------


def test_log_std_excluded_from_optimizer_when_anneal_active():
    trainer = _make_trainer(log_std_anneal_updates=10, log_std_final_offset=-1.0)
    optim_params = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert id(trainer.model.log_std) not in optim_params
    trainer.vec_env.close()


def test_log_std_in_optimizer_by_default():
    trainer = _make_trainer()
    optim_params = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert id(trainer.model.log_std) in optim_params
    trainer.vec_env.close()


# --- snapshot RNG seed flow ----------------------------------------------


@pytest.mark.skipif(not _CKPT.exists(), reason="reference checkpoint not present")
def test_snapshot_sampling_varies_with_env_seed():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="noop")
    env.reset(seed=3)
    a = SnapshotPolicy(_CKPT, stochastic=True)
    b = SnapshotPolicy(_CKPT, stochastic=True)
    c = SnapshotPolicy(_CKPT, stochastic=True)
    a.reset(seed=111)
    b.reset(seed=222)
    c.reset(seed=111)
    act_a = a.act(env._sim, [3, 4, 5])
    act_b = b.act(env._sim, [3, 4, 5])
    act_c = c.act(env._sim, [3, 4, 5])
    env.close()
    assert not np.allclose(act_a, act_b)
    assert np.allclose(act_a, act_c)


# --- async propagation of the new setters --------------------------------


def _env_fn():
    return Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="hold_and_shoot")


def test_async_vector_env_propagates_new_setters():
    vec = XushiAsyncVectorEnv([_env_fn, _env_fn], critic_obs_dim=CRITIC_DIM)
    try:
        vec.set_opponent_bots(["weak_basic_v2", "noop"])
        vec.set_opponent_handicap("weak_basic_v2", 1.0, 30)
        vec.reset(seed=5)
        obs, *_rest = vec.step(np.zeros((2, 3, 6), dtype=np.float32))
        assert np.isfinite(obs).all()
    finally:
        vec.close()
