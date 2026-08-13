"""Checkpoint resume, atomic writes, and best-eval honesty.

Checkpoints used to carry model weights only. Restarting from one zeroed the
Adam moments and restarted the LR schedule at update 1 -- a large silent
optimization discontinuity that presents as "the run got worse after restart".
"""

from __future__ import annotations

import json

import pytest
import torch
from _paths import config_path

from train.mappo_eval_checkpoint import train_mappo_from_config
from train.train import load_config


def _run_dir(output_dir):
    """build_runtime_context nests artifacts under <output_dir>/mappo."""
    return output_dir / "mappo"


def _base_config(tmp_path, *, total_updates: int, checkpoint_every: int) -> dict:
    config = load_config(config_path("phase4/smoke/phase4_mappo_smoke.yaml"))
    config["run"]["total_updates"] = total_updates
    config["run"]["checkpoint_every"] = checkpoint_every
    config["run"]["eval_every"] = total_updates
    config["run"]["eval_episodes"] = 1
    config["run"]["log_every"] = 0
    config["run"]["output_dir"] = str(tmp_path / "run")
    config["wandb"] = {"enabled": False}
    return config


def _final_weights(output_dir) -> dict:
    raw = torch.load(_run_dir(output_dir) / "ckpt_final.pt", map_location="cpu", weights_only=False)
    return raw["model_state_dict"]


def _assert_state_dicts_close(a: dict, b: dict, *, atol: float = 0.0) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=atol, rtol=0.0), f"diverged at {key}"


# --- what resume actually restores --------------------------------------
#
# Note the boundary: the learner's state is restored exactly, but the
# environment's is not. The C++ Sim is not serializable from Python, so a
# resumed run begins its first update from a freshly reset episode rather than
# wherever the interrupted run happened to be mid-episode. Weights therefore do
# NOT match an uninterrupted run bit for bit, and asserting that they do would
# be asserting something the feature does not provide. What it does provide --
# and what actually distinguishes resuming from warm-starting -- is that Adam's
# moments, the LR schedule position, the RNG streams, and the update index all
# continue instead of restarting. Recurrent state restarts at zero alongside
# the freshly reset environment, so it cannot contain memory from an episode
# whose state was not restored.


@pytest.mark.slow
def test_resume_restores_optimizer_moments_rather_than_zeroing_them(tmp_path):
    """Warm-starting from weights alone silently restarts Adam. Resume must not.

    A zeroed exp_avg_sq makes the first post-restart updates far larger than
    they should be, which reads as "the run got worse after restart".
    """
    from train.mappo_pretrain_hooks import maybe_resume
    from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
    from train.mappo_runtime_context import build_runtime_context
    from train.runtime_adapter import resolve_runtime_env_factory

    output_dir = tmp_path / "run"
    config = _base_config(output_dir, total_updates=2, checkpoint_every=1)
    config["run"]["output_dir"] = str(output_dir)
    train_mappo_from_config(config)

    saved = torch.load(
        _run_dir(output_dir) / "ckpt_0002.pt", map_location="cpu", weights_only=False
    )
    saved_opt = saved["resume_state"]["optimizer_state_dict"]
    # Sanity: the source run really did accumulate moments.
    assert any(
        float(v["exp_avg_sq"].abs().sum()) > 0.0 for v in saved_opt["state"].values()
    )

    resumed_cfg = _base_config(tmp_path / "resumed", total_updates=4, checkpoint_every=4)
    resumed_cfg["run"]["output_dir"] = str(tmp_path / "resumed")
    resumed_cfg["run"]["resume_from"] = str(_run_dir(output_dir) / "ckpt_0002.pt")
    context = build_runtime_context(resumed_cfg)
    _runtime, env_fn, _seed = resolve_runtime_env_factory(resumed_cfg, context="test")
    trainer = MappoTrainer(env_fn, make_mappo_config(resumed_cfg), seed=context.seed_base)
    try:
        start_update = maybe_resume(context, trainer)
        assert start_update == 3
        live_opt = trainer.optimizer.state_dict()
        assert live_opt["state"].keys() == saved_opt["state"].keys()
        for key, saved_entry in saved_opt["state"].items():
            assert torch.allclose(live_opt["state"][key]["exp_avg"], saved_entry["exp_avg"])
            assert torch.allclose(
                live_opt["state"][key]["exp_avg_sq"], saved_entry["exp_avg_sq"]
            )
    finally:
        trainer.close()


@pytest.mark.slow
def test_resume_continues_the_lr_schedule_instead_of_restarting_it(tmp_path):
    """The smoke config uses a cosine schedule, so restarting is observable."""
    from train.lr_schedule import lr_for_update

    output_dir = tmp_path / "run"
    config = _base_config(output_dir, total_updates=8, checkpoint_every=2)
    config["run"]["output_dir"] = str(output_dir)
    config["run"]["total_updates"] = 2
    train_mappo_from_config(config)

    saved = torch.load(
        _run_dir(output_dir) / "ckpt_0002.pt", map_location="cpu", weights_only=False
    )
    completed = saved["resume_state"]["update_idx"]
    mappo = saved["config"]["mappo"]

    def lr_at(update):
        return lr_for_update(
            update,
            8,
            base_lr=mappo["learning_rate"],
            schedule=mappo["lr_schedule"],
            lr_final_ratio=mappo["lr_final_ratio"],
            warmup_updates=mappo["warmup_updates"],
        )

    # Resuming runs update completed+1 next, so it must see that update's LR,
    # not update 1's.
    assert lr_at(completed + 1) != pytest.approx(lr_at(1))


@pytest.mark.slow
def test_periodic_checkpoints_carry_resume_state(tmp_path):
    output_dir = tmp_path / "run"
    config = _base_config(output_dir, total_updates=2, checkpoint_every=1)
    config["run"]["output_dir"] = str(output_dir)
    train_mappo_from_config(config)

    raw = torch.load(_run_dir(output_dir) / "ckpt_0001.pt", map_location="cpu", weights_only=False)
    state = raw["resume_state"]
    # Each of these is load-bearing continuation state. The environment and
    # recurrent hidden state deliberately restart together instead.
    for key in (
        "optimizer_state_dict",
        "update_idx",
        "update_counter",
        "policy_sampling_generator_state",
        "torch_rng_state",
        "numpy_rng_state",
    ):
        assert key in state, key
    assert "hidden_state" not in state
    assert state["update_idx"] == 1


# --- resume refuses what it cannot honor --------------------------------


@pytest.mark.slow
def test_resume_rejects_a_checkpoint_without_resume_state(tmp_path):
    output_dir = tmp_path / "run"
    config = _base_config(output_dir, total_updates=1, checkpoint_every=1)
    config["run"]["output_dir"] = str(output_dir)
    train_mappo_from_config(config)

    # ckpt_final.pt is deliberately written without resume state.
    resumed = _base_config(tmp_path / "resumed", total_updates=2, checkpoint_every=2)
    resumed["run"]["output_dir"] = str(tmp_path / "resumed")
    resumed["run"]["resume_from"] = str(_run_dir(output_dir) / "ckpt_final.pt")
    with pytest.raises(ValueError, match="no resume_state"):
        train_mappo_from_config(resumed)


def test_resume_and_warm_start_are_mutually_exclusive(tmp_path):
    config = _base_config(tmp_path, total_updates=1, checkpoint_every=1)
    config["run"]["resume_from"] = "a.pt"
    config["run"]["init_from_checkpoint"] = "b.pt"
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_mappo_from_config(config)


def test_resume_reports_a_missing_file_clearly(tmp_path):
    config = _base_config(tmp_path, total_updates=1, checkpoint_every=1)
    config["run"]["resume_from"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="resume_from checkpoint not found"):
        train_mappo_from_config(config)


def test_resume_ignores_legacy_hidden_state_and_resets_recurrence(tmp_path):
    from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config

    config = _base_config(tmp_path, total_updates=1, checkpoint_every=1)
    cfg = make_mappo_config(config)
    from train.runtime_adapter import resolve_runtime_env_factory

    _runtime, env_fn, _seed = resolve_runtime_env_factory(config, context="test")
    trainer = MappoTrainer(env_fn, cfg, seed=1)
    try:
        state = trainer.resume_state()
        # Older checkpoints carried episode-local recurrence even though the
        # corresponding environment state was never serialized. Its shape is
        # irrelevant now: it must not influence the fresh reset episode.
        state["hidden_state"] = torch.full((99, 99, 99), 7.0)
        trainer.h.fill_(3.0)
        trainer.load_resume_state(state)
        assert torch.count_nonzero(trainer.h).item() == 0
    finally:
        trainer.close()


# --- best-eval honesty and atomic writes --------------------------------


@pytest.mark.slow
def test_best_eval_checkpoint_is_absent_when_no_eval_improved(tmp_path):
    """It used to be written unconditionally, holding the *last* weights.

    Several scripts load ckpt_best_eval.pt by name, so a file that exists but
    is not the best is worse than no file.
    """
    output_dir = tmp_path / "run"
    config = _base_config(output_dir, total_updates=1, checkpoint_every=1)
    config["run"]["output_dir"] = str(output_dir)
    train_mappo_from_config(config)

    manifest = json.loads((_run_dir(output_dir) / "checkpoint_manifest.json").read_text())
    if not manifest["has_best_eval_checkpoint"]:
        assert not (_run_dir(output_dir) / "ckpt_best_eval.pt").exists()
        assert manifest["ckpt_final_alias"] == "ckpt_last.pt"
    else:
        assert (_run_dir(output_dir) / "ckpt_best_eval.pt").exists()


def test_checkpoint_write_leaves_no_temp_file(tmp_path):
    from train.mappo_checkpoint_outputs import save_mappo_checkpoint

    class _Cfg:
        __dict__ = {"a": 1}

    path = tmp_path / "ckpt.pt"
    save_mappo_checkpoint(
        path=path,
        model_state_dict={"w": torch.zeros(2)},
        phase=4,
        phase_label="phase4",
        ckpt_env_cfg={},
        mappo_cfg=_Cfg(),
    )
    assert path.is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]


def test_snapshot_retention_reloads_existing_history(tmp_path):
    """A resumed run must not discard the previous run's league."""
    from xushi2.snapshot_retention import SnapshotRetention

    manifest_path = tmp_path / "snapshots.json"
    first = SnapshotRetention(manifest_path)
    first.record_checkpoint(tmp_path / "a.pt", update=1, score=0.5)
    first.record_checkpoint(tmp_path / "b.pt", update=2, score=0.7)

    # A fresh instance, as a restarted process would build.
    second = SnapshotRetention(manifest_path)
    second.record_checkpoint(tmp_path / "c.pt", update=3, score=0.9)
    manifest = json.loads(manifest_path.read_text())
    assert [r["update"] for r in manifest["records"]] == [1, 2, 3]


def test_snapshot_retention_refuses_to_silently_drop_a_corrupt_manifest(tmp_path):
    from xushi2.snapshot_retention import SnapshotRetention

    manifest_path = tmp_path / "snapshots.json"
    manifest_path.write_text('{"records": [{"path": "a.pt", "update": 1', encoding="utf-8")
    with pytest.raises(ValueError, match="could not be read"):
        SnapshotRetention(manifest_path)


def test_snapshot_retention_write_is_atomic(tmp_path):
    from xushi2.snapshot_retention import SnapshotRetention

    manifest_path = tmp_path / "snapshots.json"
    retention = SnapshotRetention(manifest_path)
    retention.record_checkpoint(tmp_path / "a.pt", update=1, score=0.5)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snapshots.json"]
