from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import pytest
import yaml
from _paths import config_path, script_path

from train.mappo import (
    MappoActorCritic,
    make_mappo_config,
    train_phase4_from_config,
)
from train.mappo_model import _eval_outcome_counts
from train.phases import resolve_phase


def _load_config(relative_path: str) -> dict:
    with open(config_path(relative_path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_checkpoint(
    path: Path,
    config_relative_path: str,
    *,
    phase: int,
    mutate_config: Callable[[dict], None] | None = None,
) -> Path:
    config = _load_config(config_relative_path)
    if mutate_config is not None:
        mutate_config(config)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, env_cfg, _seed = spec["env_bundle"](config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 8,
                "env": env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        path,
    )
    return path


def _run_matrix_cli(
    checkpoint: Path,
    output: Path,
    *,
    anchor_bots: Sequence[str] = (),
    opponent_checkpoints: Sequence[Path] = (),
    episodes: int = 1,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script_path("eval_mappo_matrix.py")),
        "--checkpoint",
        str(checkpoint),
    ]
    for bot in anchor_bots:
        command.extend(["--anchor-bot", str(bot)])
    for opponent_checkpoint in opponent_checkpoints:
        command.extend(["--opponent-checkpoint", str(opponent_checkpoint)])
    command.extend(["--episodes", str(episodes), "--output", str(output)])
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _load_matrix_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase4_smoke_matrix_config(tmp_path: Path, output_name: str) -> dict:
    config = _load_config("phase4/smoke/phase4_mappo_smoke.yaml")
    run_cfg = dict(config["run"])
    run_cfg.update(
        {
            "total_updates": 1,
            "eval_every": 1,
            "eval_episodes": 1,
            "checkpoint_every": 1,
            "output_dir": str(tmp_path / output_name),
            "matrix_eval": {
                "episodes": 1,
                "anchor_bots": ["noop"],
                "output": "matrix_eval.json",
            },
        }
    )
    config["run"] = run_cfg
    return config


def test_eval_outcome_counts_current_selfplay_decisive_games_as_draws() -> None:
    assert _eval_outcome_counts(
        winner="A",
        learner_team="both",
        truncated=False,
    ) == (0, 0, 1)
    assert _eval_outcome_counts(
        winner="B",
        learner_team="both",
        truncated=False,
    ) == (0, 0, 1)


def test_eval_mappo_matrix_writes_bot_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase4_multi_enemy.pt"
    _write_checkpoint(
        checkpoint,
        "phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
        phase=8,
    )
    output = tmp_path / "matrix.json"

    result = _run_matrix_cli(
        checkpoint,
        output,
        anchor_bots=("noop",),
    )

    assert "opponent=bot:noop" in result.stdout
    rows = _load_matrix_rows(output)
    assert len(rows) == 1
    assert {row["opponent_type"] for row in rows} == {"bot"}
    for row in rows:
        assert row["episodes"] == 1
        assert 0.0 <= row["win_rate"] <= 1.0
        assert 0.0 <= row["draw_rate"] <= 1.0
        assert isinstance(row["mean_reward"], float)


def test_pinned_matrix_width_is_independent_of_host_cpu_count(tmp_path, monkeypatch):
    import os

    from scripts.eval_mappo_matrix import evaluate_matrix

    checkpoint = _write_checkpoint(
        tmp_path / "reference.pt",
        "phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
        phase=4,
        mutate_config=lambda config: config["env"]["sim"].update(round_length_seconds=1),
    )
    kwargs = dict(
        anchor_bots=["weak_basic_v2"],
        opponent_checkpoints=[str(checkpoint)],
        episodes=2,
        num_envs=2,
        seed=0xA11CE,
        stochastic=True,
        stochastic_snapshot=True,
        canonical=False,
    )
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    first = evaluate_matrix([str(checkpoint)], **kwargs)
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    second = evaluate_matrix([str(checkpoint)], **kwargs)
    assert first == second
    assert len(first) == 2
    assert all(row["episodes"] == 2 for row in first)


@pytest.mark.parametrize("learner_team,canonical", [("A", False), ("B", True)])
def test_sampled_matrix_replays_preserve_metrics_and_reconstruct_every_hash(
    tmp_path, learner_team, canonical,
):
    from scripts.analyze_replay_combat import _config_from_header, _load_replay
    from scripts.eval_mappo_matrix import evaluate_matrix
    from xushi2 import xushi2_cpp as cpp

    def configure(config):
        sim = config["env"]["sim"]
        sim["round_length_seconds"] = 2
        sim["mechanics"]["respawn_ticks"] = 600
        sim["map"] = {"min_x": 0.0, "min_y": 0.0, "max_x": 47.123456, "max_y": 49.0}
        sim["cover_circles"] = [{"x": 8.1234567, "y": 30.0, "radius": 0.81234567}]
        sim["wall_segments"] = [{"x1": 30.123456, "y1": 8.0, "x2": 33.0,
                                  "y2": 9.0, "half_width": 0.21234567}]

    checkpoint = _write_checkpoint(
        tmp_path / "reference.pt",
        "phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml",
        phase=4, mutate_config=configure,
    )
    kwargs = dict(anchor_bots=["weak_basic_v2"], opponent_checkpoints=[str(checkpoint)],
                  episodes=3, num_envs=2, seed=0xA11CE, stochastic=True,
                  stochastic_snapshot=True, canonical=canonical, learner_team=learner_team)
    plain = evaluate_matrix([str(checkpoint)], **kwargs)
    directory = tmp_path / "replays"
    captured = evaluate_matrix([str(checkpoint)], replay_dir=directory, **kwargs)
    assert captured == plain
    for cell, row in zip(sorted(directory.iterdir()), captured, strict=True):
        index = json.loads((cell / "index.json").read_text())
        assert index["completed_episodes"] == 3
        assert len(list(cell.glob("*.replay"))) == 3  # discard uncounted fourth completion
        assert [ep["vector_lane"] for ep in index["episodes"]] == [0, 1, 0]
        assert [ep["lane_episode"] for ep in index["episodes"]] == [0, 0, 1]
        scores = []
        for episode in index["episodes"]:
            header, decisions = _load_replay(cell / episode["replay"])
            cfg = _config_from_header(header)
            assert cfg.seed == index["seed"] + episode["vector_lane"] + 10_000 * episode["lane_episode"]
            assert cfg.mechanics.respawn_ticks == (240 if canonical else 600)
            assert cfg.objective_unlock_ticks == round(row["objective_unlock_seconds"] * cpp.TICK_HZ)
            assert cfg.objective_capture_ticks == round(row["objective_capture_seconds"] * cpp.TICK_HZ)
            sim = cpp.Sim(cfg)
            assert f"0x{sim.state_hash:016x}" == episode["initial_state_hash"]
            sidecar = json.loads((cell / episode["sidecar"]).read_text())
            for decision, (tick, expected_hash) in zip(decisions, sidecar["state_hashes"], strict=True):
                assert sim.tick == decision.tick
                sim.step_decision(decision.actions)
                assert sim.tick == tick
                assert f"0x{sim.state_hash:016x}" == expected_hash
            assert sim.episode_over
            assert sim.team_a_score == episode["final"]["team_a_score"]
            assert sim.team_b_score == episode["final"]["team_b_score"]
            scores.append(sim.team_a_score)
        assert sum(scores) / len(scores) == row["mean_score_a"]
    with pytest.raises(FileExistsError):
        evaluate_matrix([str(checkpoint)], replay_dir=directory, **kwargs)


def test_eval_mappo_matrix_adapts_phase4_current_selfplay_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "phase4_selfplay.pt"
    _write_checkpoint(
        checkpoint,
        "phase4/probe/phase4_mappo_current_selfplay_smoke.yaml",
        phase=4,
    )
    output = tmp_path / "matrix.json"

    result = _run_matrix_cli(
        checkpoint,
        output,
        anchor_bots=("noop",),
    )

    assert "opponent=bot:noop" in result.stdout
    rows = _load_matrix_rows(output)
    assert len(rows) == 1
    assert rows[0]["opponent_type"] == "bot"
    assert rows[0]["opponent"] == "noop"
    assert rows[0]["episodes"] == 1
    assert isinstance(rows[0]["mean_score_a"], float)


def test_train_config_matrix_eval_writes_post_training_artifact(
    tmp_path: Path,
) -> None:
    config = _phase4_smoke_matrix_config(tmp_path, "phase4_matrix")

    result = train_phase4_from_config(config)
    assert "mappo" in result
    output = tmp_path / "phase4_matrix" / "mappo" / "matrix_eval.json"
    rows = _load_matrix_rows(output)
    assert len(rows) == 1
    assert rows[0]["opponent_type"] == "bot"
    assert rows[0]["opponent"] == "noop"
    assert rows[0]["episodes"] == 1
    transfer_json = tmp_path / "phase4_matrix" / "mappo" / "transfer_summary.json"
    transfer_md = tmp_path / "phase4_matrix" / "mappo" / "transfer_summary.md"
    transfer_payload = json.loads(transfer_json.read_text(encoding="utf-8"))
    assert transfer_payload["gate_status"] in {"pass", "evidence_insufficient"}
    assert len(transfer_payload["rows"]) == 1
    assert transfer_md.exists()


def test_train_config_transfer_gate_can_fail_on_insufficient_evidence(tmp_path: Path) -> None:
    # The noop evidence gate applies when a noop row is requested.
    config = _phase4_smoke_matrix_config(tmp_path, "phase4_matrix_fail")
    config["run"]["matrix_eval"]["anchor_bots"] = ["noop"]
    config["run"]["matrix_eval"]["transfer_bots"] = ["noop"]
    config["run"]["matrix_eval"]["transfer_fail_on_insufficient"] = True
    try:
        train_phase4_from_config(config)
    except RuntimeError as exc:
        assert "transfer gate evidence insufficient" in str(exc)
    else:
        raise AssertionError("expected transfer gate failure")


def test_train_config_transfer_gate_ungated_without_noop(tmp_path: Path) -> None:
    # Matrices without a noop row are "ungated", not "evidence_insufficient"
    # (2026-08-02 review: ladder/self-play summaries were permanently
    # mislabeled), and the fail flag must not fire.
    config = _phase4_smoke_matrix_config(tmp_path, "phase4_matrix_ungated")
    config["run"]["matrix_eval"]["anchor_bots"] = ["basic"]
    config["run"]["matrix_eval"]["transfer_bots"] = ["basic"]
    config["run"]["matrix_eval"]["transfer_fail_on_insufficient"] = True
    train_phase4_from_config(config)
    transfer_md = tmp_path / "phase4_matrix_ungated" / "mappo" / "transfer_summary.md"
    assert "ungated" in transfer_md.read_text(encoding="utf-8")


def test_matrix_eval_updates_snapshot_retention_manifest(tmp_path: Path) -> None:
    config = _phase4_smoke_matrix_config(tmp_path, "phase4_matrix_retention")
    config["run"]["snapshot_retention"] = {
        "manifest": "snapshot_league.json",
        "max_latest": 2,
        "preserve_best": 1,
        "include_config_anchors": False,
    }

    train_phase4_from_config(config)

    manifest_path = tmp_path / "phase4_matrix_retention" / "mappo" / "snapshot_league.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_record = next(r for r in manifest["records"] if Path(r["path"]).name == "ckpt_final.pt")
    assert final_record["matrix_rows"] == 1
    assert "matrix_score" in final_record
    assert Path(manifest["historical"][0]).name == "ckpt_final.pt"
