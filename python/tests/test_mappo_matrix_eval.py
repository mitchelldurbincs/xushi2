from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from train.mappo import MappoActorCritic, make_mappo_config
from train.mappo import train_phase4_from_config
from train.phases import resolve_phase


def _write_phase8_checkpoint(path: Path) -> None:
    with open(
        "../experiments/configs/phase8_random_map_probe.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
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


def test_eval_mappo_matrix_writes_bot_and_snapshot_rows(tmp_path: Path) -> None:
    checkpoint = tmp_path / "phase8.pt"
    _write_phase8_checkpoint(checkpoint)
    output = tmp_path / "matrix.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_mappo_matrix.py",
            "--checkpoint",
            str(checkpoint),
            "--anchor-bot",
            "noop",
            "--opponent-checkpoint",
            str(checkpoint),
            "--episodes",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "opponent=bot:noop" in result.stdout
    assert "opponent=snapshot:phase8.pt" in result.stdout
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {row["opponent_type"] for row in rows} == {"bot", "snapshot"}
    for row in rows:
        assert row["episodes"] == 1
        assert 0.0 <= row["win_rate"] <= 1.0
        assert 0.0 <= row["draw_rate"] <= 1.0
        assert isinstance(row["mean_reward"], float)


def test_train_config_matrix_eval_writes_post_training_artifact(
    tmp_path: Path,
) -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4_matrix")
    config["run"]["matrix_eval"] = {
        "episodes": 1,
        "anchor_bots": ["noop"],
        "output": "matrix_eval.json",
    }

    result = train_phase4_from_config(config)
    assert "mappo" in result
    output = tmp_path / "phase4_matrix" / "mappo" / "matrix_eval.json"
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["opponent_type"] == "bot"
    assert rows[0]["opponent"] == "noop"
    assert rows[0]["episodes"] == 1


def test_matrix_eval_updates_snapshot_retention_manifest(tmp_path: Path) -> None:
    with open(
        "../experiments/configs/phase4_mappo_smoke.yaml",
        "r",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4_matrix_retention")
    config["run"]["snapshot_retention"] = {
        "manifest": "snapshot_league.json",
        "max_latest": 2,
        "preserve_best": 1,
        "include_config_anchors": False,
    }
    config["run"]["matrix_eval"] = {
        "episodes": 1,
        "anchor_bots": ["noop"],
        "output": "matrix_eval.json",
    }

    train_phase4_from_config(config)

    manifest_path = (
        tmp_path
        / "phase4_matrix_retention"
        / "mappo"
        / "snapshot_league.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_record = next(
        r for r in manifest["records"] if Path(r["path"]).name == "ckpt_final.pt"
    )
    assert final_record["matrix_rows"] == 1
    assert "matrix_score" in final_record
    assert Path(manifest["historical"][0]).name == "ckpt_final.pt"
