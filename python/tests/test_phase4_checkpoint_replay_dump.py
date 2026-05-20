from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import yaml
from _paths import config_path, script_path

from train.mappo import MappoActorCritic, make_mappo_config
from train.phases import resolve_phase


def test_dump_replay_supports_phase4_mappo_checkpoint(tmp_path: Path) -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_objective_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase4_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 4,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase4_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=4" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase4_current_selfplay_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    with torch.no_grad():
        model.actor_mean_head.bias[0] = torch.atanh(torch.tensor(0.5))
        model.actor_mean_head.bias[1] = torch.atanh(torch.tensor(-0.25))
        model.actor_mean_head.bias[2] = torch.atanh(torch.tensor(0.1))
        model.actor_binary_head.bias.fill_(2.0)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase4_selfplay_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 4,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase4_selfplay_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=4" in lines[0]
    assert "match_type=current" in lines[0]
    assert "loss_mask=1,1,1,1,1,1" in lines[0]
    fields = [float(v) for v in lines[1].split()]
    assert len(fields) == 1 + 6 * 6
    slot0 = fields[1:7]
    slot3 = fields[1 + 3 * 6 : 1 + 4 * 6]
    assert slot0[0] > 0.0
    assert slot0[1] < 0.0
    assert slot3[0] < 0.0
    assert slot3[1] > 0.0


def test_dump_replay_supports_phase5_entity_attention_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase5_entity_attention_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase5_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 5,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase5_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=5" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase6_entity_grid_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase6_entity_grid_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase6_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 6,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase6_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=6" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase7_team_fog_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase7_team_fog_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase7_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 7,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase7_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=7" in lines[0]
    assert "fog=1" in lines[0]
    assert "last_seen=1" in lines[0]
    assert "fog_mode=team_shared" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase7_per_agent_fog_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase7_per_agent_fog_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase7b_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 7,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase7b_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=7" in lines[0]
    assert "fog=1" in lines[0]
    assert "last_seen=1" in lines[0]
    assert "fog_mode=per_agent" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase8_random_map_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase8_random_map_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase8_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 8,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase8_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=8" in lines[0]
    assert "map_min_x=" in lines[0]
    assert "map_max_y=" in lines[0]
    assert "layout=0x" in lines[0]
    assert "cover=" in lines[0]
    assert "walls=" in lines[0]
    assert "fog=1" in lines[0]
    assert "team_size=3" in lines[0]
    assert "loss_mask=1,1,1" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase9_snapshot_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase8_random_map_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        snapshot_config = yaml.safe_load(fh)
    snapshot_cfg = make_mappo_config(snapshot_config)
    snapshot_model = MappoActorCritic(snapshot_cfg)
    _phase, spec = resolve_phase(snapshot_config)
    _env_fn, snapshot_env_cfg, _seed = spec["env_bundle"](snapshot_config)
    snapshot_path = tmp_path / "snapshot.pt"
    torch.save(
        {
            "model_state_dict": snapshot_model.state_dict(),
            "config": {
                "phase": 8,
                "env": snapshot_env_cfg,
                "mappo": snapshot_cfg.__dict__,
            },
        },
        snapshot_path,
    )

    with open(
        config_path("phase9_snapshot_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["env"] = dict(config["env"])
    config["env"]["snapshot_paths"] = [str(snapshot_path)]
    config["env"]["snapshot_league"] = {
        "latest": [str(snapshot_path)],
        "historical": [str(snapshot_path)],
        "anchor": [str(snapshot_path)],
        "weights": {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
    }
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase9_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 9,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase9_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=9" in lines[0]
    assert "map_min_x=" in lines[0]
    assert "schedule=current:0.7,snapshot:0.2,anchor:0.1" in lines[0]
    assert "league=latest:0.7:1,historical:0.2:1,anchor:0.1:1" in lines[0]
    assert "snapshot_group=" in lines[0]
    assert "snapshot=snapshot.pt" in lines[0]
    assert "fog=1" in lines[0]
    assert "team_size=3" in lines[0]
    assert "loss_mask=1,1,1" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase10_target_slot_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase10_target_slot_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase10_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 10,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase10_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=10" in lines[0]
    assert "target_slot=1" in lines[0]
    assert "heroes=vanguard,ranger,mender,vanguard,ranger,mender" in lines[0]
    assert "map_min_x=" in lines[0]
    assert "fog=1" in lines[0]
    assert "team_size=3" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 7


def test_dump_replay_supports_phase11_current_selfplay_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase11/probe/phase11_current_selfplay_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)

    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase11_mappo.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 11,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase11_policy.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=11" in lines[0]
    assert "match_type=current" in lines[0]
    assert "team_size=3" in lines[0]
    assert "loss_mask=1,1,1,1,1,1" in lines[0]
    assert "fog=1" in lines[0]
    assert "target_slot=1" not in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6


def test_dump_replay_supports_phase11_mixed_snapshot_checkpoint(
    tmp_path: Path,
) -> None:
    with open(
        config_path("phase8_random_map_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        snapshot_config = yaml.safe_load(fh)
    snapshot_cfg = make_mappo_config(snapshot_config)
    snapshot_model = MappoActorCritic(snapshot_cfg)
    _phase, snapshot_spec = resolve_phase(snapshot_config)
    _env_fn, snapshot_env_cfg, _seed = snapshot_spec["env_bundle"](snapshot_config)
    snapshot_path = tmp_path / "phase8_snapshot.pt"
    torch.save(
        {
            "model_state_dict": snapshot_model.state_dict(),
            "config": {
                "phase": 8,
                "env": snapshot_env_cfg,
                "mappo": snapshot_cfg.__dict__,
            },
        },
        snapshot_path,
    )

    with open(
        config_path("phase11/probe/phase11_mixed_league_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    config["env"]["snapshot_league"] = {
        "latest": [str(snapshot_path)],
        "historical": [str(snapshot_path)],
        "weights": {"latest": 0.7, "historical": 0.3},
    }
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / "phase11_mixed.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 11,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )

    replay_path = tmp_path / "phase11_mixed.replay"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path("dump_replay.py")),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(replay_path),
            "--seed",
            "1",
            "--max-decisions",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[dump_replay] wrote 3 decisions" in result.stdout
    lines = replay_path.read_text(encoding="ascii").splitlines()
    assert "phase=11" in lines[0]
    assert "schedule=current:0.34,snapshot:0.33,anchor:0.33" in lines[0]
    assert "match_type=snapshot" in lines[0]
    assert "snapshot_group=historical" in lines[0]
    assert "loss_mask=1,1,1,0,0,0" in lines[0]
    assert len(lines) == 4
    assert len(lines[1].split()) == 1 + 6 * 6
