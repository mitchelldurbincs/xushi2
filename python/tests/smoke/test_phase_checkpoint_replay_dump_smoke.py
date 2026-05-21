from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch
import yaml
from tests._paths import config_path, script_path

from train.mappo import MappoActorCritic, make_mappo_config
from train.phases import resolve_phase

pytestmark = pytest.mark.smoke_behavior

_MAX_DECISIONS = 3
_SIX_SLOT_REPLAY_FIELDS = 1 + 6 * 6


def _load_config(relative_path: str) -> dict:
    with open(config_path(relative_path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_checkpoint(
    tmp_path: Path,
    *,
    filename: str,
    config: dict,
    phase: int,
    mutate_model: Callable[[MappoActorCritic], None] | None = None,
) -> Path:
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    if mutate_model is not None:
        mutate_model(model)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)

    checkpoint_path = tmp_path / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": phase,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _dump_replay(
    *,
    checkpoint_path: Path,
    replay_path: Path,
    seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path("dump_replay.py")),
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(replay_path),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    command.extend(["--max-decisions", str(_MAX_DECISIONS)])

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"[dump_replay] wrote {_MAX_DECISIONS} decisions" in result.stdout
    return replay_path.read_text(encoding="ascii").splitlines()


def _assert_replay_shape(lines: Sequence[str], *, fields: int = _SIX_SLOT_REPLAY_FIELDS) -> None:
    assert len(lines) == _MAX_DECISIONS + 1
    assert len(lines[1].split()) == fields


def _assert_header_contains(lines: Sequence[str], expected: Sequence[str]) -> None:
    for fragment in expected:
        assert fragment in lines[0]


@pytest.mark.parametrize(
    ("config_relpath", "phase", "checkpoint_name", "expected_header", "fields"),
    [
        (
            "phase4/probe/phase4_mappo_objective_probe.yaml",
            4,
            "phase4_mappo.pt",
            ("phase=4", "team_size=3"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase5_entity_attention_probe.yaml",
            5,
            "phase5_mappo.pt",
            ("phase=5", "team_size=3"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase6_entity_grid_probe.yaml",
            6,
            "phase6_mappo.pt",
            ("phase=6", "team_size=3"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase7_team_fog_probe.yaml",
            7,
            "phase7_mappo.pt",
            ("phase=7", "fog=1", "last_seen=1", "fog_mode=team_shared", "team_size=3"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase7_per_agent_fog_probe.yaml",
            7,
            "phase7b_mappo.pt",
            ("phase=7", "fog=1", "last_seen=1", "fog_mode=per_agent", "team_size=3"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase8_random_map_probe.yaml",
            8,
            "phase8_mappo.pt",
            (
                "phase=8",
                "map_min_x=",
                "map_max_y=",
                "layout=0x",
                "cover=",
                "walls=",
                "fog=1",
                "team_size=3",
                "loss_mask=1,1,1",
            ),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
        (
            "phase10_target_slot_probe.yaml",
            10,
            "phase10_mappo.pt",
            (
                "phase=10",
                "target_slot=1",
                "heroes=vanguard,ranger,mender,vanguard,ranger,mender",
                "map_min_x=",
                "fog=1",
                "team_size=3",
            ),
            1 + 6 * 7,
        ),
        (
            "phase11/probe/phase11_current_selfplay_probe.yaml",
            11,
            "phase11_mappo.pt",
            ("phase=11", "match_type=current", "team_size=3", "loss_mask=1,1,1,1,1,1", "fog=1"),
            _SIX_SLOT_REPLAY_FIELDS,
        ),
    ],
)
def test_dump_replay_supports_phase_checkpoint(
    tmp_path: Path,
    config_relpath: str,
    phase: int,
    checkpoint_name: str,
    expected_header: tuple[str, ...],
    fields: int,
) -> None:
    checkpoint_path = _write_checkpoint(
        tmp_path,
        filename=checkpoint_name,
        config=_load_config(config_relpath),
        phase=phase,
    )
    lines = _dump_replay(
        checkpoint_path=checkpoint_path,
        replay_path=tmp_path / checkpoint_name.replace(".pt", ".replay"),
    )

    _assert_header_contains(lines, expected_header)
    if phase == 11:
        assert "target_slot=1" not in lines[0]
    _assert_replay_shape(lines, fields=fields)


def test_dump_replay_supports_phase4_current_selfplay_checkpoint(
    tmp_path: Path,
) -> None:
    def bias_current_policy(model: MappoActorCritic) -> None:
        with torch.no_grad():
            model.actor_mean_head.bias[0] = torch.atanh(torch.tensor(0.5))
            model.actor_mean_head.bias[1] = torch.atanh(torch.tensor(-0.25))
            model.actor_mean_head.bias[2] = torch.atanh(torch.tensor(0.1))
            model.actor_binary_head.bias.fill_(2.0)

    checkpoint_path = _write_checkpoint(
        tmp_path,
        filename="phase4_selfplay_mappo.pt",
        config=_load_config("phase4/probe/phase4_mappo_current_selfplay_smoke.yaml"),
        phase=4,
        mutate_model=bias_current_policy,
    )
    lines = _dump_replay(
        checkpoint_path=checkpoint_path,
        replay_path=tmp_path / "phase4_selfplay_policy.replay",
    )

    _assert_header_contains(
        lines,
        ("phase=4", "match_type=current", "loss_mask=1,1,1,1,1,1"),
    )
    fields = [float(v) for v in lines[1].split()]
    assert len(fields) == _SIX_SLOT_REPLAY_FIELDS
    slot0 = fields[1:7]
    slot3 = fields[1 + 3 * 6 : 1 + 4 * 6]
    assert slot0[0] > 0.0
    assert slot0[1] < 0.0
    assert slot3[0] < 0.0
    assert slot3[1] > 0.0


def test_dump_replay_supports_phase9_snapshot_checkpoint(
    tmp_path: Path,
) -> None:
    snapshot_path = _write_checkpoint(
        tmp_path,
        filename="snapshot.pt",
        config=_load_config("phase8_random_map_probe.yaml"),
        phase=8,
    )

    config = _load_config("phase9_snapshot_probe.yaml")
    config["env"] = dict(config["env"])
    config["env"]["snapshot_paths"] = [str(snapshot_path)]
    config["env"]["snapshot_league"] = {
        "latest": [str(snapshot_path)],
        "historical": [str(snapshot_path)],
        "anchor": [str(snapshot_path)],
        "weights": {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
    }
    checkpoint_path = _write_checkpoint(
        tmp_path,
        filename="phase9_mappo.pt",
        config=config,
        phase=9,
    )
    lines = _dump_replay(
        checkpoint_path=checkpoint_path,
        replay_path=tmp_path / "phase9_policy.replay",
    )

    _assert_header_contains(
        lines,
        (
            "phase=9",
            "map_min_x=",
            "schedule=current:0.7,snapshot:0.2,anchor:0.1",
            "league=latest:0.7:1,historical:0.2:1,anchor:0.1:1",
            "snapshot_group=",
            "snapshot=snapshot.pt",
            "fog=1",
            "team_size=3",
            "loss_mask=1,1,1",
        ),
    )
    _assert_replay_shape(lines)


def test_dump_replay_supports_phase11_mixed_snapshot_checkpoint(
    tmp_path: Path,
) -> None:
    snapshot_path = _write_checkpoint(
        tmp_path,
        filename="phase8_snapshot.pt",
        config=_load_config("phase8_random_map_probe.yaml"),
        phase=8,
    )

    config = _load_config("phase11/probe/phase11_mixed_league_probe.yaml")
    config["env"] = dict(config["env"])
    config["env"]["snapshot_league"] = {
        "latest": [str(snapshot_path)],
        "historical": [str(snapshot_path)],
        "weights": {"latest": 0.7, "historical": 0.3},
    }
    checkpoint_path = _write_checkpoint(
        tmp_path,
        filename="phase11_mixed.pt",
        config=config,
        phase=11,
    )
    lines = _dump_replay(
        checkpoint_path=checkpoint_path,
        replay_path=tmp_path / "phase11_mixed.replay",
        seed=1,
    )

    _assert_header_contains(
        lines,
        (
            "phase=11",
            "schedule=current:0.34,snapshot:0.33,anchor:0.33",
            "match_type=snapshot",
            "snapshot_group=historical",
            "loss_mask=1,1,1,0,0,0",
        ),
    )
    _assert_replay_shape(lines)
