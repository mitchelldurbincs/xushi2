from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from envs.phase9_snapshot_mappo import Phase9SnapshotMappoEnv
from train.mappo import MappoActorCritic, make_mappo_config
from train.phases import resolve_phase
from xushi2 import xushi2_cpp as _cpp
from xushi2.grid_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.map_randomization import map_layout_hash
from xushi2.obs_manifest import CRITIC_DIM
from xushi2.runner import _build_config
from xushi2.self_play_schedule import SelfPlaySchedule
from xushi2.snapshot_policy import SnapshotLeague, SnapshotPolicy, SnapshotPool
from xushi2.snapshot_retention import SnapshotRetention


def _write_snapshot(path: Path) -> None:
    with open(
        "../experiments/configs/phase8_random_map_probe.yaml",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "phase": 8,
                "env": ckpt_env_cfg,
                "mappo": cfg.__dict__,
            },
        },
        path,
    )


def test_snapshot_pool_sampling_is_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    pool = SnapshotPool(paths)
    assert pool.sample_path(123) == pool.sample_path(123)
    assert pool.sample_path(123) in {str(paths[0]), str(paths[1])}


def test_snapshot_league_sampling_is_deterministic_and_weighted(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    historical = tmp_path / "historical.pt"
    anchor = tmp_path / "anchor.pt"
    league = SnapshotLeague(
        {
            "latest": [latest],
            "historical": [historical],
            "anchor": [anchor],
        },
        {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
    )
    a = league.sample(123)
    b = league.sample(123)
    assert a == b
    assert a.group in {"latest", "historical", "anchor"}
    assert a.path in {str(latest), str(historical), str(anchor)}
    assert league.summary == "latest:0.7:1,historical:0.2:1,anchor:0.1:1"


def test_snapshot_retention_caps_latest_and_preserves_best(tmp_path: Path) -> None:
    retention = SnapshotRetention(
        tmp_path / "snapshot_league.json",
        max_latest=2,
        preserve_best=1,
        anchor_paths=[tmp_path / "anchor.pt"],
    )
    retention.record_checkpoint(tmp_path / "ckpt_0001.pt", update=1, score=1.0)
    retention.record_checkpoint(tmp_path / "ckpt_0002.pt", update=2, score=3.0)
    manifest = retention.record_checkpoint(tmp_path / "ckpt_0003.pt", update=3, score=2.0)

    assert manifest["latest"] == [
        str(tmp_path / "ckpt_0002.pt"),
        str(tmp_path / "ckpt_0003.pt"),
    ]
    assert manifest["historical"] == [str(tmp_path / "ckpt_0002.pt")]
    assert manifest["anchor"] == [str(tmp_path / "anchor.pt")]
    loaded = yaml.safe_load((tmp_path / "snapshot_league.json").read_text())
    league = SnapshotLeague.from_config((), loaded)
    assert league.summary == "latest:0.7:2,historical:0.2:1,anchor:0.1:1"


def test_snapshot_retention_prefers_matrix_passing_records(tmp_path: Path) -> None:
    retention = SnapshotRetention(
        tmp_path / "snapshot_league.json",
        max_latest=3,
        preserve_best=2,
    )
    retention.record_checkpoint(
        tmp_path / "score_only.pt",
        update=1,
        score=10.0,
        matrix_score=0.0,
        matrix_gate_passed=False,
        matrix_rows=2,
    )
    retention.record_checkpoint(
        tmp_path / "matrix_pass.pt",
        update=2,
        score=1.0,
        matrix_score=0.5,
        matrix_gate_passed=True,
        matrix_rows=2,
    )
    manifest = retention.record_checkpoint(
        tmp_path / "matrix_better.pt",
        update=3,
        score=2.0,
        matrix_score=0.9,
        matrix_gate_passed=True,
        matrix_rows=3,
    )

    assert manifest["historical"] == [
        str(tmp_path / "matrix_better.pt"),
        str(tmp_path / "matrix_pass.pt"),
    ]
    records = {Path(r["path"]).name: r for r in manifest["records"]}
    assert records["matrix_better.pt"]["matrix_gate_passed"] is True
    assert records["matrix_better.pt"]["matrix_rows"] == 3


def test_self_play_schedule_samples_current_snapshot_and_anchor(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    historical = tmp_path / "historical.pt"
    anchor = tmp_path / "anchor.pt"
    schedule = SelfPlaySchedule(
        weights={"current": 1.0, "snapshot": 0.0, "anchor": 0.0},
        latest=[latest],
        historical=[historical],
        anchor=[anchor],
    )
    assert schedule.sample(1).match_type == "current"
    schedule = SelfPlaySchedule(
        weights={"current": 0.0, "snapshot": 1.0, "anchor": 0.0},
        latest=[latest],
        historical=[historical],
        anchor=[anchor],
    )
    snapshot = schedule.sample(1)
    assert snapshot.match_type == "snapshot"
    assert snapshot.group in {"latest", "historical"}
    assert snapshot.snapshot_path in {str(latest), str(historical)}
    schedule = SelfPlaySchedule(
        weights={"current": 0.0, "snapshot": 0.0, "anchor": 1.0},
        anchor_bot="noop",
    )
    anchor_sample = schedule.sample(1)
    assert anchor_sample.match_type == "anchor"
    assert anchor_sample.anchor_bot == "noop"
    assert schedule.summary == "anchor:1"


def test_snapshot_policy_loads_and_emits_actions(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.pt"
    _write_snapshot(snapshot_path)
    policy = SnapshotPolicy(snapshot_path)
    assert policy.phase == 8
    assert policy.cfg.obs_encoder == "entity_attention_grid"


def test_snapshot_policy_uses_live_map_bounds_for_randomized_obs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot_path = tmp_path / "snapshot.pt"
    _write_snapshot(snapshot_path)
    policy = SnapshotPolicy(snapshot_path)
    live_bounds = {
        "min_x": -11.0,
        "min_y": -13.0,
        "max_x": 61.0,
        "max_y": 67.0,
    }
    captured_multi: list[dict[str, float]] = []
    captured_norm: list[dict[str, float]] = []

    def fake_multi_enemy_obs(flat, **kwargs):
        captured_multi.append(dict(kwargs["map_bounds"]))
        return np.zeros((flat.shape[0], policy.cfg.obs_dim), dtype=np.float32)

    def fake_normalize_world_for_team(world_xy, map_bounds, *, team_b_view):
        captured_norm.append(dict(map_bounds))
        return np.zeros(2, dtype=np.float32)

    monkeypatch.setattr(
        "xushi2.snapshot_policy.actor_obs_to_multi_enemy_entity_grid_obs",
        fake_multi_enemy_obs,
    )
    monkeypatch.setattr(
        "xushi2.snapshot_policy.normalize_world_for_team",
        fake_normalize_world_for_team,
    )
    with open(
        "../experiments/configs/phase8_random_map_probe.yaml",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = _build_config(config["env"]["sim"], seed_override=123)
    cfg.team_size = 3
    sim = _cpp.Sim(cfg)

    policy.act(sim, (3, 4, 5), map_bounds=live_bounds)

    assert captured_multi == [live_bounds]
    assert captured_norm
    assert all(bounds == live_bounds for bounds in captured_norm)


def test_phase9_env_uses_snapshot_opponent(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.pt"
    _write_snapshot(snapshot_path)
    with open(
        "../experiments/configs/phase9_snapshot_probe.yaml",
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    sim_cfg = dict(config["env"]["sim"])
    league_cfg = {
        "latest": [str(snapshot_path)],
        "historical": [str(snapshot_path)],
        "anchor": [str(snapshot_path)],
        "weights": dict(config["env"]["snapshot_league"]["weights"]),
    }
    env = Phase9SnapshotMappoEnv(
        sim_cfg,
        opponent_bot="snapshot",
        snapshot_paths=[str(snapshot_path)],
        reward_cfg=config["env"]["reward"],
        fog_mode=config["env"]["fog_mode"],
        visible_radius=float(config["env"]["visible_radius"]),
        map_randomization=config["env"]["map_randomization"],
        snapshot_league=league_cfg,
    )
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert info["snapshot_path"] == str(snapshot_path)
        assert info["snapshot_group"] in {"latest", "historical", "anchor"}
        assert info["snapshot_league"] == "latest:0.7:1,historical:0.2:1,anchor:0.1:1"
        assert "map_bounds" in info
        assert len(info["cover_markers"]) == 4
        assert len(info["wall_segments"]) == 2
        assert info["map_layout_hash"] == map_layout_hash(
            info["map_bounds"], info["cover_markers"], info["wall_segments"]
        )
        reset_layout = info["map_layout_hash"]
        critic_obs = np.zeros(CRITIC_DIM, dtype=np.float32)
        env.build_critic_obs(critic_obs)
        assert np.all(np.isfinite(critic_obs))
        next_obs, reward, term, trunc, info = env.step(np.zeros((3, 6), dtype=np.float32))
        assert next_obs.shape == (3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
        assert reward.shape == (3,)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert info["snapshot_path"] == str(snapshot_path)
        assert info["snapshot_group"] in {"latest", "historical", "anchor"}
        assert info["map_layout_hash"] == reset_layout
        assert len(info["wall_segments"]) == 2
    finally:
        env.close()
