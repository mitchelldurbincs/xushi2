from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from train.mappo import MappoActorCritic, make_mappo_config
from train.phases import resolve_phase
from xushi2 import xushi2_cpp as _cpp
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM
from xushi2.runner import _build_config
from xushi2.self_play_schedule import SelfPlaySchedule
from xushi2.snapshot_policy import SnapshotLeague, SnapshotPolicy, SnapshotPool
from xushi2.snapshot_retention import SnapshotRetention
from _paths import config_path


def _write_snapshot(path: Path) -> None:
    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"),
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


@pytest.fixture(scope="module")
def shared_snapshot_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    snapshot_path = tmp_path_factory.mktemp("phase9_snapshot") / "snapshot.pt"
    _write_snapshot(snapshot_path)
    return snapshot_path


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


def test_snapshot_policy_loads_and_emits_actions(shared_snapshot_path: Path) -> None:
    policy = SnapshotPolicy(shared_snapshot_path)
    assert policy.phase == 8
    assert policy.cfg.obs_encoder == "entity_attention_grid"


def test_snapshot_policy_observes_live_sim_geometry(
    shared_snapshot_path: Path,
    monkeypatch,
) -> None:
    # Under map randomization the frozen opponent must see the LIVE map, not
    # the checkpoint's stored one. The legacy conversion threaded a
    # map_bounds argument (and this test used to pin that plumbing); the
    # native ObservationEngine reads sim.config().map directly, so the check
    # is now that act()'s internal obs match a reference engine build on the
    # same live sim — including one with bounds unlike the checkpoint's.
    from xushi2.entity_obs_native import snapshot_obs_config
    from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM

    policy = SnapshotPolicy(shared_snapshot_path)
    assert policy._obs_engine is not None

    captured: list[np.ndarray] = []
    real_greedy = policy.model.greedy_action

    def capture_greedy(obs_t, h):
        captured.append(obs_t.cpu().numpy().copy())
        return real_greedy(obs_t, h)

    monkeypatch.setattr(policy.model, "greedy_action", capture_greedy)

    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = _build_config(config["env"]["sim"], seed_override=123)
    cfg.team_size = 3
    # Live map deliberately unlike the checkpoint's stored sim config.
    cfg.map.min_x = -11.0
    cfg.map.min_y = -13.0
    cfg.map.max_x = 61.0
    cfg.map.max_y = 67.0
    sim = _cpp.Sim(cfg)

    policy.act(sim, (3, 4, 5))

    reference = _cpp.ObservationEngine(
        snapshot_obs_config(policy.phase, policy.env_cfg)
    )
    expected = np.zeros((3, MULTI_ENEMY_ENTITY_GRID_OBS_DIM), dtype=np.float32)
    for i, slot in enumerate((3, 4, 5)):
        reference.build_entity_obs(sim, slot, expected[i])

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], expected)
    # The self token's normalized position only lands in [-1, 1] if the
    # LIVE bounds were used; the checkpoint's 50x50 map would put the
    # spawn elsewhere.
    self_pos = captured[0][:, 8:10]
    assert np.all(np.abs(self_pos) <= 1.0)
