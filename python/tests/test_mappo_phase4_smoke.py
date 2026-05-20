from pathlib import Path

import pytest
import yaml
from _paths import config_path

from train.mappo import (
    MappoTrainer,
    evaluate_mappo,
    make_mappo_config,
    train_phase4_from_config,
)
from train.phases import resolve_phase

pytestmark = pytest.mark.slow


def test_phase4_mappo_smoke_train_runs_one_update(tmp_path: Path) -> None:
    with open(
        config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    config["run"] = dict(config["run"])
    config["run"]["total_updates"] = 1
    config["run"]["eval_every"] = 1
    config["run"]["eval_episodes"] = 1
    config["run"]["checkpoint_every"] = 1
    config["run"]["output_dir"] = str(tmp_path / "phase4")
    result = train_phase4_from_config(config)
    assert set(result) == {"mappo"}
    assert (tmp_path / "phase4" / "mappo" / "ckpt_final.pt").exists()


def test_phase4_mappo_eval_reports_diagnostics() -> None:
    with open(
        config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8"
    ) as fh:
        config = yaml.safe_load(fh)
    phase, spec = resolve_phase(config)
    assert phase == 4
    env_fn, _ckpt_env_cfg, seed_base = spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    trainer = MappoTrainer(env_fn, cfg, seed=seed_base)
    try:
        stats = evaluate_mappo(trainer.model, env_fn, episodes=1, seed=seed_base + 1)
    finally:
        trainer.close()
    assert stats.episodes == 1
    assert stats.wins + stats.losses + stats.draws == 1
    assert stats.terminated + stats.truncated == 1
    assert stats.mean_final_tick > 0
