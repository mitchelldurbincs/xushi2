from __future__ import annotations

import pytest

from train.benchmark.run import _run_once
from train.train import load_config
from _paths import config_path


@pytest.fixture(scope="module")
def mappo_smoke_config() -> dict:
    return load_config(config_path("phase4/smoke/phase4_mappo_smoke.yaml"))


@pytest.fixture(scope="module")
def ppo_smoke_config() -> dict:
    return load_config(config_path("phase3/smoke/phase3_ranger_smoke.yaml"))


def test_run_once_mappo_smoke(mappo_smoke_config: dict) -> None:
    result = _run_once(
        config=mappo_smoke_config,
        target="mappo",
        warmup_iterations=0,
        measured_iterations=1,
        seed=0,
        repeat_index=0,
        vector_env="sync",
    )

    assert result.measured_iterations == 1
    assert result.total_samples_processed == result.env_steps_per_iteration
    assert result.rollout_wall_time_sec > 0.0
    assert result.total_wall_time_sec >= result.rollout_wall_time_sec
    assert result.env_steps_per_sec > 0.0


def test_run_once_env_step_only_does_not_bill_update(ppo_smoke_config: dict) -> None:
    result = _run_once(
        config=ppo_smoke_config,
        target="env_step_only",
        warmup_iterations=0,
        measured_iterations=1,
        seed=0,
        repeat_index=0,
        vector_env="sync",
    )

    assert result.update_wall_time_sec == 0.0
    assert result.rollout_wall_time_sec > 0.0
    assert result.learner_steps_per_sec == 0.0
    assert result.env_steps_per_sec > 0.0


def test_run_once_update_only_does_not_bill_rollout(ppo_smoke_config: dict) -> None:
    result = _run_once(
        config=ppo_smoke_config,
        target="update_only",
        warmup_iterations=0,
        measured_iterations=1,
        seed=0,
        repeat_index=0,
        vector_env="sync",
    )

    assert result.rollout_wall_time_sec == 0.0
    assert result.update_wall_time_sec > 0.0
    assert result.env_steps_per_sec == 0.0
    assert result.learner_steps_per_sec > 0.0
