"""Per-platform golden regression for the NATIVE entity-obs path.

The parity suite (test_entity_obs_native_parity.py) checks native-vs-legacy;
it dies with the legacy code at the Phase-3 cutover. THIS test pins the
native path against committed fixtures so post-cutover regressions are
caught. Fixtures are per-platform (repo convention: no cross-machine
bit-reproducibility) and live in
tests/fixtures/entity_obs/<platform>/<scenario>.npz, holding a sha256 of the
full obs stream plus full tensors at sampled steps for debuggability.

Regenerate after an INTENTIONAL obs change:

    X2_REGEN_ENTITY_OBS_GOLDEN=1 pytest tests/test_entity_obs_golden.py

or python scripts/dump_entity_obs_golden.py.
"""

from __future__ import annotations

import hashlib
import os
import platform as _platform
import sys
from pathlib import Path

import numpy as np
import pytest

from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv
from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv

_PLATFORM_KEY = f"{sys.platform}-{_platform.machine()}"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "entity_obs" / _PLATFORM_KEY
_SAMPLED_STEPS = (0, 12, 30, 55)
_TOTAL_STEPS = 66
_REGEN = bool(os.environ.get("X2_REGEN_ENTITY_OBS_GOLDEN"))

_MECHANICS = {
    "revolver_damage_centi_hp": 7500,
    "revolver_fire_cooldown_ticks": 15,
    "revolver_hitbox_radius": 0.75,
    "respawn_ticks": 240,
}


def _scenario_env(name: str):
    if name in ("phase11_team_shared", "phase11_per_agent"):
        return Phase11CurrentSelfplayMappoEnv(
            {
                "round_length_seconds": 20,
                "action_repeat": 3,
                "seed": 991,
                "fog_of_war_enabled": True,
                "mechanics": dict(_MECHANICS),
            },
            reward_cfg={},
            fog_mode=name.removeprefix("phase11_"),
            visible_radius=0.65,
            map_randomization={},
            native_entity_obs=True,
        )
    if name == "phase4_multi_enemy":
        return Phase4MultiEnemyMappoEnv(
            {
                "seed": 0xD1CEDA7A,
                "round_length_seconds": 20,
                "fog_of_war_enabled": True,
                "randomize_map": False,
                "action_repeat": 3,
                "cover_circles": [{"x": 20.0, "y": 15.0, "radius": 4.0}],
                "mechanics": dict(_MECHANICS),
            },
            opponent_bot="basic",
            native_entity_obs=True,
        )
    raise ValueError(f"unknown scenario {name!r}")


def _scenario_action(name: str, step: int, n_rows: int) -> np.ndarray:
    act = np.zeros((n_rows, 6), dtype=np.float32)
    direction = 1.0 if step < 42 else -1.0
    act[:, 1] = direction  # team-relative: +1 walks both teams together
    act[0, 0] = 0.4
    act[:, 2] = 0.1
    act[min(1, n_rows - 1), 3] = 1.0
    return act


def _run_scenario(name: str) -> dict[str, np.ndarray]:
    env = _scenario_env(name)
    try:
        obs, _ = env.reset(seed=991)
        digest = hashlib.sha256()
        digest.update(obs.tobytes())
        samples: dict[str, np.ndarray] = {}
        if 0 in _SAMPLED_STEPS:
            samples["step_0"] = obs.copy()
        n_rows = obs.shape[0]
        for step in range(1, _TOTAL_STEPS):
            obs, _rew, term, trunc, _ = env.step(
                _scenario_action(name, step, n_rows)
            )
            digest.update(obs.tobytes())
            if step in _SAMPLED_STEPS:
                samples[f"step_{step}"] = obs.copy()
            if term or trunc:
                break
        samples["stream_sha256"] = np.frombuffer(
            digest.digest(), dtype=np.uint8
        ).copy()
        return samples
    finally:
        env.close()


def regenerate_all() -> list[Path]:
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("phase11_team_shared", "phase11_per_agent", "phase4_multi_enemy"):
        path = _FIXTURE_DIR / f"{name}.npz"
        np.savez_compressed(path, **_run_scenario(name))
        written.append(path)
    return written


@pytest.mark.parametrize(
    "scenario", ["phase11_team_shared", "phase11_per_agent", "phase4_multi_enemy"]
)
def test_native_entity_obs_matches_golden(scenario: str) -> None:
    fixture_path = _FIXTURE_DIR / f"{scenario}.npz"
    if _REGEN:
        _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fixture_path, **_run_scenario(scenario))
        pytest.skip(f"regenerated {fixture_path}")
    if not fixture_path.exists():
        pytest.skip(
            f"no entity-obs golden fixture for platform {_PLATFORM_KEY}; "
            "generate with X2_REGEN_ENTITY_OBS_GOLDEN=1 pytest "
            "tests/test_entity_obs_golden.py"
        )
    fixture = np.load(fixture_path)
    actual = _run_scenario(scenario)
    np.testing.assert_array_equal(
        actual["stream_sha256"], fixture["stream_sha256"],
        err_msg=(
            f"native entity-obs stream changed for {scenario} on "
            f"{_PLATFORM_KEY}. If intentional, regenerate the fixtures and "
            "note the obs change in the commit."
        ),
    )
    for key in fixture.files:
        if key == "stream_sha256":
            continue
        np.testing.assert_array_equal(
            actual[key], fixture[key], err_msg=f"{scenario}:{key} diverged"
        )
