from __future__ import annotations

import copy

import yaml

from tests.test__paths import config_path
from train.mappo import make_mappo_config
from train.runtime_specs import resolve_runtime_spec
from train.train import normalize_entry_config


def _phase4_smoke_config() -> dict:
    with open(config_path("phase4/smoke/phase4_mappo_smoke.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _explicit_phase4_equivalent_config() -> dict:
    cfg = copy.deepcopy(_phase4_smoke_config())
    cfg.pop("phase", None)
    cfg["experiment"] = {"phase": "phase4", "tags": ["runtime-spec-test"]}
    cfg["learner"] = {"kind": "mappo"}
    cfg["env"]["kind"] = "mappo_match"
    cfg["env"]["actor_obs"] = "flat"
    cfg["env"]["critic_obs"] = "team_global"
    cfg["env"]["team_size"] = 3
    cfg["env"]["opponent"] = {"kind": cfg["env"].get("opponent_bot", "basic")}
    cfg["env"]["features"] = {
        "fog": "none",
        "map_randomization": False,
        "target_slot": False,
        "current_selfplay": False,
    }
    return cfg


def test_legacy_phase_config_resolves_to_runtime_spec() -> None:
    runtime = resolve_runtime_spec(_phase4_smoke_config())

    assert runtime.experiment.phase == 4
    assert runtime.phase_label == "phase4"
    assert runtime.learner.kind == "mappo"
    assert runtime.env.kind == "mappo_match"
    assert runtime.shapes.obs_dim == 31
    assert runtime.shapes.critic_obs_dim == 135
    assert runtime.shapes.n_agents == 3
    assert runtime.env_fn is not None


def test_explicit_runtime_config_builds_same_mappo_shapes_as_phase4_smoke() -> None:
    legacy_cfg = _phase4_smoke_config()
    explicit_cfg = _explicit_phase4_equivalent_config()

    legacy = make_mappo_config(legacy_cfg)
    explicit = make_mappo_config(explicit_cfg)

    assert explicit.obs_dim == legacy.obs_dim == 31
    assert explicit.critic_obs_dim == legacy.critic_obs_dim == 135
    assert explicit.n_agents == legacy.n_agents == 3
    assert explicit.action_dim == legacy.action_dim == 6
    assert explicit.target_action_dim == legacy.target_action_dim == 0


def test_explicit_runtime_config_does_not_need_top_level_phase_for_dispatch() -> None:
    normalized = normalize_entry_config(_explicit_phase4_equivalent_config())

    assert normalized.phase_int == 4
    assert normalized.phase_label == "phase4"
    assert normalized.learner_kind == "mappo"
    assert normalized.env_kind == "mappo_match"
