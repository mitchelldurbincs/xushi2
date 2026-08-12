"""Config keys nothing reads must be rejected, not silently ignored.

Every consumer reads config with `.get(key, default)`, so a typo used to fall
back to the default and the run completed looking entirely normal. Two
committed keys proved the cost: model.use_recurrence appeared in every Phase-4
config and in eight test files with zero readers, and ppo.minibatch_size was a
*required* key stored on MappoConfig and read by nothing.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import yaml
from _paths import REPO_ROOT

from train.config_schema import ConfigKeyError, validate_config_keys
from train.train import load_config


def _minimal() -> dict:
    return {"phase": 4, "env": {}, "run": {}, "ppo": {}, "model": {}}


# --- every committed config passes --------------------------------------


def test_every_committed_config_passes_key_validation():
    """The registry must describe the configs that actually exist.

    If this fails, either a config gained a key nothing reads, or the registry
    fell behind the code -- both worth stopping for.
    """
    failures = []
    for path in sorted(glob.glob(str(REPO_ROOT / "experiments/configs/**/*.yaml"), recursive=True)):
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            continue
        try:
            validate_config_keys(config)
        except ConfigKeyError as exc:
            failures.append(f"{path}: {exc}")
    assert not failures, "\n".join(failures)


# --- typos are caught ----------------------------------------------------


@pytest.mark.parametrize(
    ("section", "typo", "expected"),
    [
        ("ppo", "entropy_coeff", "entropy_coef"),
        ("ppo", "gae_lamda", "gae_lambda"),
        ("env", "fog_moad", "fog_mode"),
        ("run", "total_update", "total_updates"),
        ("model", "gru_hiden", "gru_hidden"),
    ],
)
def test_near_miss_keys_are_rejected_with_a_suggestion(section, typo, expected):
    config = _minimal()
    config[section][typo] = 1
    with pytest.raises(ConfigKeyError) as exc:
        validate_config_keys(config)
    message = str(exc.value)
    assert f"{section}.{typo}" in message
    assert expected in message, "the error should point at the intended key"


def test_unknown_top_level_section_is_rejected():
    config = _minimal()
    config["pp0"] = {}
    with pytest.raises(ConfigKeyError, match="unknown top-level key"):
        validate_config_keys(config)


def test_all_problems_are_reported_at_once():
    """One run should surface every bad key, not just the first."""
    config = _minimal()
    config["ppo"]["entropy_coeff"] = 1
    config["run"]["output_dirr"] = "x"
    with pytest.raises(ConfigKeyError) as exc:
        validate_config_keys(config)
    message = str(exc.value)
    assert "ppo.entropy_coeff" in message
    assert "run.output_dirr" in message


# --- removed keys get a pointer, not a bare rejection --------------------


@pytest.mark.parametrize(
    ("section", "key", "hint"),
    [
        ("model", "use_recurrence", "always recurrent"),
        ("ppo", "minibatch_size", "full-batch"),
        # env.snapshot_paths left this list 2026-08-09: the self-play campaign
        # implemented it (seeds the snapshot opponent pool).
        ("env", "target_slot", "target_selection_dim"),
    ],
)
def test_removed_keys_explain_themselves(section, key, hint):
    config = _minimal()
    config[section][key] = 1
    with pytest.raises(ConfigKeyError) as exc:
        validate_config_keys(config)
    message = str(exc.value)
    assert "has been removed" in message
    assert hint in message


# --- scope is stated, not accidental -------------------------------------


def test_metadata_is_free_form_and_not_key_checked():
    config = _minimal()
    config["metadata"] = {"hypothesis": "anything", "whatever": 1}
    validate_config_keys(config)


def test_env_sim_block_is_left_to_the_runner():
    # xushi2.runner._build_config already rejects unknown sim keys, and
    # duplicating that here would create a second source of truth.
    config = _minimal()
    config["env"]["sim"] = {"anything_at_all": 1}
    validate_config_keys(config)


def test_load_config_rejects_a_bad_key(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"phase": 4, "ppo": {"entropy_coeff": 0.01}}), encoding="utf-8")
    with pytest.raises(ConfigKeyError, match="entropy_coeff"):
        load_config(path)


def test_load_config_accepts_a_good_config(tmp_path):
    path = tmp_path / "good.yaml"
    path.write_text(
        yaml.safe_dump({"phase": 4, "ppo": {"entropy_coef": 0.01}, "run": {"total_updates": 1}}),
        encoding="utf-8",
    )
    assert load_config(path)["ppo"]["entropy_coef"] == 0.01
