import pytest

from xushi2.runner import _build_config


def _valid_sim_cfg() -> dict:
    return {
        "seed": 7,
        "round_length_seconds": 180,
        "fog_of_war_enabled": True,
        "randomize_map": False,
        "map": {"min_x": -5.0, "min_y": -6.0, "max_x": 5.0, "max_y": 6.0},
        "cover_circles": [{"x": 0.0, "y": 1.0, "radius": 1.5}],
        "wall_segments": [{"x1": -1.0, "y1": 0.0, "x2": 1.0, "y2": 0.0, "half_width": 0.25}],
        "mechanics": {
            "revolver_damage_centi_hp": 35,
            "revolver_fire_cooldown_ticks": 8,
            "revolver_hitbox_radius": 0.6,
            "respawn_ticks": 60,
        },
    }


def test_build_config_missing_required_root_key_raises_keyerror() -> None:
    cfg = _valid_sim_cfg()
    del cfg["round_length_seconds"]
    with pytest.raises(KeyError, match="round_length_seconds"):
        _build_config(cfg)


def test_build_config_unknown_root_key_raises_valueerror() -> None:
    cfg = _valid_sim_cfg()
    cfg["fog_of_war_enabld"] = cfg.pop("fog_of_war_enabled")
    with pytest.raises(ValueError, match="fog_of_war_enabld"):
        _build_config(cfg)


def test_build_config_accepts_valid_schema() -> None:
    out = _build_config(_valid_sim_cfg())
    assert out.seed == 7
    assert out.round_length_seconds == 180
    assert out.fog_of_war_enabled is True


def test_build_config_typo_in_nested_section_is_caught() -> None:
    cfg = _valid_sim_cfg()
    cfg["map"]["mix_x"] = cfg["map"].pop("min_x")
    with pytest.raises(ValueError, match="mix_x"):
        _build_config(cfg)
