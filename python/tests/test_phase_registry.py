import functools

from envs.runtime_factory import make_mappo_match_env
from train.phases import PHASE_REGISTRY, resolve_phase
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM

_SIM_CFG = {
    "seed": 7,
    "round_length_seconds": 10,
    "fog_of_war_enabled": False,
    "randomize_map": False,
    "action_repeat": 3,
    "mechanics": {
        "revolver_damage_centi_hp": 7500,
        "revolver_fire_cooldown_ticks": 15,
        "revolver_hitbox_radius": 0.75,
        "respawn_ticks": 240,
    },
}


def test_phase_registry_entries_have_required_shapes() -> None:
    for phase, spec in PHASE_REGISTRY.items():
        assert "label" in spec
        assert "training_variants" in spec
        variants = spec["training_variants"]
        assert isinstance(variants, tuple)
        if variants:
            for key in (
                "obs_dim",
                "action_dim",
                "continuous_action_dim",
                "binary_action_dim",
                "env_bundle",
            ):
                assert key in spec, f"phase={phase} missing {key}"
            assert callable(spec["env_bundle"])
        else:
            assert "seed_deriver" in spec
            assert callable(spec["seed_deriver"])


def test_phase4_registry_declares_mappo_shapes() -> None:
    phase, spec = resolve_phase({"phase": 4})
    assert phase == 4
    assert spec["label"] == "phase4"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == 31
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase11_registry_declares_current_selfplay_shapes() -> None:
    phase, spec = resolve_phase({"phase": 11})
    assert phase == 11
    assert spec["label"] == "phase11"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 6
    assert spec["action_dim"] == 6


def test_phase_env_bundles_use_the_canonical_env_factory() -> None:
    """Phase bundles must produce make_mappo_match_env partials.

    This is the single-pipeline invariant: the sim_pool vector backend
    recovers env parameters by introspecting these partials, so a phase
    bundle that builds env_fns any other way silently loses sim_pool
    support for every `phase:`-style config.
    """
    phase4_cfg = {"phase": 4, "env": {"sim": dict(_SIM_CFG)}}
    phase11_cfg = {"phase": 11, "env": {"sim": dict(_SIM_CFG)}}
    for cfg in (phase4_cfg, phase11_cfg):
        _phase, spec = resolve_phase(cfg)
        env_fn, _ckpt_env_cfg, _seed = spec["env_bundle"](cfg)
        assert isinstance(env_fn, functools.partial)
        assert env_fn.func is make_mappo_match_env
