from train.phases import PHASE_REGISTRY, resolve_phase
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM


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
