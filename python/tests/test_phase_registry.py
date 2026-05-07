from train.phases import PHASE_REGISTRY


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
