from train.phases import PHASE_REGISTRY, resolve_phase
from xushi2.entity_obs import ENTITY_OBS_DIM, MULTI_ENEMY_TOKEN_COUNT
from xushi2.grid_obs import ENTITY_GRID_OBS_DIM, MULTI_ENEMY_ENTITY_GRID_OBS_DIM

PHASE10_TARGET_OBS_DIM = MULTI_ENEMY_ENTITY_GRID_OBS_DIM + MULTI_ENEMY_TOKEN_COUNT


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


def test_phase5_registry_declares_entity_attention_shapes() -> None:
    phase, spec = resolve_phase({"phase": 5})
    assert phase == 5
    assert spec["label"] == "phase5"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == ENTITY_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase6_registry_declares_entity_grid_shapes() -> None:
    phase, spec = resolve_phase({"phase": 6})
    assert phase == 6
    assert spec["label"] == "phase6"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase7_registry_declares_partial_obs_shapes() -> None:
    phase, spec = resolve_phase({"phase": 7})
    assert phase == 7
    assert spec["label"] == "phase7"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase8_registry_declares_random_map_shapes() -> None:
    phase, spec = resolve_phase({"phase": 8})
    assert phase == 8
    assert spec["label"] == "phase8"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase9_registry_declares_snapshot_shapes() -> None:
    phase, spec = resolve_phase({"phase": 9})
    assert phase == 9
    assert spec["label"] == "phase9"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 6


def test_phase10_registry_declares_target_slot_shapes() -> None:
    phase, spec = resolve_phase({"phase": 10})
    assert phase == 10
    assert spec["label"] == "phase10"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == PHASE10_TARGET_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 3
    assert spec["action_dim"] == 7
    assert spec["target_action_dim"] == MULTI_ENEMY_TOKEN_COUNT


def test_phase11_registry_declares_current_selfplay_shapes() -> None:
    phase, spec = resolve_phase({"phase": 11})
    assert phase == 11
    assert spec["label"] == "phase11"
    assert spec["training_variants"] == ("mappo",)
    assert spec["obs_dim"] == MULTI_ENEMY_ENTITY_GRID_OBS_DIM
    assert spec["critic_obs_dim"] == 135
    assert spec["n_agents"] == 6
    assert spec["action_dim"] == 6
