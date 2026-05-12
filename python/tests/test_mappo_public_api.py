from train import mappo


def test_mappo_public_api_surface() -> None:
    assert mappo.__all__ == [
        "MappoActorCritic",
        "MappoConfig",
        "MappoEvalStats",
        "MappoRollout",
        "MappoTrainer",
        "bc_pretrain_walk_to_objective",
        "compute_team_spirit",
        "evaluate_mappo",
        "make_mappo_config",
        "train_phase4_from_config",
    ]
