"""Phase-4 recurrent MAPPO trainer (modularized)."""

from train.mappo_model import (
    MappoActorCritic,
    MappoConfig,
    MappoEvalStats,
    _eval_outcome_counts,
    compute_team_spirit,
)
from train.mappo_rollout_trainer import MappoRollout, MappoTrainer, make_mappo_config
from train.mappo_bc_pretrain import (
    _collect_walk_bc_sequence,
    _walk_to_objective_targets,
    bc_pretrain_walk_to_objective,
)
from train.mappo_eval_checkpoint import (
    _eval_stats_dict,
    _matrix_gate_label,
    _matrix_native_bot_env_fn,
    _matrix_retention_summary,
    _matrix_snapshot_env_fn,
    _mappo_matrix_row,
    _run_eval_gate,
    _run_mappo_matrix_eval,
    evaluate_mappo,
    train_phase4_from_config,
)

__all__ = [
    "MappoConfig",
    "MappoEvalStats",
    "MappoActorCritic",
    "MappoRollout",
    "MappoTrainer",
    "make_mappo_config",
    "_walk_to_objective_targets",
    "_collect_walk_bc_sequence",
    "bc_pretrain_walk_to_objective",
    "evaluate_mappo",
    "train_phase4_from_config",
    "compute_team_spirit",
    "_eval_outcome_counts",
    "_eval_stats_dict",
    "_run_eval_gate",
    "_matrix_native_bot_env_fn",
    "_matrix_snapshot_env_fn",
    "_mappo_matrix_row",
    "_matrix_retention_summary",
    "_matrix_gate_label",
    "_run_mappo_matrix_eval",
]
