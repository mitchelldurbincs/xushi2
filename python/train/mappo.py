"""Phase-4 recurrent MAPPO trainer (modularized)."""

from train.mappo_bc_pretrain import (
    _collect_walk_bc_sequence,
    _walk_to_objective_targets,
    bc_pretrain_walk_to_objective,
)
from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_eval_gate_io import run_eval_gate
from train.mappo_matrix_eval import (
    mappo_matrix_row,
    matrix_gate_label,
    matrix_native_bot_env_fn,
    matrix_retention_summary,
    matrix_snapshot_env_fn,
    run_mappo_matrix_eval,
)
from train.mappo_eval_checkpoint import train_phase4_from_config

from train.mappo_model import (
    MappoActorCritic,
    MappoConfig,
    MappoEvalStats,
    _eval_outcome_counts,
    compute_team_spirit,
)
from train.mappo_rollout_trainer import MappoRollout, MappoTrainer, make_mappo_config

__all__ = [
    "MappoActorCritic",
    "MappoConfig",
    "MappoEvalStats",
    "MappoRollout",
    "MappoTrainer",
    "_collect_walk_bc_sequence",
    "_eval_outcome_counts",
    "eval_stats_dict",
    "mappo_matrix_row",
    "matrix_gate_label",
    "matrix_native_bot_env_fn",
    "matrix_retention_summary",
    "matrix_snapshot_env_fn",
    "run_eval_gate",
    "run_mappo_matrix_eval",
    "_walk_to_objective_targets",
    "bc_pretrain_walk_to_objective",
    "compute_team_spirit",
    "evaluate_mappo",
    "make_mappo_config",
    "train_phase4_from_config",
]
