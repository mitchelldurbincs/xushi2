"""Public API surface for recurrent MAPPO training and evaluation."""

import warnings

from train.mappo_bc_pretrain import bc_pretrain_walk_to_objective
from train.mappo_eval_checkpoint import train_phase4_from_config
from train.mappo_evaluate import evaluate_mappo
from train.mappo_model import (
    MappoActorCritic,
    MappoConfig,
    MappoEvalStats,
    compute_majority_on_point_alpha,
    compute_objective_timing_seconds,
    compute_team_spirit,
)
from train.mappo_rollout_trainer import MappoRollout, MappoTrainer, make_mappo_config

__all__ = [
    "MappoActorCritic",
    "MappoConfig",
    "MappoEvalStats",
    "MappoRollout",
    "MappoTrainer",
    "bc_pretrain_walk_to_objective",
    "compute_majority_on_point_alpha",
    "compute_objective_timing_seconds",
    "compute_team_spirit",
    "evaluate_mappo",
    "make_mappo_config",
    "train_phase4_from_config",
]


def __getattr__(name: str):
    deprecated_targets = {
        "_collect_walk_bc_sequence": ("train.mappo_bc_pretrain", "_collect_walk_bc_sequence"),
        "_walk_to_objective_targets": ("train.mappo_bc_pretrain", "_walk_to_objective_targets"),
        "_eval_stats_dict": ("train.mappo_evaluate", "eval_stats_dict"),
        "_mappo_matrix_row": ("train.mappo_matrix_eval", "mappo_matrix_row"),
        "_matrix_gate_label": ("train.mappo_matrix_eval", "matrix_gate_label"),
        "_matrix_native_bot_env_fn": ("train.mappo_matrix_eval", "matrix_native_bot_env_fn"),
        "_matrix_retention_summary": ("train.mappo_matrix_eval", "matrix_retention_summary"),
        "_matrix_snapshot_env_fn": ("train.mappo_matrix_eval", "matrix_snapshot_env_fn"),
        "_run_eval_gate": ("train.mappo_eval_gate_io", "run_eval_gate"),
        "_run_mappo_matrix_eval": ("train.mappo_matrix_eval", "run_mappo_matrix_eval"),
    }
    if name in deprecated_targets:
        module_name, attr = deprecated_targets[name]
        module = __import__(module_name, fromlist=[attr])
        warnings.warn(
            f"train.mappo.{name} is deprecated and will be removed in a future release; "
            "import it from its defining module instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
