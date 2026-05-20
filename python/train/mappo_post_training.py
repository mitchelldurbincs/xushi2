from __future__ import annotations

from typing import Any

from train.mappo_eval_gate_io import read_json_artifact
from train.mappo_matrix_eval import (
    CheckpointEnvConfig,
    MatrixEvalConfig,
    matrix_gate_label,
    matrix_retention_summary,
    run_mappo_matrix_eval,
)
from train.mappo_model import MappoActorCritic
from train.mappo_runtime_context import RuntimeContext


def maybe_run_post_training_matrix_eval(
    *,
    context: RuntimeContext,
    final_state: dict[str, Any],
    best_eval: float,
    last_eval: float,
    total_updates: int,
) -> None:
    run_cfg = context.run_cfg
    if not run_cfg.get("matrix_eval"):
        return

    matrix_model = MappoActorCritic(context.cfg)
    matrix_model.load_state_dict(final_state)
    matrix_model.eval()
    rows = run_mappo_matrix_eval(
        model=matrix_model,
        phase=0 if context.phase is None else context.phase,
        ckpt_env_cfg=CheckpointEnvConfig(context.ckpt_env_cfg),
        matrix_cfg=MatrixEvalConfig.from_dict(dict(run_cfg.get("matrix_eval", {}))),
        output_dir=context.output_dir,
        seed=context.seed_base,
    )
    if context.retention is None:
        return

    gate: dict | None = None
    matrix_cfg = dict(run_cfg.get("matrix_eval", {}))
    if matrix_cfg.get("gate"):
        gate_path = context.output_dir / str(matrix_cfg.get("gate_output", "matrix_gate.json"))
        gate = read_json_artifact(gate_path)
    summary = matrix_retention_summary(rows, gate)
    manifest = context.retention.record_checkpoint(
        context.output_dir / "ckpt_final.pt",
        update=total_updates,
        score=best_eval if best_eval > float("-inf") else last_eval,
        matrix_score=float(summary["matrix_score"]),
        matrix_gate_passed=(
            bool(summary["matrix_gate_passed"])
            if summary["matrix_gate_passed"] is not None
            else None
        ),
        matrix_rows=int(summary["matrix_rows"]),
    )
    print(
        f"[{context.phase_label}/mappo] snapshot_pool_matrix "
        f"score={float(summary['matrix_score']):+.3f} "
        f"gate={matrix_gate_label(summary['matrix_gate_passed'])} "
        f"latest={len(manifest['latest'])} "
        f"historical={len(manifest['historical'])} "
        f"anchor={len(manifest['anchor'])}",
        flush=True,
    )
