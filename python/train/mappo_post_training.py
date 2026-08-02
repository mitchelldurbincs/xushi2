from __future__ import annotations

import json
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


def transfer_rows_and_gate(
    rows: list[dict[str, Any]], transfer_bots: tuple[str, ...]
) -> tuple[list[dict[str, Any]], str]:
    """Select transfer-summary rows and derive the gate status.

    Filters by opponent NAME only — snapshot rows explicitly listed in
    transfer_bots must appear (they were silently dropped by an
    opponent_type == "bot" requirement until 2026-08-02). The noop cap-gain
    check only means something when a noop row was requested; matrices
    without noop are "ungated" rather than forever "evidence_insufficient".
    """
    transfer_rows = [
        row for row in rows if str(row.get("opponent", "")) in transfer_bots
    ]
    noop_rows = [row for row in transfer_rows if str(row.get("opponent", "")) == "noop"]
    if not noop_rows:
        return transfer_rows, "ungated"
    noop_cap_gain = max(
        float(row.get("mean_cap_progress_gain_ticks", 0.0)) for row in noop_rows
    )
    return transfer_rows, ("pass" if noop_cap_gain > 0.0 else "evidence_insufficient")


def _transfer_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    wins = int(round(float(row.get("win_rate", 0.0)) * int(row.get("episodes", 0))))
    losses = int(round(float(row.get("loss_rate", 0.0)) * int(row.get("episodes", 0))))
    draws = int(round(float(row.get("draw_rate", 0.0)) * int(row.get("episodes", 0))))
    return {
        "opponent": str(row.get("opponent", "")),
        "opponent_type": str(row.get("opponent_type", "")),
        "episodes": int(row.get("episodes", 0)),
        "score_team_a": float(row.get("mean_score_a", 0.0)),
        "score_team_b": float(row.get("mean_score_b", 0.0)),
        "win_loss_draw": {"wins": wins, "losses": losses, "draws": draws},
        "majority_seconds": {
            "team_a": float(row.get("mean_majority_on_point_seconds_a", 0.0)),
            "team_b": float(row.get("mean_majority_on_point_seconds_b", 0.0)),
        },
        "uncontested_seconds": {
            "team_a": float(row.get("mean_uncontested_on_point_seconds_a", 0.0)),
            "team_b": float(row.get("mean_uncontested_on_point_seconds_b", 0.0)),
        },
        "cap_progress_gain_ticks": float(row.get("mean_cap_progress_gain_ticks", 0.0)),
    }


def _write_transfer_summary(output_dir, transfer_rows: list[dict[str, Any]], *, gate_status: str) -> None:
    payload = {
        "gate_status": gate_status,
        "rows": [_transfer_row_summary(row) for row in transfer_rows],
    }
    json_path = output_dir / "transfer_summary.json"
    md_path = output_dir / "transfer_summary.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Transfer summary",
        "",
        f"- gate_status: **{gate_status}**",
        "",
        "| opponent | type | score A/B | W/L/D | majority A/B s | uncontested A/B s | cap-gain |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        wld = row["win_loss_draw"]
        maj = row["majority_seconds"]
        unc = row["uncontested_seconds"]
        lines.append(
            f"| {row['opponent']} | {row['opponent_type']} | "
            f"{row['score_team_a']:.2f}/{row['score_team_b']:.2f} | "
            f"{wld['wins']}/{wld['losses']}/{wld['draws']} | "
            f"{maj['team_a']:.2f}/{maj['team_b']:.2f} | "
            f"{unc['team_a']:.2f}/{unc['team_b']:.2f} | "
            f"{row['cap_progress_gain_ticks']:.1f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    transfer_bots = tuple(
        str(bot)
        for bot in dict(run_cfg.get("matrix_eval", {})).get(
            "transfer_bots", ("noop", "weak_basic_v2", "basic")
        )
    )
    transfer_rows, gate_status = transfer_rows_and_gate(rows, transfer_bots)
    _write_transfer_summary(context.output_dir, transfer_rows, gate_status=gate_status)
    transfer_fail_on_insufficient = bool(
        dict(run_cfg.get("matrix_eval", {})).get("transfer_fail_on_insufficient", False)
    )
    if gate_status == "evidence_insufficient" and transfer_fail_on_insufficient:
        raise RuntimeError(
            "MAPPO transfer gate evidence insufficient: nonzero objective conversion vs noop not observed"
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
