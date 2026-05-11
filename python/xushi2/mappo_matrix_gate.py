"""Gate helpers for compact MAPPO matchup-matrix diagnostics."""

from __future__ import annotations

from typing import Any


def check_matrix_gate(rows: list[dict[str, Any]], gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a compact pass/fail summary for matchup-matrix rows.

    ``gate_cfg`` accepts:
    - ``min_win_rate``: mapping from opponent_type to minimum win rate.
    - ``max_draw_rate``: mapping from opponent_type to maximum draw rate.
    - ``min_rows``: minimum number of matrix rows expected.
    """
    min_rows = int(gate_cfg.get("min_rows", 1))
    failures: list[str] = []
    if len(rows) < min_rows:
        failures.append(f"row_count {len(rows)} < min_rows {min_rows}")

    min_win_rate = {str(k): float(v) for k, v in dict(gate_cfg.get("min_win_rate", {})).items()}
    max_draw_rate = {str(k): float(v) for k, v in dict(gate_cfg.get("max_draw_rate", {})).items()}
    counts_by_type: dict[str, int] = {}
    for idx, row in enumerate(rows):
        opponent_type = str(row.get("opponent_type", ""))
        counts_by_type[opponent_type] = counts_by_type.get(opponent_type, 0) + 1
        win_rate = float(row.get("win_rate", 0.0))
        draw_rate = float(row.get("draw_rate", 0.0))
        if opponent_type in min_win_rate and win_rate < min_win_rate[opponent_type]:
            failures.append(
                f"row {idx} {opponent_type} win_rate {win_rate:.3f} "
                f"< {min_win_rate[opponent_type]:.3f}"
            )
        if opponent_type in max_draw_rate and draw_rate > max_draw_rate[opponent_type]:
            failures.append(
                f"row {idx} {opponent_type} draw_rate {draw_rate:.3f} "
                f"> {max_draw_rate[opponent_type]:.3f}"
            )

    for opponent_type in sorted(set(min_win_rate) | set(max_draw_rate)):
        if counts_by_type.get(opponent_type, 0) == 0:
            failures.append(f"missing opponent_type {opponent_type!r}")

    return {
        "passed": not failures,
        "failures": failures,
        "row_count": len(rows),
        "counts_by_type": counts_by_type,
        "criteria": {
            "min_rows": min_rows,
            "min_win_rate": min_win_rate,
            "max_draw_rate": max_draw_rate,
        },
    }
