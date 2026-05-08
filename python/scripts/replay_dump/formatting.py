from __future__ import annotations

from typing import Any

import numpy as np

_AIM_DELTA_LIMIT = float(3.141592653589793 / 4.0)


def action_to_fields(action_arr: Any, *, include_target: bool = False) -> list[float]:
    """Convert a raw policy action vector to replay action fields.

    Mirrors ``Phase3RangerEnv._action_to_dict`` but returns the *radians*
    aim_delta the sim actually sees (already scaled by π/4)."""
    arr = np.asarray(action_arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 6:
        raise ValueError(f"action must have at least 6 fields, got {arr.shape[0]}")
    mx = float(np.clip(arr[0], -1.0, 1.0))
    my = float(np.clip(arr[1], -1.0, 1.0))
    ad = float(np.clip(arr[2], -1.0, 1.0)) * _AIM_DELTA_LIMIT
    pf = int(np.clip(arr[3], 0.0, 1.0) >= 0.5)
    a1 = int(np.clip(arr[4], 0.0, 1.0) >= 0.5)
    a2 = int(np.clip(arr[5], 0.0, 1.0) >= 0.5)
    fields = [mx, my, ad, float(pf), float(a1), float(a2)]
    if include_target:
        target = 0
        if arr.shape[0] >= 7:
            target = int(np.rint(arr[6]).clip(0, 255))
        fields.append(float(target))
    return fields


def format_decision(tick: int, slot0: list[float], slot3: list[float]) -> str:
    fields = [f"{tick}"]
    for v in slot0 + slot3:
        fields.append(f"{v:.7g}")
    return " ".join(fields)


def format_decision_six(tick: int, slots: list[list[float]]) -> str:
    fields = [f"{tick}"]
    for slot in slots:
        for v in slot:
            fields.append(f"{v:.7g}")
    return " ".join(fields)
