from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint so a crash cannot leave a truncated file.

    torch.save writes in place, and the three final checkpoints are written
    back to back, so an OOM kill or a full disk partway through could corrupt
    all of them at once. Write to a sibling temp file and rename: os.replace is
    atomic within a filesystem, so readers see either the old file or the new
    one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class FinalCheckpointOutputs:
    last_path: Path
    best_eval_path: Path
    final_path: Path
    manifest_path: Path
    final_alias: str
    manifest: dict[str, Any]


def _checkpoint_config(
    *,
    phase: int | None,
    phase_label: str,
    ckpt_env_cfg: dict[str, Any],
    mappo_cfg: Any,
) -> dict[str, Any]:
    return {
        "phase": phase_label if phase is None else phase,
        "env": ckpt_env_cfg,
        "mappo": mappo_cfg.__dict__,
    }


def save_mappo_checkpoint(
    *,
    path: Path,
    model_state_dict: dict[str, Any],
    phase: int | None,
    phase_label: str,
    ckpt_env_cfg: dict[str, Any],
    mappo_cfg: Any,
    resume_state: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint, optionally including everything needed to resume.

    ``resume_state`` carries optimizer moments, the update index, and RNG state.
    Recurrent state intentionally restarts with the freshly reset environments;
    environment state is not serializable at this boundary. Without resume
    state a checkpoint can only be warm-started from, which resets Adam and
    restarts the LR schedule -- a large silent optimization discontinuity that
    presents as "the run got worse after restart".
    """
    payload: dict[str, Any] = {
        "model_state_dict": model_state_dict,
        "config": _checkpoint_config(
            phase=phase,
            phase_label=phase_label,
            ckpt_env_cfg=ckpt_env_cfg,
            mappo_cfg=mappo_cfg,
        ),
    }
    if resume_state is not None:
        payload["resume_state"] = resume_state
    _atomic_torch_save(payload, path)


def save_final_mappo_checkpoints(
    *,
    output_dir: Path,
    last_state: dict[str, Any],
    final_state: dict[str, Any],
    has_best_state: bool,
    phase: int | None,
    phase_label: str,
    ckpt_env_cfg: dict[str, Any],
    mappo_cfg: Any,
    best_eval_update_idx: int | None,
    best_eval: float,
    best_eval_stats: dict[str, float | int] | None,
) -> FinalCheckpointOutputs:
    ckpt_last_path = (output_dir / "ckpt_last.pt").resolve()
    ckpt_best_eval_path = (output_dir / "ckpt_best_eval.pt").resolve()
    ckpt_final_path = (output_dir / "ckpt_final.pt").resolve()
    manifest_path = (output_dir / "checkpoint_manifest.json").resolve()

    common = {
        "phase": phase,
        "phase_label": phase_label,
        "ckpt_env_cfg": ckpt_env_cfg,
        "mappo_cfg": mappo_cfg,
    }
    save_mappo_checkpoint(path=ckpt_last_path, model_state_dict=last_state, **common)
    # ckpt_best_eval.pt is written only when a best state actually exists.
    # Writing it unconditionally meant it silently held the *last* weights
    # whenever no eval had improved, and several scripts load it by name.
    if has_best_state:
        save_mappo_checkpoint(
            path=ckpt_best_eval_path,
            model_state_dict=final_state,
            **common,
        )
    else:
        ckpt_best_eval_path.unlink(missing_ok=True)
    save_mappo_checkpoint(path=ckpt_final_path, model_state_dict=final_state, **common)

    final_alias = "ckpt_best_eval.pt" if has_best_state else "ckpt_last.pt"
    manifest = {
        "ckpt_final_alias": final_alias,
        "has_best_eval_checkpoint": has_best_state,
        "best_eval_update_idx": best_eval_update_idx,
        "best_eval_score": (
            float(best_eval) if math.isfinite(best_eval) and best_eval > float("-inf") else None
        ),
        "best_eval_stats": best_eval_stats,
    }
    _atomic_write_text(json.dumps(manifest, indent=2) + "\n", manifest_path)
    return FinalCheckpointOutputs(
        last_path=ckpt_last_path,
        best_eval_path=ckpt_best_eval_path,
        final_path=ckpt_final_path,
        manifest_path=manifest_path,
        final_alias=final_alias,
        manifest=manifest,
    )
