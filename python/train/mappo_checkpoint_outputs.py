from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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
) -> None:
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "config": _checkpoint_config(
                phase=phase,
                phase_label=phase_label,
                ckpt_env_cfg=ckpt_env_cfg,
                mappo_cfg=mappo_cfg,
            ),
        },
        path,
    )


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
    save_mappo_checkpoint(
        path=ckpt_best_eval_path,
        model_state_dict=final_state,
        **common,
    )
    save_mappo_checkpoint(path=ckpt_final_path, model_state_dict=final_state, **common)

    final_alias = "ckpt_best_eval.pt" if has_best_state else "ckpt_last.pt"
    manifest = {
        "ckpt_final_alias": final_alias,
        "best_eval_update_idx": best_eval_update_idx,
        "best_eval_score": (float(best_eval) if best_eval > float("-inf") else None),
        "best_eval_stats": best_eval_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return FinalCheckpointOutputs(
        last_path=ckpt_last_path,
        best_eval_path=ckpt_best_eval_path,
        final_path=ckpt_final_path,
        manifest_path=manifest_path,
        final_alias=final_alias,
        manifest=manifest,
    )
