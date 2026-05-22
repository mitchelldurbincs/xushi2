"""Run the opt-in Phase 4 multi-enemy supervised bridge without W&B/PPO."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

from train.full_env_rehearsal import (
    closed_loop_supervised_bridge_pretrain,
    full_env_rehearsal_pretrain,
    run_full_env_rehearsal_gate,
)
from train.mappo_pretrain_hooks import maybe_warm_start
from train.mappo_rollout_trainer import MappoTrainer
from train.mappo_runtime_context import build_runtime_context


def run_bridge_gate(config: dict[str, Any]) -> dict[str, Any]:
    context = build_runtime_context(config)
    run_cfg = context.run_cfg
    bridge_cfg = dict(run_cfg.get("multi_enemy_supervised_bridge", {}))
    if not bool(bridge_cfg.get("enabled", False)):
        raise ValueError("run.multi_enemy_supervised_bridge.enabled must be true")
    teacher = str(bridge_cfg.get("teacher", "multi_enemy_visible"))
    if teacher != "multi_enemy_visible":
        raise ValueError("multi-enemy supervised bridge only supports teacher='multi_enemy_visible'")
    if context.cfg.obs_encoder != "entity_attention_grid":
        raise ValueError("multi-enemy supervised bridge requires entity_attention_grid obs")
    if context.cfg.target_action_dim != 0:
        raise ValueError("multi-enemy supervised bridge must not add action-space target fields")

    trainer = MappoTrainer(context.env_fn, context.cfg, seed=context.seed_base)
    try:
        maybe_warm_start(context, trainer)
        pretrain_cfg = {
            **bridge_cfg,
            "seed": int(bridge_cfg.get("seed", context.seed_base + 70_000)),
            "log_label": context.phase_label,
        }
        closed_loop_enabled = bool(
            dict(bridge_cfg.get("closed_loop", {})).get("enabled", False)
        )
        if closed_loop_enabled:
            metrics = closed_loop_supervised_bridge_pretrain(
                trainer.model,
                context.env_fn,
                pretrain_cfg,
            )
            agreement_path = context.output_dir / str(
                bridge_cfg.get(
                    "agreement_output",
                    "multi_enemy_closed_loop_supervised_bridge_agreement.json",
                )
            )
            agreement_payload = {
                "mode": "closed_loop_policy_states",
                "teacher": teacher,
                "metrics": {
                    key: float(value)
                    for key, value in metrics.items()
                    if key.startswith("agreement_") or key.startswith("final_agreement_")
                },
            }
            agreement_path.parent.mkdir(parents=True, exist_ok=True)
            agreement_path.write_text(
                json.dumps(agreement_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            metrics = full_env_rehearsal_pretrain(
                trainer.model,
                context.env_fn,
                pretrain_cfg,
            )
            agreement_path = None
        checkpoint_name = (
            "ckpt_multi_enemy_closed_loop_supervised_bridge.pt"
            if closed_loop_enabled
            else "ckpt_multi_enemy_supervised_bridge.pt"
        )
        checkpoint_path = context.output_dir / checkpoint_name
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "config": {"mappo": trainer.model.cfg.__dict__, "env": context.ckpt_env_cfg},
            },
            checkpoint_path,
        )
        gate = run_full_env_rehearsal_gate(
            trainer.model,
            context.env_fn,
            gate=dict(bridge_cfg.get("gate", {})),
            output_dir=context.output_dir,
            seed=context.seed_base + 96_000,
            checkpoint_path=checkpoint_path,
        )
    finally:
        trainer.close()

    return {
        "status": gate.status,
        "passed": gate.passed,
        "pretrain_metrics": metrics,
        "gate_path": str(gate.path),
        "checkpoint_path": str(checkpoint_path),
        "agreement_path": None if agreement_path is None else str(agreement_path),
        "gate_metrics": gate.metrics,
        "gate_thresholds": gate.thresholds,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    summary = run_bridge_gate(config)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
