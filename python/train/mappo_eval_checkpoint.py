from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from train.common_orchestration import LoopConfig, run_training_loop
from train.composition_rehearsal import (
    build_phase4_env_fn_with_overrides,
    composition_rehearsal_pretrain,
    load_frozen_mappo_teacher,
    run_composition_diagnostics,
)
from train.mappo_bc_pretrain import (
    bc_pretrain_walk_and_shoot_to_objective,
    bc_pretrain_walk_to_objective,
)
from train.mappo_eval_gate_io import EvalGateConfig, read_json_artifact, run_eval_gate
from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_matrix_eval import (
    CheckpointEnvConfig,
    MatrixEvalConfig,
    matrix_gate_label,
    matrix_retention_summary,
    run_mappo_matrix_eval,
)
from train.mappo_model import MappoActorCritic, compute_team_spirit
from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.phases import resolve_phase
from train.wandb_logger import make_logger
from xushi2.snapshot_retention import SnapshotRetention


def train_phase4_from_config(config: dict) -> dict[str, float]:
    phase, phase_spec = resolve_phase(config)
    phase_label = str(phase_spec["label"])
    env_fn, ckpt_env_cfg, seed_base = phase_spec["env_bundle"](config)
    cfg = make_mappo_config(config)
    run_cfg = config.get("run", {})
    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 10))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    output_dir = Path(str(run_cfg.get("output_dir", "runs/phase4_mappo"))) / "mappo"
    output_dir.mkdir(parents=True, exist_ok=True)
    retention: SnapshotRetention | None = None
    if run_cfg.get("snapshot_retention"):
        retention_cfg = dict(run_cfg.get("snapshot_retention", {}))
        env_cfg = config.get("env", {})
        retention = SnapshotRetention(
            output_dir / str(retention_cfg.get("manifest", "snapshot_league.json")),
            max_latest=int(retention_cfg.get("max_latest", 20)),
            preserve_best=int(retention_cfg.get("preserve_best", 3)),
            anchor_paths=tuple(retention_cfg.get("anchor_paths", env_cfg.get("snapshot_paths", ())))
            if bool(retention_cfg.get("include_config_anchors", True))
            else (),
            weights=dict(
                retention_cfg.get(
                    "weights",
                    env_cfg.get("snapshot_league", {}).get(
                        "weights",
                        {"latest": 0.7, "historical": 0.2, "anchor": 0.1},
                    ),
                )
            ),
        )

    trainer = MappoTrainer(env_fn, cfg, seed=seed_base)

    wandb_logger = make_logger(
        config=config,
        run_name=f"{phase_label}_mappo_seed{seed_base}",
        run_config={
            "phase": int(phase),
            "phase_label": phase_label,
            "variant": "mappo",
            "seed": int(seed_base),
            "total_updates": total_updates,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "mappo": dict(cfg.__dict__),
            "env": dict(ckpt_env_cfg),
        },
        tags=[f"phase{int(phase)}", phase_label, "mappo"],
    )

    # Warm-start: optionally load a previously-trained checkpoint into the
    # newly-constructed trainer's model. Used by the Phase 4 cap-training
    # escalation (docs/plans/2026-05-08-phase4-cap-training-escalation.md)
    # to seed a basic-opponent run from a noop-trained policy that already
    # knows how to hold the cap. Loaded BEFORE BC pretrain so BC, if also
    # configured, fine-tunes on top of the warm-started weights.
    init_ckpt = run_cfg.get("init_from_checkpoint")
    if init_ckpt:
        raw = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        if (
            cfg.aim_aux_coef > 0.0
            or cfg.target_selection_dim > 0
            or cfg.target_conditioned_combat
            or cfg.mode_gated_combat
        ):
            load_result = trainer.model.load_state_dict(raw["model_state_dict"], strict=False)
            unexpected = list(load_result.unexpected_keys)
            allowed_missing_prefixes = ["actor_aim_aux_head."]
            if cfg.target_selection_dim > 0:
                allowed_missing_prefixes.append("actor_target_selection_head.")
            if cfg.target_conditioned_combat:
                allowed_missing_prefixes.append("actor_target_condition.")
            if cfg.mode_gated_combat:
                allowed_missing_prefixes.append("actor_mode_head.")
            missing = [
                key
                for key in load_result.missing_keys
                if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "warm-start checkpoint mismatch outside new auxiliary/target heads: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            trainer.model.load_state_dict(raw["model_state_dict"], strict=True)
        print(
            f"[{phase_label}/mappo] warm-start: loaded {init_ckpt}",
            flush=True,
        )

    best_eval = float("-inf")
    best_state: dict | None = None
    best_eval_update_idx: int | None = None
    best_eval_stats: dict[str, float | int] | None = None
    last_eval = float("nan")
    try:
        composition_gate_passed = True
        if bool(run_cfg.get("composition_pretrain", False)):
            objective_teacher_ckpt = run_cfg.get("composition_objective_teacher_checkpoint")
            combat_teacher_ckpt = run_cfg.get("composition_combat_teacher_checkpoint")
            if not objective_teacher_ckpt:
                raise ValueError(
                    "run.composition_objective_teacher_checkpoint is required when "
                    "composition_pretrain is true"
                )
            if not combat_teacher_ckpt:
                raise ValueError(
                    "run.composition_combat_teacher_checkpoint is required when "
                    "composition_pretrain is true"
                )
            objective_teacher = load_frozen_mappo_teacher(objective_teacher_ckpt)
            combat_teacher = load_frozen_mappo_teacher(combat_teacher_ckpt)
            objective_env_fn = build_phase4_env_fn_with_overrides(
                ckpt_env_cfg,
                dict(run_cfg.get("composition_objective_env", {})),
            )
            combat_env_fn = build_phase4_env_fn_with_overrides(
                ckpt_env_cfg,
                dict(run_cfg.get("composition_combat_env", {})),
            )
            full_eval_env_fn = build_phase4_env_fn_with_overrides(
                ckpt_env_cfg,
                {
                    "opponent_bot": "weak_basic_v2",
                    "mini_game": None,
                    "mini_game_config": {},
                },
            )
            metrics = composition_rehearsal_pretrain(
                trainer.model,
                objective_teacher,
                combat_teacher,
                objective_env_fn,
                combat_env_fn,
                {
                    "steps": int(run_cfg.get("composition_pretrain_steps", 1000)),
                    "objective_batch_size": int(
                        run_cfg.get("composition_objective_batch_size", 256)
                    ),
                    "combat_batch_size": int(
                        run_cfg.get("composition_combat_batch_size", 256)
                    ),
                    "learning_rate": float(
                        run_cfg.get(
                            "composition_learning_rate",
                            run_cfg.get("bc_learning_rate", 1.0e-3),
                        )
                    ),
                    "seed": seed_base + 40_000,
                    "log_label": phase_label,
                },
            )
            if metrics:
                wandb_logger.log(
                    {f"composition_pretrain/{k}": float(v) for k, v in metrics.items()},
                    step=0,
                )
            diagnostics = run_composition_diagnostics(
                trainer.model,
                objective_env_fn=objective_env_fn,
                combat_env_fn=combat_env_fn,
                full_env_fn=full_eval_env_fn,
                episodes=int(run_cfg.get("composition_eval_episodes", eval_episodes)),
                seed=seed_base + 80_000,
                gate=dict(run_cfg.get("composition_gate", {})),
            )
            composition_gate_passed = diagnostics.passed
            gate_metrics = diagnostics.metrics
            print(
                f"[{phase_label}/mappo] composition_gate "
                f"passed={diagnostics.passed} "
                f"objective_onpt={diagnostics.objective_on_point:.3f}>"
                f"{gate_metrics['gate_objective_on_point']:.3f} "
                f"objective_losses={diagnostics.objective_losses}<="
                f"{gate_metrics['gate_objective_losses']:.0f} "
                f"combat_kills={diagnostics.combat_kills:.2f}>="
                f"{gate_metrics['gate_combat_kills']:.2f} "
                f"full_hit_fire={diagnostics.full_hit_fire:.4f}>"
                f"{gate_metrics['gate_hit_fire']:.4f} "
                f"full_aim_error={diagnostics.full_aim_error:.3f}<"
                f"{gate_metrics['gate_aim_error']:.3f}",
                flush=True,
            )
            wandb_logger.log(
                {
                    f"composition_eval/{k}": float(v)
                    for k, v in diagnostics.metrics.items()
                },
                step=0,
            )
            if not composition_gate_passed:
                total_updates = 0

        bc_steps = int(run_cfg.get("bc_pretrain_steps", 0))
        if bc_steps > 0 and composition_gate_passed:
            bc_variant = str(run_cfg.get("bc_pretrain_variant", "walk_to_objective"))
            bc_kwargs = {
                "steps": bc_steps,
                "batch_size": int(run_cfg.get("bc_batch_size", 1024)),
                "learning_rate": float(run_cfg.get("bc_learning_rate", 1.0e-3)),
                "seed": seed_base + 50_000,
                "log_label": phase_label,
                "freeze_actor_aim": bool(run_cfg.get("bc_freeze_actor_aim", False)),
            }
            if bc_variant == "walk_and_shoot":
                aim_rehearsal_env_fn = None
                aim_rehearsal_batch_size = int(run_cfg.get("bc_aim_rehearsal_batch_size", 0))
                if aim_rehearsal_batch_size > 0:
                    from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv

                    target_ckpt = run_cfg.get("bc_aim_target_checkpoint")
                    if not target_ckpt:
                        raise ValueError(
                            "bc_aim_rehearsal_batch_size requires bc_aim_target_checkpoint"
                        )
                    target_raw = torch.load(target_ckpt, map_location="cpu", weights_only=False)
                    mini_game_cfg = dict(
                        target_raw.get("config", {}).get("env", {}).get(
                            "mini_game_config", {}
                        )
                    )
                    if not mini_game_cfg:
                        raise ValueError(
                            "bc_aim_target_checkpoint does not contain env.mini_game_config"
                        )

                    def aim_rehearsal_env_fn() -> Phase4AimOnlyMappoEnv:
                        return Phase4AimOnlyMappoEnv(**mini_game_cfg)
                bc_pretrain_walk_and_shoot_to_objective(
                    trainer.model,
                    env_fn,
                    cfg,
                    aim_target_checkpoint=run_cfg.get("bc_aim_target_checkpoint"),
                    aim_rehearsal_env_fn=aim_rehearsal_env_fn,
                    aim_rehearsal_batch_size=aim_rehearsal_batch_size,
                    **bc_kwargs,
                )
            else:
                bc_pretrain_walk_to_objective(
                    trainer.model,
                    env_fn,
                    cfg,
                    **bc_kwargs,
                )
            eval_stats = evaluate_mappo(
                trainer.model,
                env_fn,
                episodes=eval_episodes,
                seed=seed_base + 90_000,
            )
            last_eval = eval_stats.mean_reward
            if last_eval > best_eval:
                best_eval = last_eval
                best_state = copy.deepcopy(trainer.model.state_dict())
                best_eval_update_idx = 0
                best_eval_stats = {
                    "wins": int(eval_stats.wins),
                    "losses": int(eval_stats.losses),
                    "draws": int(eval_stats.draws),
                    "score_team_a": float(eval_stats.mean_team_a_score),
                    "score_team_b": float(eval_stats.mean_team_b_score),
                    "kills_team_a": float(eval_stats.mean_team_a_kills),
                    "kills_team_b": float(eval_stats.mean_team_b_kills),
                }
            print(
                f"[{phase_label}/mappo] bc_eval "
                f"mean_reward={eval_stats.mean_reward:+.3f} "
                f"wins={eval_stats.wins}/{eval_stats.episodes} "
                f"draws={eval_stats.draws}/{eval_stats.episodes} "
                f"score={eval_stats.mean_team_a_score:.2f}/"
                f"{eval_stats.mean_team_b_score:.2f} "
                f"hit_fire={eval_stats.team_a_hit_fire:.4f}/"
                f"{eval_stats.team_b_hit_fire:.4f} "
                f"aim_err={eval_stats.team_a_aim_error_rad:.3f}/"
                f"{eval_stats.team_b_aim_error_rad:.3f} "
                f"same_tgt={eval_stats.team_a_same_target_fraction:.3f}/"
                f"{eval_stats.team_b_same_target_fraction:.3f} "
                f"focus_H={eval_stats.team_a_target_selection_entropy:.3f}/"
                f"{eval_stats.team_b_target_selection_entropy:.3f} "
                f"pcombat={eval_stats.mean_p_combat:.3f} "
                f"mode_acc={eval_stats.mode_accuracy:.3f}",
                flush=True,
            )
            wandb_logger.log(
                {f"bc_eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()},
                step=0,
            )
            bc_gate = run_cfg.get("bc_combat_gate")
            if bc_gate:
                min_hit_fire = float(bc_gate.get("min_team_a_hit_fire", 0.0))
                max_aim_error = float(bc_gate.get("max_team_a_aim_error_rad", float("inf")))
                passed = (
                    eval_stats.team_a_hit_fire > min_hit_fire
                    and eval_stats.team_a_aim_error_rad < max_aim_error
                )
                print(
                    f"[{phase_label}/mappo] bc_combat_gate "
                    f"passed={passed} "
                    f"team_a_hit_fire={eval_stats.team_a_hit_fire:.4f}"
                    f">{min_hit_fire:.4f} "
                    f"team_a_aim_error={eval_stats.team_a_aim_error_rad:.3f}"
                    f"<{max_aim_error:.3f}",
                    flush=True,
                )
                wandb_logger.log({"bc_eval/combat_gate_passed": float(passed)}, step=0)
                if not passed:
                    total_updates = 0

        class _MappoHooks:
            def set_learning_rate(self, lr: float) -> None:
                trainer.set_learning_rate(lr)

            def collect_rollout(self, update_idx: int):
                return trainer.collect_rollout()

            def update_step(self, update_idx: int, rollout, lr: float):
                tau = compute_team_spirit(
                    update=update_idx,
                    total=total_updates,
                    initial=cfg.team_spirit_initial,
                    final=cfg.team_spirit_final,
                    ramp_fraction=cfg.team_spirit_ramp_fraction,
                )
                trainer.set_team_spirit(tau)
                metrics = trainer.update(rollout)
                metrics["team_spirit"] = tau
                return metrics

            def evaluate_step(self, update_idx: int, lr: float):
                return evaluate_mappo(
                    trainer.model,
                    env_fn,
                    episodes=eval_episodes,
                    seed=seed_base + 100_000 + update_idx,
                )

            def checkpoint_payload(self, update_idx: int) -> dict:
                return {
                    "update_idx": update_idx,
                    "path": output_dir / f"ckpt_{update_idx:04d}.pt",
                }

            def on_log(self, update_idx: int, lr: float, metrics: dict[str, float]) -> None:
                wandb_logger.log(
                    {f"train/{k}": float(v) for k, v in metrics.items()},
                    step=update_idx,
                )
                wandb_logger.log({"train/lr": float(lr)}, step=update_idx)
                print(
                    f"[{phase_label}/mappo] update={update_idx}/{total_updates} "
                    f"policy_loss={metrics['policy_loss']:.3f} "
                    f"value_loss={metrics['value_loss']:.3f} "
                    f"entropy={metrics['entropy']:.3f} "
                    f"rew={metrics['rollout_reward_mean']:+.3f}/"
                    f"{metrics['rollout_reward_std']:.3f} "
                    f"adv={metrics['advantage_mean']:+.3f}/"
                    f"{metrics['advantage_std']:.3f} "
                    f"move={metrics['action_move_mag_mean']:.3f} "
                    f"bin={metrics['action_binary_mean']:.3f} "
                    f"dist={metrics['mean_distance_to_objective']:.3f} "
                    f"onpt={metrics['self_on_point_fraction']:.3f} "
                    f"pcombat={metrics.get('mean_p_combat', 0.0):.3f} "
                    f"mode_acc={metrics.get('mode_accuracy', 0.0):.3f} "
                    f"same_tgt={metrics.get('target_selection_same_target_fraction', 0.0):.3f} "
                    f"focus_H={metrics.get('target_selection_label_entropy', 0.0):.3f} "
                    f"fallback={metrics.get('target_selection_fallback_rate', 0.0):.3f} "
                    f"gn={metrics['actor_grad_norm']:.2e}/"
                    f"{metrics['critic_grad_norm']:.2e}/"
                    f"{metrics['trunk_grad_norm']:.2e} "
                    f"lr={lr:.2e} ts={metrics['team_spirit']:.2f}",
                    flush=True,
                )

            def on_eval(self, update_idx: int, lr: float, eval_stats) -> bool:
                nonlocal last_eval, best_eval, best_state, best_eval_update_idx, best_eval_stats
                last_eval = eval_stats.mean_reward
                print(
                    f"[{phase_label}/mappo] eval update={update_idx}/{total_updates} "
                    f"mean_reward={eval_stats.mean_reward:+.3f} "
                    f"wins={eval_stats.wins}/{eval_stats.episodes} "
                    f"losses={eval_stats.losses}/{eval_stats.episodes} "
                    f"draws={eval_stats.draws}/{eval_stats.episodes} "
                    f"term={eval_stats.terminated} trunc={eval_stats.truncated} "
                    f"tick={eval_stats.mean_final_tick:.1f} "
                    f"score={eval_stats.mean_team_a_score:.2f}/"
                    f"{eval_stats.mean_team_b_score:.2f} "
                    f"kills={eval_stats.mean_team_a_kills:.1f}/"
                    f"{eval_stats.mean_team_b_kills:.1f} "
                    f"hit_fire={eval_stats.team_a_hit_fire:.4f}/"
                    f"{eval_stats.team_b_hit_fire:.4f} "
                    f"vis_fire={eval_stats.team_a_visible_fire_rate:.3f}/"
                    f"{eval_stats.team_b_visible_fire_rate:.3f} "
                    f"aim_err={eval_stats.team_a_aim_error_rad:.3f}/"
                    f"{eval_stats.team_b_aim_error_rad:.3f} "
                    f"target_H={eval_stats.team_a_target_entropy:.3f}/"
                    f"{eval_stats.team_b_target_entropy:.3f} "
                    f"same_tgt={eval_stats.team_a_same_target_fraction:.3f}/"
                    f"{eval_stats.team_b_same_target_fraction:.3f} "
                    f"focus_H={eval_stats.team_a_target_selection_entropy:.3f}/"
                    f"{eval_stats.team_b_target_selection_entropy:.3f} "
                    f"pcombat={eval_stats.mean_p_combat:.3f} "
                    f"mode_acc={eval_stats.mode_accuracy:.3f} "
                    f"intent_fire={eval_stats.intentional_fire_fraction:.3f} "
                    f"obj_focus={eval_stats.objective_focus_fraction:.3f} "
                    f"dmg_fire={eval_stats.team_a_damage_per_fire:.1f}/"
                    f"{eval_stats.team_b_damage_per_fire:.1f}",
                    flush=True,
                )
                wandb_logger.log(
                    {f"eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()},
                    step=update_idx,
                )
                if last_eval > best_eval:
                    best_eval = last_eval
                    best_state = copy.deepcopy(trainer.model.state_dict())
                    best_eval_update_idx = update_idx
                    best_eval_stats = {
                        "wins": int(eval_stats.wins),
                        "losses": int(eval_stats.losses),
                        "draws": int(eval_stats.draws),
                        "score_team_a": float(eval_stats.mean_team_a_score),
                        "score_team_b": float(eval_stats.mean_team_b_score),
                        "kills_team_a": float(eval_stats.mean_team_a_kills),
                        "kills_team_b": float(eval_stats.mean_team_b_kills),
                    }
                if run_cfg.get("eval_gate"):
                    run_eval_gate(
                        phase_label=phase_label,
                        stats=eval_stats,
                        gate_cfg=EvalGateConfig.from_dict(dict(run_cfg.get("eval_gate", {}))),
                        output_dir=output_dir,
                    )
                return False

            def on_checkpoint(self, update_idx: int, payload: dict) -> None:
                checkpoint_path = payload["path"]
                torch.save(
                    {
                        "model_state_dict": trainer.model.state_dict(),
                        "config": {
                            "phase": phase,
                            "env": ckpt_env_cfg,
                            "mappo": cfg.__dict__,
                        },
                    },
                    checkpoint_path,
                )
                if retention is not None:
                    manifest = retention.record_checkpoint(
                        checkpoint_path,
                        update=update_idx,
                        score=last_eval,
                    )
                    print(
                        f"[{phase_label}/mappo] snapshot_pool "
                        f"latest={len(manifest['latest'])} "
                        f"historical={len(manifest['historical'])} "
                        f"anchor={len(manifest['anchor'])}",
                        flush=True,
                    )

        run_training_loop(
            LoopConfig(
                total_updates=total_updates,
                eval_every=eval_every,
                checkpoint_every=checkpoint_every,
                log_every=int(run_cfg.get("log_every", 1)),
                base_lr=cfg.learning_rate,
                lr_schedule=cfg.lr_schedule,
                lr_final_ratio=cfg.lr_final_ratio,
                warmup_updates=cfg.warmup_updates,
            ),
            _MappoHooks(),
        )
    finally:
        trainer.close()
        wandb_logger.finish()
    last_state = trainer.model.state_dict()
    final_state = best_state if best_state is not None else last_state
    ckpt_last_path = (output_dir / "ckpt_last.pt").resolve()
    ckpt_best_eval_path = (output_dir / "ckpt_best_eval.pt").resolve()
    ckpt_final_path = (output_dir / "ckpt_final.pt").resolve()
    manifest_path = (output_dir / "checkpoint_manifest.json").resolve()
    torch.save(
        {
            "model_state_dict": last_state,
            "config": {"phase": phase, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        ckpt_last_path,
    )
    torch.save(
        {
            "model_state_dict": final_state,
            "config": {"phase": phase, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        ckpt_best_eval_path,
    )
    torch.save(
        {
            "model_state_dict": final_state,
            "config": {"phase": phase, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        ckpt_final_path,
    )
    final_alias = "ckpt_best_eval.pt" if best_state is not None else "ckpt_last.pt"
    manifest = {
        "ckpt_final_alias": final_alias,
        "best_eval_update_idx": best_eval_update_idx,
        "best_eval_score": (float(best_eval) if best_eval > float("-inf") else None),
        "best_eval_stats": best_eval_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{phase_label}/mappo] checkpoint_last path={ckpt_last_path} label=last",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_best_eval path={ckpt_best_eval_path} label=best_eval",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_final path={ckpt_final_path} label={final_alias}",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_manifest path={manifest_path}",
        flush=True,
    )
    if run_cfg.get("matrix_eval"):
        matrix_model = MappoActorCritic(cfg)
        matrix_model.load_state_dict(final_state)
        matrix_model.eval()
        rows = run_mappo_matrix_eval(
            model=matrix_model,
            phase=phase,
            ckpt_env_cfg=CheckpointEnvConfig(ckpt_env_cfg),
            matrix_cfg=MatrixEvalConfig.from_dict(dict(run_cfg.get("matrix_eval", {}))),
            output_dir=output_dir,
            seed=seed_base,
        )
        if retention is not None:
            gate: dict | None = None
            matrix_cfg = dict(run_cfg.get("matrix_eval", {}))
            if matrix_cfg.get("gate"):
                gate_path = output_dir / str(matrix_cfg.get("gate_output", "matrix_gate.json"))
                gate = read_json_artifact(gate_path)
            summary = matrix_retention_summary(rows, gate)
            manifest = retention.record_checkpoint(
                output_dir / "ckpt_final.pt",
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
                f"[{phase_label}/mappo] snapshot_pool_matrix "
                f"score={float(summary['matrix_score']):+.3f} "
                f"gate={_matrix_gate_label(summary['matrix_gate_passed'])} "
                f"latest={len(manifest['latest'])} "
                f"historical={len(manifest['historical'])} "
                f"anchor={len(manifest['anchor'])}",
                flush=True,
            )
    return {"mappo": best_eval if best_eval > float("-inf") else last_eval}


# Compatibility re-exports (temporary)
_eval_stats_dict = eval_stats_dict
_run_eval_gate = run_eval_gate
_run_mappo_matrix_eval = run_mappo_matrix_eval
_matrix_retention_summary = matrix_retention_summary
_matrix_gate_label = matrix_gate_label
