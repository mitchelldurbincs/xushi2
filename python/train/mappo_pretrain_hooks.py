from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from train.composition_rehearsal import (
    build_mappo_env_fn_with_overrides,
    composition_rehearsal_pretrain,
    load_frozen_mappo_teacher,
    run_composition_diagnostics,
)
from train.full_env_rehearsal import (
    closed_loop_supervised_bridge_pretrain,
    full_env_rehearsal_pretrain,
    run_full_env_rehearsal_gate,
)
from train.mappo_bc_pretrain import (
    bc_pretrain_walk_and_shoot_to_objective,
    bc_pretrain_walk_to_objective,
)
from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_rollout_trainer import MappoTrainer
from train.mappo_runtime_context import RuntimeContext

_WARM_START_MIGRATION_COMPATIBLE_EXACT = "compatible_exact"


# Parameters a migration never carries over. Migration exists for
# architecture changes whose new tensors must be re-derived by fresh
# exploration; the checkpoint's log_std is exploration tuned for the OLD
# tensors. Probe 2026-08-09 (phase4_cap_headwidth_probe): migrating
# v5-0300's sharpened log_std (std 0.113) onto fresh heads killed
# exploration completely — 300 updates flat at -20 reward, the policy
# never once reached the objective. The config's action_log_std_init is
# the intended exploration level for re-derivation; keep it.
_MIGRATION_RESET_KEYS = frozenset({"log_std"})


def _load_compatible_warm_start(
    model: nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_unexpected: list[str] = []
    skipped_shape_mismatch: list[str] = []
    skipped_reset: list[str] = []

    for key, value in checkpoint_state.items():
        if key in _MIGRATION_RESET_KEYS:
            skipped_reset.append(key)
            continue
        target_value = model_state.get(key)
        if target_value is None:
            skipped_unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_value.shape):
            skipped_shape_mismatch.append(
                f"{key}(checkpoint={tuple(value.shape)}, model={tuple(target_value.shape)})"
            )
            continue
        compatible[key] = value

    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "missing": sorted(load_result.missing_keys),
        "skipped_unexpected": sorted(skipped_unexpected),
        "skipped_shape_mismatch": sorted(skipped_shape_mismatch),
        "skipped_reset": sorted(skipped_reset),
    }


def maybe_warm_start(context: RuntimeContext, trainer: MappoTrainer) -> None:
    run_cfg = context.run_cfg
    init_ckpt = run_cfg.get("init_from_checkpoint")
    if not init_ckpt:
        return
    cfg = context.cfg
    phase_label = context.phase_label
    raw = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    warm_start_migration = run_cfg.get("warm_start_migration")
    if warm_start_migration:
        if warm_start_migration != _WARM_START_MIGRATION_COMPATIBLE_EXACT:
            raise ValueError(f"unsupported warm_start_migration={warm_start_migration!r}")
        report = _load_compatible_warm_start(trainer.model, raw["model_state_dict"])
        print(
            f"[{phase_label}/mappo] warm-start migration={warm_start_migration} "
            f"loaded={len(report['loaded'])} missing={report['missing']} "
            f"skipped_unexpected={report['skipped_unexpected']} "
            f"skipped_shape_mismatch={report['skipped_shape_mismatch']} "
            f"skipped_reset={report['skipped_reset']}",
            flush=True,
        )
        print(f"[{phase_label}/mappo] warm-start: loaded {init_ckpt}", flush=True)
        return
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
    print(f"[{phase_label}/mappo] warm-start: loaded {init_ckpt}", flush=True)


def maybe_resume(context: RuntimeContext, trainer: MappoTrainer) -> int:
    """Continue a previous run from ``run.resume_from``. Returns the next update.

    Distinct from ``run.init_from_checkpoint`` (warm start), which deliberately
    starts a *new* run from someone else's weights. Resuming restores the
    optimizer moments, RNG streams, recurrent hidden state, and update index, so
    the LR schedule and exploration continue rather than restarting.

    Scope: the learner's state is restored, the environment's is not. The C++
    Sim cannot be serialized from Python, so the first resumed update begins
    from a freshly reset episode rather than mid-episode. A resumed run
    therefore does not reproduce an uninterrupted one bit for bit; it continues
    optimization correctly, which is the part that a weights-only restart got
    wrong.
    """
    resume_path = context.run_cfg.get("resume_from")
    if not resume_path:
        return 1
    if context.run_cfg.get("init_from_checkpoint"):
        raise ValueError(
            "run.resume_from and run.init_from_checkpoint are mutually exclusive: "
            "the first continues this run, the second starts a new one from "
            "foreign weights. Pick one."
        )
    path = Path(resume_path)
    if not path.is_file():
        raise FileNotFoundError(f"run.resume_from checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(raw["model_state_dict"], strict=True)
    resume_state = raw.get("resume_state")
    if resume_state is None:
        raise ValueError(
            f"{path} has no resume_state, so it can only be warm-started from. "
            "It was written before resumable checkpoints existed, or by a code "
            "path that does not record optimizer/RNG state. Use "
            "run.init_from_checkpoint if a fresh optimizer is acceptable."
        )
    completed = trainer.load_resume_state(resume_state)
    print(
        f"[{context.phase_label}/mappo] resume: {path} at update {completed}; "
        f"continuing at {completed + 1}",
        flush=True,
    )
    return completed + 1


def maybe_run_composition_pretrain(
    context: RuntimeContext, trainer: MappoTrainer, logger: Any
) -> bool:
    run_cfg = context.run_cfg
    if not bool(run_cfg.get("composition_pretrain", False)):
        return True
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
    objective_env_fn = build_mappo_env_fn_with_overrides(
        context.ckpt_env_cfg,
        dict(run_cfg.get("composition_objective_env", {})),
    )
    combat_env_fn = build_mappo_env_fn_with_overrides(
        context.ckpt_env_cfg,
        dict(run_cfg.get("composition_combat_env", {})),
    )
    full_eval_env_fn = build_mappo_env_fn_with_overrides(
        context.ckpt_env_cfg,
        {"opponent_bot": "weak_basic_v2", "mini_game": None, "mini_game_config": {}},
    )
    metrics = composition_rehearsal_pretrain(
        trainer.model,
        objective_teacher,
        combat_teacher,
        objective_env_fn,
        combat_env_fn,
        {
            "steps": int(run_cfg.get("composition_pretrain_steps", 1000)),
            "objective_batch_size": int(run_cfg.get("composition_objective_batch_size", 256)),
            "combat_batch_size": int(run_cfg.get("composition_combat_batch_size", 256)),
            "learning_rate": float(
                run_cfg.get("composition_learning_rate", run_cfg.get("bc_learning_rate", 1.0e-3))
            ),
            "seed": context.seed_base + 40_000,
            "log_label": context.phase_label,
        },
    )
    if metrics:
        logger.log({f"composition_pretrain/{k}": float(v) for k, v in metrics.items()}, step=0)
    diagnostics = run_composition_diagnostics(
        trainer.model,
        objective_env_fn=objective_env_fn,
        combat_env_fn=combat_env_fn,
        full_env_fn=full_eval_env_fn,
        episodes=int(run_cfg.get("composition_eval_episodes", context.eval_episodes)),
        seed=context.seed_base + 80_000,
        gate=dict(run_cfg.get("composition_gate", {})),
    )
    gate_metrics = diagnostics.metrics
    print(
        f"[{context.phase_label}/mappo] composition_gate "
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
    logger.log({f"composition_eval/{k}": float(v) for k, v in diagnostics.metrics.items()}, step=0)
    return diagnostics.passed


def maybe_run_full_env_rehearsal(
    context: RuntimeContext, trainer: MappoTrainer, logger: Any
) -> bool:
    run_cfg = context.run_cfg
    rehearsal_cfg = dict(run_cfg.get("full_env_rehearsal", {}))
    if not bool(rehearsal_cfg.get("enabled", False)):
        return True
    metrics = full_env_rehearsal_pretrain(
        trainer.model,
        context.env_fn,
        {
            **rehearsal_cfg,
            "seed": int(rehearsal_cfg.get("seed", context.seed_base + 60_000)),
            "log_label": context.phase_label,
        },
    )
    if metrics:
        logger.log({f"full_env_rehearsal/{k}": float(v) for k, v in metrics.items()}, step=0)
    checkpoint_path = context.output_dir / "ckpt_full_env_rehearsal.pt"
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
        gate=dict(rehearsal_cfg.get("gate", {})),
        output_dir=context.output_dir,
        seed=context.seed_base + 95_000,
        checkpoint_path=checkpoint_path,
    )
    print(
        f"[{context.phase_label}/mappo] full_env_rehearsal_gate "
        f"status={gate.status} "
        f"team_a_hit_fire={gate.metrics['team_a_hit_fire']:.4f}>="
        f"{gate.thresholds['min_team_a_hit_fire']:.4f} "
        f"objective_on_point={gate.metrics['objective_on_point']:.3f}>="
        f"{gate.thresholds['min_objective_on_point']:.3f} "
        f"losses={gate.metrics['losses']:.0f}<={gate.thresholds['max_losses']:.0f} "
        f"artifact={gate.path}",
        flush=True,
    )
    logger.log(
        {f"full_env_rehearsal_gate/{k}": float(v) for k, v in gate.metrics.items()},
        step=0,
    )
    logger.log({"full_env_rehearsal_gate/passed": float(gate.passed)}, step=0)
    return gate.passed


def maybe_run_multi_enemy_supervised_bridge(
    context: RuntimeContext, trainer: MappoTrainer, logger: Any
) -> bool:
    run_cfg = context.run_cfg
    bridge_cfg = dict(run_cfg.get("multi_enemy_supervised_bridge", {}))
    if not bool(bridge_cfg.get("enabled", False)):
        return True
    teacher = str(bridge_cfg.get("teacher", "multi_enemy_visible"))
    if teacher not in ("multi_enemy_visible", "multi_enemy_conversion_hold"):
        raise ValueError(
            "run.multi_enemy_supervised_bridge.teacher must be 'multi_enemy_visible' "
            "or 'multi_enemy_conversion_hold'"
        )
    if context.cfg.obs_encoder != "entity_attention_grid":
        raise ValueError("multi_enemy_supervised_bridge requires entity_attention_grid obs")
    if context.cfg.target_action_dim != 0:
        raise ValueError("multi_enemy_supervised_bridge must not add target action fields")
    pretrain_cfg = {
        **bridge_cfg,
        "seed": int(bridge_cfg.get("seed", context.seed_base + 70_000)),
        "log_label": context.phase_label,
    }
    if bool(dict(bridge_cfg.get("closed_loop", {})).get("enabled", False)):
        metrics = closed_loop_supervised_bridge_pretrain(
            trainer.model,
            context.env_fn,
            pretrain_cfg,
        )
    else:
        metrics = full_env_rehearsal_pretrain(
            trainer.model,
            context.env_fn,
            pretrain_cfg,
        )
    if metrics:
        logger.log(
            {f"multi_enemy_supervised_bridge/{k}": float(v) for k, v in metrics.items()},
            step=0,
        )
    checkpoint_path = context.output_dir / "ckpt_multi_enemy_supervised_bridge.pt"
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
    print(
        f"[{context.phase_label}/mappo] multi_enemy_supervised_bridge_gate "
        f"status={gate.status} "
        f"team_a_visible_fire_rate={gate.metrics['team_a_visible_fire_rate']:.4f}>="
        f"{gate.thresholds['min_team_a_visible_fire_rate']:.4f} "
        f"team_a_hit_fire={gate.metrics['team_a_hit_fire']:.4f}>="
        f"{gate.thresholds['min_team_a_hit_fire']:.4f} "
        f"objective_on_point={gate.metrics['objective_on_point']:.3f}>="
        f"{gate.thresholds['min_objective_on_point']:.3f} "
        f"mean_score_a={gate.metrics['mean_score_a']:.2f}>="
        f"{gate.thresholds['min_mean_score_a']:.2f} "
        f"losses={gate.metrics['losses']:.0f}<={gate.thresholds['max_losses']:.0f} "
        f"artifact={gate.path}",
        flush=True,
    )
    logger.log(
        {f"multi_enemy_supervised_bridge_gate/{k}": float(v) for k, v in gate.metrics.items()},
        step=0,
    )
    logger.log({"multi_enemy_supervised_bridge_gate/passed": float(gate.passed)}, step=0)
    return gate.passed


def maybe_run_bc_pretrain(context: RuntimeContext, trainer: MappoTrainer, logger: Any) -> bool:
    run_cfg = context.run_cfg
    bc_steps = int(run_cfg.get("bc_pretrain_steps", 0))
    if bc_steps <= 0:
        return True
    bc_variant = str(run_cfg.get("bc_pretrain_variant", "walk_to_objective"))
    bc_kwargs = {
        "steps": bc_steps,
        "batch_size": int(run_cfg.get("bc_batch_size", 1024)),
        "learning_rate": float(run_cfg.get("bc_learning_rate", 1.0e-3)),
        "seed": context.seed_base + 50_000,
        "log_label": context.phase_label,
        "freeze_actor_aim": bool(run_cfg.get("bc_freeze_actor_aim", False)),
    }
    if bc_variant == "walk_and_shoot":
        aim_rehearsal_env_fn = None
        aim_rehearsal_batch_size = int(run_cfg.get("bc_aim_rehearsal_batch_size", 0))
        if aim_rehearsal_batch_size > 0:
            from envs import Phase4AimOnlyMappoEnv

            target_ckpt = run_cfg.get("bc_aim_target_checkpoint")
            if not target_ckpt:
                raise ValueError("bc_aim_rehearsal_batch_size requires bc_aim_target_checkpoint")
            target_raw = torch.load(target_ckpt, map_location="cpu", weights_only=False)
            mini_game_cfg = dict(
                target_raw.get("config", {}).get("env", {}).get("mini_game_config", {})
            )
            if not mini_game_cfg:
                raise ValueError("bc_aim_target_checkpoint does not contain env.mini_game_config")

            def aim_rehearsal_env_fn() -> Phase4AimOnlyMappoEnv:
                return Phase4AimOnlyMappoEnv(**mini_game_cfg)

        bc_pretrain_walk_and_shoot_to_objective(
            trainer.model,
            context.env_fn,
            context.cfg,
            aim_target_checkpoint=run_cfg.get("bc_aim_target_checkpoint"),
            aim_rehearsal_env_fn=aim_rehearsal_env_fn,
            aim_rehearsal_batch_size=aim_rehearsal_batch_size,
            **bc_kwargs,
        )
    else:
        bc_pretrain_walk_to_objective(trainer.model, context.env_fn, context.cfg, **bc_kwargs)
    eval_stats = evaluate_mappo(
        trainer.model,
        context.env_fn,
        episodes=context.eval_episodes,
        seed=context.seed_base + 90_000,
    )
    print(
        f"[{context.phase_label}/mappo] bc_eval mean_reward={eval_stats.mean_reward:+.3f} "
        f"wins={eval_stats.wins}/{eval_stats.episodes} "
        f"draws={eval_stats.draws}/{eval_stats.episodes} "
        f"score={eval_stats.mean_team_a_score:.2f}/{eval_stats.mean_team_b_score:.2f} "
        f"hit_fire={eval_stats.team_a_hit_fire:.4f}/{eval_stats.team_b_hit_fire:.4f} "
        f"aim_err={eval_stats.team_a_aim_error_rad:.3f}/{eval_stats.team_b_aim_error_rad:.3f} "
        f"same_tgt={eval_stats.team_a_same_target_fraction:.3f}/"
        f"{eval_stats.team_b_same_target_fraction:.3f} "
        f"focus_H={eval_stats.team_a_target_selection_entropy:.3f}/"
        f"{eval_stats.team_b_target_selection_entropy:.3f} "
        f"pcombat={eval_stats.mean_p_combat:.3f} mode_acc={eval_stats.mode_accuracy:.3f}",
        flush=True,
    )
    logger.log({f"bc_eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()}, step=0)
    bc_gate = run_cfg.get("bc_combat_gate")
    if not bc_gate:
        return True
    min_hit_fire = float(bc_gate.get("min_team_a_hit_fire", 0.0))
    max_aim_error = float(bc_gate.get("max_team_a_aim_error_rad", float("inf")))
    passed = (
        eval_stats.team_a_hit_fire > min_hit_fire
        and eval_stats.team_a_aim_error_rad < max_aim_error
    )
    print(
        f"[{context.phase_label}/mappo] bc_combat_gate passed={passed} "
        f"team_a_hit_fire={eval_stats.team_a_hit_fire:.4f}>{min_hit_fire:.4f} "
        f"team_a_aim_error={eval_stats.team_a_aim_error_rad:.3f}<{max_aim_error:.3f}",
        flush=True,
    )
    logger.log({"bc_eval/combat_gate_passed": float(passed)}, step=0)
    return passed
