from __future__ import annotations

from train.cap_duel_distill import configure_cap_duel_distill_anchor
from train.common_orchestration import LoopConfig, run_training_loop
from train.mappo_checkpoint_outputs import save_final_mappo_checkpoints
from train.mappo_post_training import maybe_run_post_training_matrix_eval
from train.mappo_pretrain_hooks import (
    maybe_run_bc_pretrain,
    maybe_run_composition_pretrain,
    maybe_run_full_env_rehearsal,
    maybe_run_multi_enemy_supervised_bridge,
    maybe_warm_start,
)
from train.mappo_rollout_trainer import MappoTrainer
from train.mappo_runtime_context import build_runtime_context
from train.runtime_adapter import resolve_runtime_env_factory
from train.mappo_training_hooks import MappoTrainingHooks
from train.wandb_logger import make_logger


def train_mappo_from_config(config: dict) -> dict[str, float]:
    resolve_runtime_env_factory(config, require_learner="mappo", context="MAPPO train")
    context = build_runtime_context(config)
    phase = context.phase
    phase_label = context.phase_label
    ckpt_env_cfg = context.ckpt_env_cfg
    seed_base = context.seed_base
    cfg = context.cfg
    run_cfg = context.run_cfg
    total_updates = context.total_updates
    eval_every = context.eval_every
    eval_episodes = context.eval_episodes
    checkpoint_every = context.checkpoint_every

    trainer = MappoTrainer(context.env_fn, cfg, seed=seed_base)

    wandb_logger = make_logger(
        config=config,
        run_name=f"{phase_label}_mappo_seed{seed_base}",
        run_config={
            "phase": phase,
            "phase_label": phase_label,
            "variant": "mappo",
            "seed": int(seed_base),
            "total_updates": total_updates,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "mappo": dict(cfg.__dict__),
            "env": dict(ckpt_env_cfg),
        },
        tags=[
            *( [f"phase{int(phase)}"] if phase is not None else [] ),
            phase_label,
            "mappo",
        ],
    )

    # Warm-start: optionally load a previously-trained checkpoint into the
    # newly-constructed trainer's model. Used by the Phase 4 cap-training
    # escalation (docs/plans/2026-05-08-phase4-cap-training-escalation.md)
    # to seed a basic-opponent run from a noop-trained policy that already
    # knows how to hold the cap. Loaded BEFORE BC pretrain so BC, if also
    # configured, fine-tunes on top of the warm-started weights.
    maybe_warm_start(context, trainer)
    configure_cap_duel_distill_anchor(context, trainer)

    hooks = MappoTrainingHooks(
        context=context,
        trainer=trainer,
        wandb_logger=wandb_logger,
        total_updates=total_updates,
    )
    try:
        rehearsal_gate_passed = maybe_run_full_env_rehearsal(context, trainer, wandb_logger)
        if not rehearsal_gate_passed:
            total_updates = 0
        bridge_gate_passed = True
        if rehearsal_gate_passed:
            bridge_gate_passed = maybe_run_multi_enemy_supervised_bridge(
                context, trainer, wandb_logger
            )
        if not bridge_gate_passed:
            total_updates = 0
        composition_gate_passed = True
        if rehearsal_gate_passed and bridge_gate_passed:
            composition_gate_passed = maybe_run_composition_pretrain(
                context, trainer, wandb_logger
            )
        if not composition_gate_passed:
            total_updates = 0
        if rehearsal_gate_passed and bridge_gate_passed and composition_gate_passed:
            bc_gate_passed = maybe_run_bc_pretrain(context, trainer, wandb_logger)
            if not bc_gate_passed:
                total_updates = 0
        hooks.total_updates = total_updates

        # Anchor-KL reference: freeze the policy as it stands at PPO start
        # (after warm start and any BC/bridge pretrain stage), so PPO is
        # penalized for drifting away from the behavior it inherited.
        if cfg.anchor_kl_coef > 0.0 and total_updates > 0:
            trainer.init_anchor_from_current_model()
            print(
                f"[{phase_label}/mappo] anchor_kl: froze PPO-start policy as anchor "
                f"(coef={cfg.anchor_kl_coef}, anneal_updates={cfg.anchor_kl_anneal_updates})",
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
            hooks,
        )
    finally:
        trainer.close()
        wandb_logger.finish()
    last_state = trainer.model.state_dict()
    final_state = hooks.best_state if hooks.best_state is not None else last_state
    checkpoint_outputs = save_final_mappo_checkpoints(
        output_dir=context.output_dir,
        last_state=last_state,
        final_state=final_state,
        has_best_state=hooks.best_state is not None,
        phase=phase,
        phase_label=phase_label,
        ckpt_env_cfg=ckpt_env_cfg,
        mappo_cfg=cfg,
        best_eval_update_idx=hooks.best_eval_update_idx,
        best_eval=hooks.best_eval,
        best_eval_stats=hooks.best_eval_stats,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_last path={checkpoint_outputs.last_path} label=last",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_best_eval "
        f"path={checkpoint_outputs.best_eval_path} label=best_eval",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_final "
        f"path={checkpoint_outputs.final_path} label={checkpoint_outputs.final_alias}",
        flush=True,
    )
    print(
        f"[{phase_label}/mappo] checkpoint_manifest path={checkpoint_outputs.manifest_path}",
        flush=True,
    )
    maybe_run_post_training_matrix_eval(
        context=context,
        final_state=final_state,
        best_eval=hooks.best_eval,
        last_eval=hooks.last_eval,
        total_updates=total_updates,
    )
    return {"mappo": hooks.best_eval if hooks.best_eval > float("-inf") else hooks.last_eval}


# Compatibility alias for existing scripts/tests/checkpoints.
train_phase4_from_config = train_mappo_from_config
