from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from train.mappo_bc_pretrain import bc_pretrain_walk_to_objective
from train.mappo_model import (
    MappoActorCritic,
    MappoEvalStats,
    _eval_outcome_counts,
    compute_team_spirit,
)
from train.mappo_rollout_trainer import MappoTrainer, make_mappo_config
from train.phases import resolve_phase
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.wandb_logger import make_logger
from train.mappo_eval_gate_io import EvalGateConfig, read_json_artifact, run_eval_gate, write_json_artifact
from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_matrix_eval import CheckpointEnvConfig, MatrixEvalConfig, matrix_gate_label, matrix_retention_summary, run_mappo_matrix_eval
from xushi2.snapshot_retention import SnapshotRetention


def evaluate_mappo(
    model: MappoActorCritic,
    env_fn: Callable[[], gym.Env],
    episodes: int,
    seed: int,
) -> MappoEvalStats:
    was_training = model.training
    model.eval()
    rewards: list[float] = []
    final_ticks: list[int] = []
    team_a_scores: list[float] = []
    team_b_scores: list[float] = []
    team_a_kills: list[int] = []
    team_b_kills: list[int] = []
    wins = 0
    losses = 0
    draws = 0
    terminated_count = 0
    truncated_count = 0
    for i in range(int(episodes)):
        env = env_fn()
        try:
            obs, _info = env.reset(seed=int(seed) + i)
            h = model.init_hidden(model.cfg.n_agents)
            done = False
            term = False
            trunc = False
            ep_reward = 0.0
            info = {}
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, h = model.greedy_action(obs_t, h)
                obs, reward, term, trunc, info = env.step(action.cpu().numpy())
                ep_reward += float(np.mean(reward))
                done = bool(term or trunc)
            rewards.append(ep_reward)

            winner = str(info.get("winner", ""))
            learner_team = str(info.get("learner_team", ""))
            won, lost, drew = _eval_outcome_counts(
                winner=winner,
                learner_team=learner_team,
                truncated=bool(trunc),
            )
            wins += won
            losses += lost
            draws += drew

            terminated_count += int(bool(term))
            truncated_count += int(bool(trunc))
            final_ticks.append(int(info.get("tick", 0)))
            team_a_scores.append(float(info.get("team_a_score", 0.0)))
            team_b_scores.append(float(info.get("team_b_score", 0.0)))
            team_a_kills.append(int(info.get("team_a_kills", 0)))
            team_b_kills.append(int(info.get("team_b_kills", 0)))
        finally:
            env.close()
    if was_training:
        model.train()
    return MappoEvalStats(
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        episodes=len(rewards),
        wins=wins,
        losses=losses,
        draws=draws,
        terminated=terminated_count,
        truncated=truncated_count,
        mean_final_tick=float(np.mean(final_ticks)) if final_ticks else 0.0,
        mean_team_a_score=float(np.mean(team_a_scores)) if team_a_scores else 0.0,
        mean_team_b_score=float(np.mean(team_b_scores)) if team_b_scores else 0.0,
        mean_team_a_kills=float(np.mean(team_a_kills)) if team_a_kills else 0.0,
        mean_team_b_kills=float(np.mean(team_b_kills)) if team_b_kills else 0.0,
    )


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
        trainer.model.load_state_dict(raw["model_state_dict"], strict=True)
        print(
            f"[{phase_label}/mappo] warm-start: loaded {init_ckpt}",
            flush=True,
        )

    best_eval = float("-inf")
    best_state: dict | None = None
    last_eval = float("nan")
    try:
        bc_steps = int(run_cfg.get("bc_pretrain_steps", 0))
        if bc_steps > 0:
            bc_pretrain_walk_to_objective(
                trainer.model,
                env_fn,
                cfg,
                steps=bc_steps,
                batch_size=int(run_cfg.get("bc_batch_size", 1024)),
                learning_rate=float(run_cfg.get("bc_learning_rate", 1.0e-3)),
                seed=seed_base + 50_000,
                log_label=phase_label,
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
            print(
                f"[{phase_label}/mappo] bc_eval "
                f"mean_reward={eval_stats.mean_reward:+.3f} "
                f"wins={eval_stats.wins}/{eval_stats.episodes} "
                f"draws={eval_stats.draws}/{eval_stats.episodes} "
                f"score={eval_stats.mean_team_a_score:.2f}/"
                f"{eval_stats.mean_team_b_score:.2f}",
                flush=True,
            )
            wandb_logger.log(
                {f"bc_eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()},
                step=0,
            )
        for update_idx in range(1, total_updates + 1):
            lr = lr_for_update(
                update_idx,
                total_updates,
                base_lr=cfg.learning_rate,
                schedule=cfg.lr_schedule,
                lr_final_ratio=cfg.lr_final_ratio,
                warmup_updates=cfg.warmup_updates,
            )
            trainer.set_learning_rate(lr)
            tau = compute_team_spirit(
                update=update_idx,
                total=total_updates,
                initial=cfg.team_spirit_initial,
                final=cfg.team_spirit_final,
                ramp_fraction=cfg.team_spirit_ramp_fraction,
            )
            trainer.set_team_spirit(tau)
            metrics = trainer.update(trainer.collect_rollout())
            metrics["team_spirit"] = tau
            wandb_logger.log(
                {f"train/{k}": float(v) for k, v in metrics.items()},
                step=update_idx,
            )
            wandb_logger.log({"train/lr": float(lr)}, step=update_idx)
            if update_idx % int(run_cfg.get("log_every", 1)) == 0:
                print(
                    f"[{phase_label}/mappo] update={update_idx}/{total_updates} "
                    f"policy_loss={metrics['policy_loss']:.3f} "
                    f"value_loss={metrics['value_loss']:.3f} "
                    f"entropy={metrics['entropy']:.3f} "
                    f"rew={metrics['rollout_reward_mean']:+.3f}"
                    f"/{metrics['rollout_reward_std']:.3f} "
                    f"adv={metrics['advantage_mean']:+.3f}"
                    f"/{metrics['advantage_std']:.3f} "
                    f"move={metrics['action_move_mag_mean']:.3f} "
                    f"bin={metrics['action_binary_mean']:.3f} "
                    f"dist={metrics['mean_distance_to_objective']:.3f} "
                    f"onpt={metrics['self_on_point_fraction']:.3f} "
                    f"gn={metrics['actor_grad_norm']:.2e}/"
                    f"{metrics['critic_grad_norm']:.2e}/"
                    f"{metrics['trunk_grad_norm']:.2e} "
                    f"lr={lr:.2e} "
                    f"ts={metrics['team_spirit']:.2f}",
                    flush=True,
                )
            if update_idx % eval_every == 0 or update_idx == total_updates:
                eval_stats = evaluate_mappo(
                    trainer.model,
                    env_fn,
                    episodes=eval_episodes,
                    seed=seed_base + 100_000 + update_idx,
                )
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
                    f"{eval_stats.mean_team_b_kills:.1f}",
                    flush=True,
                )
                wandb_logger.log(
                    {f"eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()},
                    step=update_idx,
                )
                if last_eval > best_eval:
                    best_eval = last_eval
                    best_state = copy.deepcopy(trainer.model.state_dict())
                if run_cfg.get("eval_gate"):
                    run_eval_gate(
                        phase_label=phase_label,
                        stats=eval_stats,
                        gate_cfg=EvalGateConfig.from_dict(dict(run_cfg.get("eval_gate", {}))),
                        output_dir=output_dir,
                    )
            if update_idx % checkpoint_every == 0 or update_idx == total_updates:
                checkpoint_path = output_dir / f"ckpt_{update_idx:04d}.pt"
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
    finally:
        trainer.close()
        wandb_logger.finish()
    final_state = best_state if best_state is not None else trainer.model.state_dict()
    torch.save(
        {
            "model_state_dict": final_state,
            "config": {"phase": phase, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        output_dir / "ckpt_final.pt",
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
