from __future__ import annotations

import copy
import json
from typing import Any

from train.mappo_checkpoint_outputs import save_mappo_checkpoint
from train.mappo_eval_gate_io import EvalGateConfig, run_eval_gate
from train.mappo_evaluate import eval_stats_dict, evaluate_mappo
from train.mappo_model import (
    compute_majority_on_point_alpha,
    compute_objective_timing_seconds,
    compute_team_spirit,
)
from train.mappo_rollout_trainer import MappoTrainer
from train.mappo_runtime_context import RuntimeContext


class MappoTrainingHooks:
    def __init__(
        self,
        *,
        context: RuntimeContext,
        trainer: MappoTrainer,
        wandb_logger: Any,
        total_updates: int,
    ) -> None:
        self.context = context
        self.trainer = trainer
        self.wandb_logger = wandb_logger
        self.total_updates = total_updates

        self.best_eval = float("-inf")
        self.best_state: dict | None = None
        self.best_eval_update_idx: int | None = None
        self.best_eval_stats: dict[str, float | int] | None = None
        self.last_eval = float("nan")

        self._last_team_spirit = 0.0
        self._last_majority_on_point_alpha = 0.0
        self._last_uncontested_on_point_alpha = 0.0
        self._last_objective_timing_seconds: tuple[float, float] | None = None
        self._last_canonical_eval_stats = None

    def set_learning_rate(self, lr: float) -> None:
        self.trainer.set_learning_rate(lr)

    def _objective_timing_for_update(self, update_idx: int) -> tuple[float, float] | None:
        if not self.context.objective_timing_enabled:
            return None
        return compute_objective_timing_seconds(
            update=update_idx,
            initial_unlock_seconds=self.context.objective_initial_unlock_seconds,
            final_unlock_seconds=self.context.objective_final_unlock_seconds,
            initial_capture_seconds=self.context.objective_initial_capture_seconds,
            final_capture_seconds=self.context.objective_final_capture_seconds,
            anneal_updates=self.context.objective_timing_anneal_updates,
        )

    def collect_rollout(self, update_idx: int):
        cfg = self.context.cfg
        tau = compute_team_spirit(
            update=update_idx,
            total=self.total_updates,
            initial=cfg.team_spirit_initial,
            final=cfg.team_spirit_final,
            ramp_fraction=cfg.team_spirit_ramp_fraction,
        )
        alpha = compute_majority_on_point_alpha(
            update=update_idx,
            initial=self.context.majority_on_point_initial,
            anneal_updates=self.context.majority_on_point_anneal_updates,
        )
        uncontested_alpha = compute_majority_on_point_alpha(
            update=update_idx,
            initial=self.context.uncontested_on_point_initial,
            anneal_updates=self.context.uncontested_on_point_anneal_updates,
        )
        self.trainer.set_team_spirit(tau)
        self.trainer.set_majority_on_point_alpha(alpha)
        self.trainer.set_uncontested_on_point_alpha(uncontested_alpha)
        timing = self._objective_timing_for_update(update_idx)
        if timing is not None:
            self.trainer.set_objective_timing_seconds(timing[0], timing[1])
        self._last_team_spirit = tau
        self._last_majority_on_point_alpha = alpha
        self._last_uncontested_on_point_alpha = uncontested_alpha
        self._last_objective_timing_seconds = timing
        return self.trainer.collect_rollout()

    def update_step(self, update_idx: int, rollout, lr: float):
        self.trainer.set_update_index(update_idx)
        metrics = self.trainer.update(rollout)
        metrics["team_spirit"] = self._last_team_spirit
        metrics["majority_on_point_alpha"] = self._last_majority_on_point_alpha
        metrics["uncontested_on_point_alpha"] = self._last_uncontested_on_point_alpha
        if self._last_objective_timing_seconds is not None:
            metrics["objective_unlock_seconds"] = self._last_objective_timing_seconds[0]
            metrics["objective_capture_seconds"] = self._last_objective_timing_seconds[1]
        return metrics

    def evaluate_step(self, update_idx: int, lr: float):
        timing = self._objective_timing_for_update(update_idx)
        eval_stats = evaluate_mappo(
            self.trainer.model,
            self.context.env_fn,
            episodes=self.context.eval_episodes,
            seed=self.context.seed_base + 100_000 + update_idx,
            objective_timing_seconds=timing,
        )
        self._last_canonical_eval_stats = None
        if (
            self.context.objective_timing_enabled
            and self.context.objective_eval_canonical_every > 0
            and update_idx % self.context.objective_eval_canonical_every == 0
        ):
            self._last_canonical_eval_stats = evaluate_mappo(
                self.trainer.model,
                self.context.env_fn,
                episodes=self.context.eval_episodes,
                seed=self.context.seed_base + 200_000 + update_idx,
                objective_timing_seconds=(15.0, 8.0),
            )
        return eval_stats

    def checkpoint_payload(self, update_idx: int) -> dict:
        return {
            "update_idx": update_idx,
            "path": self.context.output_dir / f"ckpt_{update_idx:04d}.pt",
        }

    def on_log(self, update_idx: int, lr: float, metrics: dict[str, float]) -> None:
        phase_label = self.context.phase_label
        train_metrics = {
            key: value for key, value in metrics.items() if not key.startswith("distill/")
        }
        distill_metrics = {
            key: value for key, value in metrics.items() if key.startswith("distill/")
        }
        self.wandb_logger.log(
            {f"train/{k}": float(v) for k, v in train_metrics.items()},
            step=update_idx,
        )
        if distill_metrics:
            self.wandb_logger.log(
                {key: float(value) for key, value in distill_metrics.items()},
                step=update_idx,
            )
        self.wandb_logger.log({"train/lr": float(lr)}, step=update_idx)
        objective_unlock_log = metrics.get(
            "objective_unlock_seconds",
            metrics.get("rollout_objective_unlock_seconds_mean", 0.0),
        )
        objective_capture_log = metrics.get(
            "objective_capture_seconds",
            metrics.get("rollout_objective_capture_seconds_mean", 0.0),
        )
        print(
            f"[{phase_label}/mappo] update={update_idx}/{self.total_updates} "
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
            f"obj_t={objective_unlock_log:.2f}/"
            f"{objective_capture_log:.2f} "
            f"maj_alpha={metrics.get('majority_on_point_alpha', 0.0):.4f} "
            f"maj_rew={metrics.get('rollout_majority_on_point_reward_a_mean', 0.0):+.4f}/"
            f"{metrics.get('rollout_majority_on_point_reward_b_mean', 0.0):+.4f} "
            f"unc_alpha={metrics.get('uncontested_on_point_alpha', 0.0):.4f} "
            "unc_rew="
            f"{metrics.get('rollout_uncontested_on_point_reward_a_mean', 0.0):+.4f}/"
            f"{metrics.get('rollout_uncontested_on_point_reward_b_mean', 0.0):+.4f} "
            f"maj_sec={metrics.get('rollout_majority_on_point_seconds_a_mean', 0.0):.3f}/"
            f"{metrics.get('rollout_majority_on_point_seconds_b_mean', 0.0):.3f} "
            f"pcombat={metrics.get('mean_p_combat', 0.0):.3f} "
            f"mode_acc={metrics.get('mode_accuracy', 0.0):.3f} "
            f"same_tgt={metrics.get('target_selection_same_target_fraction', 0.0):.3f} "
            f"focus_H={metrics.get('target_selection_label_entropy', 0.0):.3f} "
            f"fallback={metrics.get('target_selection_fallback_rate', 0.0):.3f} "
            f"distill={metrics.get('distill/loss', 0.0):.4f}/"
            f"{metrics.get('distill/aim_loss', 0.0):.4f}/"
            f"{metrics.get('distill/fire_loss', 0.0):.4f} "
            f"gn={metrics['actor_grad_norm']:.2e}/"
            f"{metrics['critic_grad_norm']:.2e}/"
            f"{metrics['trunk_grad_norm']:.2e} "
            f"lr={lr:.2e} ts={metrics['team_spirit']:.2f}",
            flush=True,
        )

    def on_eval(self, update_idx: int, lr: float, eval_stats) -> bool:
        phase_label = self.context.phase_label
        self.last_eval = eval_stats.mean_reward
        print(
            f"[{phase_label}/mappo] eval update={update_idx}/{self.total_updates} "
            f"mean_reward={eval_stats.mean_reward:+.3f} "
            f"wins={eval_stats.wins}/{eval_stats.episodes} "
            f"losses={eval_stats.losses}/{eval_stats.episodes} "
            f"draws={eval_stats.draws}/{eval_stats.episodes} "
            f"term={eval_stats.terminated} trunc={eval_stats.truncated} "
            f"tick={eval_stats.mean_final_tick:.1f} "
            f"score={eval_stats.mean_team_a_score:.2f}/"
            f"{eval_stats.mean_team_b_score:.2f} "
            f"obj_t={eval_stats.objective_unlock_seconds:.2f}/"
            f"{eval_stats.objective_capture_seconds:.2f} "
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
            f"maj_sec={eval_stats.mean_majority_on_point_seconds_a:.2f}/"
            f"{eval_stats.mean_majority_on_point_seconds_b:.2f} "
            f"uncont={eval_stats.mean_uncontested_on_point_seconds_a:.2f}/"
            f"{eval_stats.mean_uncontested_on_point_seconds_b:.2f} "
            f"edge_noscore={eval_stats.mean_alive_edge_no_score_seconds_a:.2f}/"
            f"{eval_stats.mean_alive_edge_no_score_seconds_b:.2f} "
            f"cap_gain={eval_stats.mean_cap_progress_gain_ticks:.1f} "
            f"dmg_fire={eval_stats.team_a_damage_per_fire:.1f}/"
            f"{eval_stats.team_b_damage_per_fire:.1f}",
            flush=True,
        )
        self.wandb_logger.log(
            {f"eval/{k}": float(v) for k, v in eval_stats_dict(eval_stats).items()},
            step=update_idx,
        )
        self._log_canonical_eval(update_idx)
        self._record_best_eval(update_idx, eval_stats)
        if self.context.run_cfg.get("eval_gate"):
            run_eval_gate(
                phase_label=phase_label,
                stats=eval_stats,
                gate_cfg=EvalGateConfig.from_dict(
                    dict(self.context.run_cfg.get("eval_gate", {}))
                ),
                output_dir=self.context.output_dir,
            )
        return self._maybe_stop_cap_duel_distill(update_idx, eval_stats)

    def _maybe_stop_cap_duel_distill(self, update_idx: int, eval_stats) -> bool:
        gate_cfg = dict(self.context.run_cfg.get("cap_duel_distill_early_stop", {}))
        if not bool(gate_cfg.get("enabled", False)):
            return False
        stop_update = int(gate_cfg.get("update", 50))
        if int(update_idx) < stop_update:
            return False
        min_hit_fire = float(gate_cfg.get("min_team_a_hit_fire", 0.04))
        max_score = float(gate_cfg.get("max_mean_score_a", 0.0))
        if (
            float(eval_stats.team_a_hit_fire) >= min_hit_fire
            or float(eval_stats.mean_team_a_score) > max_score
        ):
            return False
        payload = {
            "status": "EARLY_STOP",
            "reason": "cap_duel_distill_hit_fire_zero_score",
            "update": int(update_idx),
            "team_a_hit_fire": float(eval_stats.team_a_hit_fire),
            "min_team_a_hit_fire": min_hit_fire,
            "mean_score_a": float(eval_stats.mean_team_a_score),
            "max_mean_score_a": max_score,
        }
        path = self.context.output_dir / "early_stop_decision.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{self.context.phase_label}/mappo] early_stop "
            f"reason={payload['reason']} update={update_idx} "
            f"hit_fire={eval_stats.team_a_hit_fire:.4f}<{min_hit_fire:.4f} "
            f"score_a={eval_stats.mean_team_a_score:.2f}<={max_score:.2f} "
            f"path={path}",
            flush=True,
        )
        return True

    def _log_canonical_eval(self, update_idx: int) -> None:
        canonical_stats = self._last_canonical_eval_stats
        if canonical_stats is None:
            return
        phase_label = self.context.phase_label
        print(
            f"[{phase_label}/mappo] canonical_eval update={update_idx}/"
            f"{self.total_updates} "
            f"mean_reward={canonical_stats.mean_reward:+.3f} "
            f"wins={canonical_stats.wins}/{canonical_stats.episodes} "
            f"losses={canonical_stats.losses}/{canonical_stats.episodes} "
            f"draws={canonical_stats.draws}/{canonical_stats.episodes} "
            f"score={canonical_stats.mean_team_a_score:.2f}/"
            f"{canonical_stats.mean_team_b_score:.2f} "
            f"kills={canonical_stats.mean_team_a_kills:.1f}/"
            f"{canonical_stats.mean_team_b_kills:.1f} "
            f"maj_sec={canonical_stats.mean_majority_on_point_seconds_a:.2f}/"
            f"{canonical_stats.mean_majority_on_point_seconds_b:.2f} "
            f"uncont={canonical_stats.mean_uncontested_on_point_seconds_a:.2f}/"
            f"{canonical_stats.mean_uncontested_on_point_seconds_b:.2f} "
            f"cap_gain={canonical_stats.mean_cap_progress_gain_ticks:.1f}",
            flush=True,
        )
        self.wandb_logger.log(
            {
                f"canonical_eval/{k}": float(v)
                for k, v in eval_stats_dict(canonical_stats).items()
            },
            step=update_idx,
        )

    def _record_best_eval(self, update_idx: int, eval_stats) -> None:
        if self.last_eval <= self.best_eval:
            return
        self.best_eval = self.last_eval
        self.best_state = copy.deepcopy(self.trainer.model.state_dict())
        self.best_eval_update_idx = update_idx
        self.best_eval_stats = {
            "wins": int(eval_stats.wins),
            "losses": int(eval_stats.losses),
            "draws": int(eval_stats.draws),
            "score_team_a": float(eval_stats.mean_team_a_score),
            "score_team_b": float(eval_stats.mean_team_b_score),
            "kills_team_a": float(eval_stats.mean_team_a_kills),
            "kills_team_b": float(eval_stats.mean_team_b_kills),
        }

    def on_checkpoint(self, update_idx: int, payload: dict) -> None:
        checkpoint_path = payload["path"]
        save_mappo_checkpoint(
            path=checkpoint_path,
            model_state_dict=self.trainer.model.state_dict(),
            phase=self.context.phase,
            phase_label=self.context.phase_label,
            ckpt_env_cfg=self.context.ckpt_env_cfg,
            mappo_cfg=self.context.cfg,
        )
        if self.context.retention is None:
            return
        manifest = self.context.retention.record_checkpoint(
            checkpoint_path,
            update=update_idx,
            score=self.last_eval,
        )
        print(
            f"[{self.context.phase_label}/mappo] snapshot_pool "
            f"latest={len(manifest['latest'])} "
            f"historical={len(manifest['historical'])} "
            f"anchor={len(manifest['anchor'])}",
            flush=True,
        )
