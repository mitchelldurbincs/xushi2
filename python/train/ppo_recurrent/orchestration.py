"""Top-level training orchestration for recurrent PPO phases."""

from __future__ import annotations

import copy
from functools import partial
from pathlib import Path
from typing import Callable

import gymnasium as gym
import torch


def _make_phase2_env(episode_length: int, cue_visible_ticks: int):
    from envs.memory_toy import MemoryToyEnv

    return MemoryToyEnv(
        episode_length=episode_length, cue_visible_ticks=cue_visible_ticks
    )


def _make_phase3_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase3_ranger import Phase3RangerEnv

    return Phase3RangerEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


from train.models import ActorCritic
from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.evaluate import evaluate_policy_stats
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.ppo_recurrent.logging import format_human_event, log_checkpoint, log_early_stop, log_eval, log_update
from train.ppo_recurrent.trainer import PPOTrainer


def _phase_task_spec(config: dict) -> dict:
    phase = int(config.get("phase", 2))
    if phase == 2:
        return {
            "label": "phase2",
            "obs_dim": 3,
            "action_dim": 2,
            "continuous_action_dim": 2,
            "binary_action_dim": 0,
        }
    if phase == 3:
        return {
            "label": "phase3",
            "obs_dim": 31,
            "action_dim": 6,
            "continuous_action_dim": 3,
            "binary_action_dim": 3,
        }
    raise ValueError(f"unsupported recurrent PPO phase: {phase}")


def make_ppo_config(config: dict, *, use_recurrence: bool) -> PPOConfig:
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    task_spec = _phase_task_spec(config)
    return PPOConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=int(task_spec["obs_dim"]),
        action_dim=int(task_spec["action_dim"]),
        continuous_action_dim=int(task_spec["continuous_action_dim"]),
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
        action_log_std_init=float(model_cfg["action_log_std_init"]),
        use_recurrence=bool(use_recurrence),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_ratio=float(ppo_cfg["clip_ratio"]),
        value_clip_ratio=float(ppo_cfg["value_clip_ratio"]),
        value_coef=float(ppo_cfg["value_coef"]),
        entropy_coef=float(ppo_cfg["entropy_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        learning_rate=float(ppo_cfg["learning_rate"]),
        num_epochs=int(ppo_cfg["num_epochs"]),
        minibatch_size=int(ppo_cfg["minibatch_size"]),
        lr_schedule=str(ppo_cfg.get("lr_schedule", "constant")),
        lr_final_ratio=float(ppo_cfg.get("lr_final_ratio", 1.0)),
        warmup_updates=int(ppo_cfg.get("warmup_updates", 0)),
        value_normalization=bool(ppo_cfg.get("value_normalization", True)),
        binary_action_dim=int(task_spec["binary_action_dim"]),
        vector_env=str(ppo_cfg.get("vector_env", "sync")),
        torch_num_threads=int(ppo_cfg.get("torch_num_threads", 0)),
    )


def make_env_fn(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    phase = int(config.get("phase", 2))
    env_cfg = config.get("env", {})
    if phase == 2:
        ep_len = int(env_cfg.get("episode_length", 64))
        cue_ticks = int(env_cfg.get("cue_visible_ticks", 4))
        env_fn = partial(_make_phase2_env, ep_len, cue_ticks)
        return (
            env_fn,
            {"episode_length": ep_len, "cue_visible_ticks": cue_ticks},
            int(env_cfg.get("seed_base", 0)),
        )

    if phase == 3:
        sim_cfg = dict(env_cfg.get("sim", {}))
        opponent_bot = str(env_cfg.get("opponent_bot", "basic"))
        learner_team = str(env_cfg.get("learner_team", "A"))
        reward_cfg = dict(env_cfg.get("reward", {}))
        env_fn = partial(
            _make_phase3_env, sim_cfg, opponent_bot, learner_team, reward_cfg
        )
        return (
            env_fn,
            {
                "sim": sim_cfg,
                "opponent_bot": opponent_bot,
                "learner_team": learner_team,
                "reward": reward_cfg,
            },
            int(env_cfg.get("seed_base", sim_cfg.get("seed", 0))),
        )

    raise ValueError(f"unsupported recurrent PPO phase: {phase}")


def _save_checkpoint(model: ActorCritic, path: Path, ckpt_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": ckpt_config},
        path,
    )


# Topology-relevant model config fields. A mismatch on any of these means the
# saved state_dict will not fit the new model — fail loud before torch's
# generic shape-mismatch error makes the cause hard to read.
_WARM_START_TOPOLOGY_KEYS = (
    "obs_dim",
    "action_dim",
    "continuous_action_dim",
    "binary_action_dim",
    "use_recurrence",
    "embed_dim",
    "gru_hidden",
    "head_hidden",
)


def _load_init_checkpoint(
    model: ActorCritic,
    ckpt_path: str | Path,
    expected_model_cfg: dict,
) -> None:
    """Load model weights from ``ckpt_path`` into ``model`` for warm-start.

    Optimizer state, RNG state, and rollout buffer are intentionally not
    loaded — fresh Adam moments and a clean LR schedule are appropriate
    when warm-starting across reward / opponent / curriculum changes.

    Raises ``ValueError`` if the checkpoint's saved model config disagrees
    with ``expected_model_cfg`` on any topology-relevant field.
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_model_cfg = state.get("config", {}).get("model", {})
    for key in _WARM_START_TOPOLOGY_KEYS:
        saved = saved_model_cfg.get(key)
        expected = expected_model_cfg.get(key)
        if saved is not None and expected is not None and saved != expected:
            raise ValueError(
                f"warm-start checkpoint architecture mismatch on '{key}': "
                f"checkpoint={saved!r} config={expected!r} (path={ckpt_path})"
            )
    model.load_state_dict(state["model_state_dict"])


def _run_variant(
    config: dict,
    *,
    use_recurrence: bool,
    output_dir: Path,
) -> float:
    run_cfg = config.get("run", {})
    ppo_cfg = make_ppo_config(config, use_recurrence=use_recurrence)
    env_fn, ckpt_env_cfg, seed_base = make_env_fn(config)
    task_spec = _phase_task_spec(config)
    phase_label = str(task_spec["label"])

    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 50))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    log_every = int(run_cfg.get("log_every", 1))
    early_stop_patience_evals = int(run_cfg.get("early_stop_patience_evals", 0))
    early_stop_min_delta = float(run_cfg.get("early_stop_min_delta", 0.0))
    max_regression_from_best = float(run_cfg.get("max_regression_from_best", -1.0))

    # Offset the feedforward seed so the two variants don't share their
    # RNG trajectory (different weight init, different action noise).
    variant_seed = seed_base + (0 if use_recurrence else 1_000_000)

    trainer = PPOTrainer(env_fn=env_fn, config=ppo_cfg, seed=variant_seed)

    ckpt_cfg = {
        "phase": int(config.get("phase", 2)),
        "env": ckpt_env_cfg,
        "model": {
            **config.get("model", {}),
            "use_recurrence": use_recurrence,
            "obs_dim": int(task_spec["obs_dim"]),
            "action_dim": int(task_spec["action_dim"]),
            "continuous_action_dim": int(task_spec["continuous_action_dim"]),
            "binary_action_dim": int(task_spec["binary_action_dim"]),
        },
        "ppo": dict(config.get("ppo", {})),
    }

    variant_name = "recurrent" if use_recurrence else "feedforward"

    init_ckpt = run_cfg.get("init_from_checkpoint")
    if init_ckpt:
        _load_init_checkpoint(trainer.model, init_ckpt, ckpt_cfg["model"])
        print(
            f"[{phase_label}/{variant_name}] warm-start: loaded {init_ckpt}",
            flush=True,
        )

    # Best-checkpoint tracking. PPO is known to oscillate after a policy
    # breakthrough on this task (regresses from -0.28 back to -0.40+ if
    # training continues past the breakthrough), so the last-state model
    # is often worse than an intermediate one. ``ckpt_final.pt`` is
    # therefore defined as the best-eval checkpoint seen during training.
    best_eval = float("-inf")
    best_state: dict | None = None
    best_update: int = 0
    no_improve_eval_count: int = 0

    last_eval = float("nan")
    stop_reason: str | None = None
    try:
        for update_idx in range(1, total_updates + 1):
            current_lr = lr_for_update(
                update_idx,
                total_updates,
                base_lr=ppo_cfg.learning_rate,
                schedule=ppo_cfg.lr_schedule,
                lr_final_ratio=ppo_cfg.lr_final_ratio,
                warmup_updates=ppo_cfg.warmup_updates,
            )
            trainer.set_learning_rate(current_lr)
            rollout = trainer.collect_rollout()
            metrics = trainer.update(rollout)

            if log_every > 0 and update_idx % log_every == 0:
                update_record = log_update(
                    phase=phase_label,
                    variant=variant_name,
                    update=update_idx,
                    total_updates=total_updates,
                    metrics=metrics,
                )
                print(format_human_event(update_record), flush=True)

            if update_idx % eval_every == 0 or update_idx == total_updates:
                eval_stats = evaluate_policy_stats(
                    trainer.model,
                    env_fn,
                    num_episodes=eval_episodes,
                    seed=variant_seed + 100_000 + update_idx,
                )
                last_eval = eval_stats.mean_reward
                eval_record = log_eval(
                    phase=phase_label,
                    variant=variant_name,
                    update=update_idx,
                    total_updates=total_updates,
                    lr=current_lr,
                    eval_stats=eval_stats,
                )
                print(format_human_event(eval_record), flush=True)
                if last_eval > (best_eval + early_stop_min_delta):
                    best_eval = last_eval
                    best_update = update_idx
                    best_state = copy.deepcopy(trainer.model.state_dict())
                    no_improve_eval_count = 0
                else:
                    no_improve_eval_count += 1

                if (
                    max_regression_from_best >= 0.0
                    and best_eval > float("-inf")
                    and (best_eval - last_eval) > max_regression_from_best
                ):
                    stop_reason = (
                        "eval regression exceeded max_regression_from_best: "
                        f"best={best_eval:+.3f} current={last_eval:+.3f} "
                        f"drop={best_eval - last_eval:+.3f} "
                        f"threshold={max_regression_from_best:+.3f} "
                        f"at update={update_idx}"
                    )
                    break
                if (
                    early_stop_patience_evals > 0
                    and no_improve_eval_count >= early_stop_patience_evals
                ):
                    stop_reason = (
                        "eval improvement stagnated past patience: "
                        f"no_improve_evals={no_improve_eval_count} "
                        f"patience={early_stop_patience_evals} "
                        f"min_delta={early_stop_min_delta:+.3f} "
                        f"at update={update_idx}"
                    )
                    break

            if update_idx % checkpoint_every == 0 or update_idx == total_updates:
                ckpt_path = output_dir / f"ckpt_{update_idx:04d}.pt"
                _save_checkpoint(trainer.model, ckpt_path, ckpt_cfg)
                checkpoint_record = log_checkpoint(
                    phase=phase_label,
                    variant=variant_name,
                    update=update_idx,
                    total_updates=total_updates,
                    path=str(ckpt_path),
                )
                print(format_human_event(checkpoint_record), flush=True)
    finally:
        envs = getattr(trainer, "envs", None)
        if envs is not None:
            envs.close()

    # ckpt_final.pt holds the best-eval snapshot (per the note above).
    # If no eval ever ran (total_updates < eval_every and not aligned to
    # total_updates), fall back to the last state.
    output_dir.mkdir(parents=True, exist_ok=True)
    if stop_reason is not None:
        early_stop_record = log_early_stop(
            phase=phase_label,
            variant=variant_name,
            update=best_update if best_update > 0 else total_updates,
            total_updates=total_updates,
            reason=stop_reason,
        )
        print(format_human_event(early_stop_record), flush=True)
    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state, "config": ckpt_cfg},
            output_dir / "ckpt_final.pt",
        )
        print(
            f"[{phase_label}/{variant_name}] best checkpoint: "
            f"eval@{best_update}={best_eval:+.3f}",
            flush=True,
        )
        return best_eval
    _save_checkpoint(trainer.model, output_dir / "ckpt_final.pt", ckpt_cfg)
    return last_eval


def train_from_config(config: dict) -> dict[str, float]:
    """Train one or more variants for the selected recurrent PPO phase."""
    phase = int(config.get("phase", 2))
    run_cfg = config.get("run", {})
    default_out = "runs/phase2_memory_toy" if phase == 2 else "runs/phase3_ranger"
    out_root = Path(str(run_cfg.get("output_dir", default_out)))
    rec_eval = _run_variant(
        config, use_recurrence=True, output_dir=out_root / "recurrent"
    )
    if phase == 2:
        ff_eval = _run_variant(
            config, use_recurrence=False, output_dir=out_root / "feedforward"
        )
        return {"recurrent": rec_eval, "feedforward": ff_eval}
    return {"recurrent": rec_eval}
