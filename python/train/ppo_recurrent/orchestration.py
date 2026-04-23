"""Top-level training orchestration for Phase-2 memory-toy runs.

Wraps two ``PPOTrainer`` invocations (recurrent + feedforward variants),
periodic evaluation, and best-eval checkpointing into a single
``train_from_config`` entry point.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from train.models import ActorCritic
from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.evaluate import evaluate_policy
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.ppo_recurrent.trainer import PPOTrainer


def make_ppo_config(config: dict, *, use_recurrence: bool) -> PPOConfig:
    model_cfg = config.get("model", {})
    ppo_cfg = config.get("ppo", {})
    return PPOConfig(
        num_envs=int(ppo_cfg["num_envs"]),
        rollout_len=int(ppo_cfg["rollout_len"]),
        obs_dim=3,
        action_dim=2,
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
    )


def _save_checkpoint(model: ActorCritic, path: Path, ckpt_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": ckpt_config},
        path,
    )


def _run_variant(
    config: dict,
    *,
    use_recurrence: bool,
    output_dir: Path,
) -> float:
    from envs.memory_toy import MemoryToyEnv

    env_cfg = config.get("env", {})
    run_cfg = config.get("run", {})
    ppo_cfg = make_ppo_config(config, use_recurrence=use_recurrence)

    total_updates = int(run_cfg.get("total_updates"))
    eval_every = int(run_cfg.get("eval_every", max(1, total_updates)))
    eval_episodes = int(run_cfg.get("eval_episodes", 50))
    checkpoint_every = int(run_cfg.get("checkpoint_every", max(1, total_updates)))
    log_every = int(run_cfg.get("log_every", 1))
    early_stop_patience_evals = int(run_cfg.get("early_stop_patience_evals", 0))
    early_stop_min_delta = float(run_cfg.get("early_stop_min_delta", 0.0))
    max_regression_from_best = float(run_cfg.get("max_regression_from_best", -1.0))

    seed_base = int(env_cfg.get("seed_base", 0))
    # Offset the feedforward seed so the two variants don't share their
    # RNG trajectory (different weight init, different action noise).
    variant_seed = seed_base + (0 if use_recurrence else 1_000_000)

    ep_len = int(env_cfg.get("episode_length", 64))
    cue_ticks = int(env_cfg.get("cue_visible_ticks", 4))

    def env_fn() -> MemoryToyEnv:
        return MemoryToyEnv(episode_length=ep_len, cue_visible_ticks=cue_ticks)

    trainer = PPOTrainer(env_fn=env_fn, config=ppo_cfg, seed=variant_seed)

    ckpt_cfg = {
        "env": {"episode_length": ep_len, "cue_visible_ticks": cue_ticks},
        "model": {
            **config.get("model", {}),
            "use_recurrence": use_recurrence,
            "obs_dim": 3,
            "action_dim": 2,
        },
        "ppo": dict(config.get("ppo", {})),
    }

    variant_name = "recurrent" if use_recurrence else "feedforward"

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
            print(
                f"[phase2/{variant_name}] update={update_idx:4d}/{total_updates} "
                f"policy_loss={metrics['policy_loss']:+.3f} "
                f"value_loss={metrics['value_loss']:.3f} "
                f"entropy={metrics['entropy']:+.3f} "
                f"approx_kl={metrics['approx_kl']:+.4f} "
                f"actor_gn={metrics['actor_grad_norm']:.3f} "
                f"critic_gn={metrics['critic_grad_norm']:.3f} "
                f"trunk_gn={metrics['trunk_grad_norm']:.3f} "
                f"term_adv_std={metrics['terminal_adv_std']:.3f} "
                f"mean_log_std={metrics['mean_log_std']:+.3f} "
                f"lr={metrics['lr']:.3e}",
                flush=True,
            )

        if update_idx % eval_every == 0 or update_idx == total_updates:
            last_eval = evaluate_policy(
                trainer.model,
                env_fn,
                num_episodes=eval_episodes,
                seed=variant_seed + 100_000 + update_idx,
            )
            print(
                f"[phase2/{variant_name}] eval@{update_idx}={last_eval:+.3f} "
                f"lr={current_lr:.3e}",
                flush=True,
            )
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
            _save_checkpoint(
                trainer.model,
                output_dir / f"ckpt_{update_idx:04d}.pt",
                ckpt_cfg,
            )

    # ckpt_final.pt holds the best-eval snapshot (per the note above).
    # If no eval ever ran (total_updates < eval_every and not aligned to
    # total_updates), fall back to the last state.
    output_dir.mkdir(parents=True, exist_ok=True)
    if stop_reason is not None:
        print(f"[phase2/{variant_name}] early-stop: {stop_reason}", flush=True)
    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state, "config": ckpt_cfg},
            output_dir / "ckpt_final.pt",
        )
        print(
            f"[phase2/{variant_name}] best checkpoint: "
            f"eval@{best_update}={best_eval:+.3f}",
            flush=True,
        )
        return best_eval
    _save_checkpoint(trainer.model, output_dir / "ckpt_final.pt", ckpt_cfg)
    return last_eval


def train_from_config(config: dict) -> dict[str, float]:
    """Train recurrent and feedforward variants; return final eval rewards.

    Produces two checkpoint trees:
        {output_dir}/recurrent/ckpt_final.pt
        {output_dir}/feedforward/ckpt_final.pt

    The recurrent checkpoint is the one fed to the ablation gate.
    """
    run_cfg = config.get("run", {})
    out_root = Path(str(run_cfg.get("output_dir", "runs/phase2_memory_toy")))
    rec_eval = _run_variant(
        config, use_recurrence=True, output_dir=out_root / "recurrent"
    )
    ff_eval = _run_variant(
        config, use_recurrence=False, output_dir=out_root / "feedforward"
    )
    return {"recurrent": rec_eval, "feedforward": ff_eval}
