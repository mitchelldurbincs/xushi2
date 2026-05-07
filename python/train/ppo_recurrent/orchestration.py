"""Top-level training orchestration for recurrent PPO phases."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from train.models import ActorCritic
from train.phases import resolve_phase
from train.ppo_recurrent.config import PPOConfig
from train.ppo_recurrent.evaluate import evaluate_policy_stats
from train.ppo_recurrent.lr_schedule import lr_for_update
from train.ppo_recurrent.logging import (
    format_human_event,
    log_checkpoint,
    log_early_stop,
    log_eval,
    log_update,
)
from train.ppo_recurrent.trainer import PPOTrainer

_CKPT_SCHEMA_VERSION = 1


def make_env_fn(config: dict) -> tuple[Callable[[], object], dict, int]:
    _, phase_spec = resolve_phase(config)
    env_bundle = phase_spec.get("env_bundle")
    if env_bundle is None:
        raise ValueError(f"unsupported phase/config shape: phase={config.get('phase')!r}")
    return env_bundle(config)


def _phase_task_spec(config: dict) -> dict:
    _, phase_spec = resolve_phase(config)
    required = (
        "label",
        "obs_dim",
        "action_dim",
        "continuous_action_dim",
        "binary_action_dim",
    )
    missing = [k for k in required if k not in phase_spec]
    if missing:
        raise ValueError(
            f"unsupported phase/config shape: phase={config.get('phase')!r} missing={missing}"
        )
    return {k: phase_spec[k] for k in required}




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

def _save_checkpoint(model: ActorCritic, path: Path, ckpt_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": ckpt_config},
        path,
    )


@dataclass(frozen=True)
class CheckpointModelTopology:
    obs_dim: int
    action_dim: int
    continuous_action_dim: int
    binary_action_dim: int
    use_recurrence: bool
    embed_dim: int
    gru_hidden: int
    head_hidden: int


@dataclass(frozen=True)
class CheckpointRunContext:
    phase: int
    env: dict
    ppo: dict
    schema_version: int = _CKPT_SCHEMA_VERSION


@dataclass(frozen=True)
class CheckpointConfig:
    model: CheckpointModelTopology
    run: CheckpointRunContext

    def to_dict(self) -> dict:
        out = asdict(self)
        run_cfg = out.pop("run")
        out.update(run_cfg)
        return out


def _build_checkpoint_config(
    config: dict, ckpt_env_cfg: dict, task_spec: dict, *, use_recurrence: bool
) -> CheckpointConfig:
    model_cfg = config.get("model", {})
    topology = CheckpointModelTopology(
        obs_dim=int(task_spec["obs_dim"]),
        action_dim=int(task_spec["action_dim"]),
        continuous_action_dim=int(task_spec["continuous_action_dim"]),
        binary_action_dim=int(task_spec["binary_action_dim"]),
        use_recurrence=bool(use_recurrence),
        embed_dim=int(model_cfg["embed_dim"]),
        gru_hidden=int(model_cfg["gru_hidden"]),
        head_hidden=int(model_cfg["head_hidden"]),
    )
    run = CheckpointRunContext(
        phase=int(config.get("phase", 2)),
        env=ckpt_env_cfg,
        ppo=dict(config.get("ppo", {})),
    )
    return CheckpointConfig(model=topology, run=run)


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
    ckpt_config = _normalize_checkpoint_config(state.get("config", {}))
    _validate_checkpoint_topology(
        ckpt_config.get("model", {}), expected_model_cfg, ckpt_path=ckpt_path
    )
    model.load_state_dict(state["model_state_dict"])


def _normalize_checkpoint_config(raw_config: dict) -> dict:
    """Normalize older/newer checkpoint config shapes into current schema."""
    schema_version = int(raw_config.get("schema_version", 0))
    normalized = dict(raw_config)
    if schema_version == 0:
        normalized["schema_version"] = _CKPT_SCHEMA_VERSION
    return normalized


def _validate_checkpoint_topology(
    saved_model_cfg: dict,
    expected_model_cfg: dict,
    *,
    ckpt_path: str | Path,
) -> None:
    missing_keys = [k for k in _WARM_START_TOPOLOGY_KEYS if k not in saved_model_cfg]
    mismatches: list[tuple[str, object, object]] = []
    for key in _WARM_START_TOPOLOGY_KEYS:
        if key in saved_model_cfg and key in expected_model_cfg:
            saved = saved_model_cfg[key]
            expected = expected_model_cfg[key]
            if saved != expected:
                mismatches.append((key, saved, expected))
    if missing_keys or mismatches:
        chunks = []
        if missing_keys:
            chunks.append(f"missing topology keys={missing_keys}")
        if mismatches:
            mismatch_desc = ", ".join(
                f"{k}(checkpoint={s!r}, config={e!r})" for k, s, e in mismatches
            )
            chunks.append(f"mismatched topology fields: {mismatch_desc}")
        raise ValueError(
            f"warm-start checkpoint incompatible ({'; '.join(chunks)}) "
            f"(path={ckpt_path})"
        )


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

    ckpt_cfg = _build_checkpoint_config(
        config, ckpt_env_cfg, task_spec, use_recurrence=use_recurrence
    ).to_dict()

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
    phase, phase_spec = resolve_phase(config)
    run_cfg = config.get("run", {})
    default_out = "runs/phase2_memory_toy" if phase == 2 else "runs/phase3_ranger"
    out_root = Path(str(run_cfg.get("output_dir", default_out)))
    rec_eval = _run_variant(
        config, use_recurrence=True, output_dir=out_root / "recurrent"
    )
    if "feedforward" in phase_spec.get("training_variants", ()):
        ff_eval = _run_variant(
            config, use_recurrence=False, output_dir=out_root / "feedforward"
        )
        return {"recurrent": rec_eval, "feedforward": ff_eval}
    return {"recurrent": rec_eval}
