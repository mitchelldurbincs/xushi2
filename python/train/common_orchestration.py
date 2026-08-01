from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from train.lr_schedule import lr_for_update

RolloutT = TypeVar("RolloutT")
UpdateMetricsT = TypeVar("UpdateMetricsT")
EvalResultT = TypeVar("EvalResultT")
CheckpointPayloadT = TypeVar("CheckpointPayloadT")


@dataclass(frozen=True)
class LoopConfig:
    """Cadence and LR schedule for :func:`run_training_loop`.

    Cadence convention, uniform across all three knobs: a value of ``0``
    disables the periodic action, and negative values are a config error
    rejected at parse time (see ``train.mappo_runtime_context``).

    Disabling is *periodic only*. The final update always evaluates and
    always checkpoints regardless of ``eval_every`` / ``checkpoint_every``,
    because ``last_eval`` is the training loop's return value and a run that
    saved nothing is worse than one that saved once.
    """

    total_updates: int
    eval_every: int
    checkpoint_every: int
    log_every: int
    base_lr: float
    lr_schedule: str
    lr_final_ratio: float
    warmup_updates: int


class OrchestrationHooks(Protocol[RolloutT, UpdateMetricsT, EvalResultT, CheckpointPayloadT]):
    def set_learning_rate(self, lr: float) -> None: ...

    def collect_rollout(self, update_idx: int) -> RolloutT: ...

    def update_step(self, update_idx: int, rollout: RolloutT, lr: float) -> UpdateMetricsT: ...

    def evaluate_step(self, update_idx: int, lr: float) -> EvalResultT: ...

    def checkpoint_payload(self, update_idx: int) -> CheckpointPayloadT: ...

    def on_log(self, update_idx: int, lr: float, metrics: UpdateMetricsT) -> None: ...

    def on_eval(self, update_idx: int, lr: float, eval_result: EvalResultT) -> bool: ...

    def on_checkpoint(self, update_idx: int, payload: CheckpointPayloadT) -> None: ...


def run_training_loop(
    cfg: LoopConfig,
    hooks: OrchestrationHooks[RolloutT, UpdateMetricsT, EvalResultT, CheckpointPayloadT],
) -> None:
    for update_idx in range(1, cfg.total_updates + 1):
        lr = lr_for_update(
            update_idx,
            cfg.total_updates,
            base_lr=cfg.base_lr,
            schedule=cfg.lr_schedule,
            lr_final_ratio=cfg.lr_final_ratio,
            warmup_updates=cfg.warmup_updates,
        )
        hooks.set_learning_rate(lr)
        rollout = hooks.collect_rollout(update_idx)
        metrics = hooks.update_step(update_idx, rollout, lr)

        if cfg.log_every > 0 and update_idx % cfg.log_every == 0:
            hooks.on_log(update_idx, lr, metrics)

        is_final_update = update_idx == cfg.total_updates

        should_stop = False
        if is_final_update or (cfg.eval_every > 0 and update_idx % cfg.eval_every == 0):
            eval_result = hooks.evaluate_step(update_idx, lr)
            should_stop = hooks.on_eval(update_idx, lr, eval_result)

        if is_final_update or (cfg.checkpoint_every > 0 and update_idx % cfg.checkpoint_every == 0):
            hooks.on_checkpoint(update_idx, hooks.checkpoint_payload(update_idx))

        if should_stop:
            break
