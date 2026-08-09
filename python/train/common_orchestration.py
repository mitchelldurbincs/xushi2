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
    total_updates: int
    eval_every: int
    checkpoint_every: int
    log_every: int
    base_lr: float
    lr_schedule: str
    lr_final_ratio: float
    warmup_updates: int
    # First update to run. Resuming starts after the last completed update
    # while keeping total_updates fixed, so the LR schedule continues from
    # where it left off instead of restarting.
    start_update: int = 1


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
    for update_idx in range(max(1, cfg.start_update), cfg.total_updates + 1):
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

        should_stop = False
        if update_idx % cfg.eval_every == 0 or update_idx == cfg.total_updates:
            eval_result = hooks.evaluate_step(update_idx, lr)
            should_stop = hooks.on_eval(update_idx, lr, eval_result)

        if update_idx % cfg.checkpoint_every == 0 or update_idx == cfg.total_updates:
            hooks.on_checkpoint(update_idx, hooks.checkpoint_payload(update_idx))

        if should_stop:
            break
