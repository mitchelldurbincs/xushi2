from __future__ import annotations

from dataclasses import dataclass, field

from train.common_orchestration import LoopConfig, run_training_loop


@dataclass
class _Hooks:
    logs: list[int] = field(default_factory=list)
    evals: list[int] = field(default_factory=list)
    ckpts: list[int] = field(default_factory=list)

    def set_learning_rate(self, lr: float) -> None:
        pass

    def collect_rollout(self, update_idx: int):
        return update_idx

    def update_step(self, update_idx: int, rollout: int, lr: float):
        return {"u": update_idx}

    def evaluate_step(self, update_idx: int, lr: float):
        return update_idx

    def checkpoint_payload(self, update_idx: int):
        return update_idx

    def on_log(self, update_idx: int, lr: float, metrics: dict):
        self.logs.append(update_idx)

    def on_eval(self, update_idx: int, lr: float, eval_result: int) -> bool:
        self.evals.append(update_idx)
        return False

    def on_checkpoint(self, update_idx: int, payload: int) -> None:
        self.ckpts.append(update_idx)


def test_trigger_cadence_static() -> None:
    hooks = _Hooks()
    run_training_loop(
        LoopConfig(
            total_updates=10,
            eval_every=4,
            checkpoint_every=3,
            log_every=2,
            base_lr=1e-3,
            lr_schedule="constant",
            lr_final_ratio=1.0,
            warmup_updates=0,
        ),
        hooks,
    )
    assert hooks.logs == [2, 4, 6, 8, 10]
    assert hooks.evals == [4, 8, 10]
    assert hooks.ckpts == [3, 6, 9, 10]
