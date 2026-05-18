"""Training entrypoint and phase routing."""

from __future__ import annotations

import argparse
import faulthandler
import signal
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from xushi2.runner import EpisodeResult


@dataclass(frozen=True)
class NormalizedEntryConfig:
    phase_int: int
    phase_label: str
    sim_cfg: dict
    env_cfg: dict
    run_cfg: dict
    base_seed: int


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if isinstance(config, dict):
        run_cfg = config.setdefault("run", {})
        if isinstance(run_cfg, dict):
            run_cfg.setdefault("composition_pretrain", False)
            run_cfg.setdefault("composition_pretrain_steps", 1000)
            run_cfg.setdefault("composition_objective_teacher_checkpoint", None)
            run_cfg.setdefault("composition_combat_teacher_checkpoint", None)
            run_cfg.setdefault("composition_objective_batch_size", 256)
            run_cfg.setdefault("composition_combat_batch_size", 256)
            run_cfg.setdefault(
                "composition_gate",
                {
                    "objective_on_point_gate": 0.25,
                    "objective_losses_gate": 0,
                    "combat_kills_gate": 12.0,
                    "hit_fire_gate": 0.02,
                    "aim_error_gate": 1.55,
                },
            )
            run_cfg.setdefault("composition_objective_env", {})
            run_cfg.setdefault("composition_combat_env", {})
    return config


def _enable_usr1_traceback_dump() -> None:
    """Make SIGUSR1 useful for diagnosing stuck training jobs.

    Python's default SIGUSR1 behavior is process termination on POSIX. Training
    runs are often launched without an attached TTY, so a terminated process can
    leave no traceback. Registering faulthandler keeps the process alive and
    dumps all Python thread stacks to stderr instead.
    """
    sigusr1 = getattr(signal, "SIGUSR1", None)
    if sigusr1 is None:
        return
    try:
        faulthandler.register(sigusr1, all_threads=True, chain=False)
    except (RuntimeError, ValueError):
        # Some embedded or redirected runtimes cannot register faulthandler.
        # Install a minimal fallback so SIGUSR1 still does not kill the process.
        def _dump_traceback(_signum: int, _frame: FrameType | None) -> None:
            faulthandler.dump_traceback(all_threads=True)

        signal.signal(sigusr1, _dump_traceback)


# --- phase0 harness ---
def _run_pass(
    sim_cfg: dict, bot_a: str, bot_b: str, episodes: int, base_seed: int
) -> list[EpisodeResult]:
    from xushi2.runner import run_episode
    results: list[EpisodeResult] = []
    for i in range(episodes):
        results.append(run_episode(sim_cfg, bot_a, bot_b, seed_override=base_seed + i))
    return results


def _assert_identical(pass_a: list[EpisodeResult], pass_b: list[EpisodeResult]) -> int:
    """Return 0 on full match, 1 on first divergence (and print it)."""
    if len(pass_a) != len(pass_b):
        print(f"[xushi2] MISMATCH: episode count {len(pass_a)} vs {len(pass_b)}")
        return 1
    for ep_idx, (a, b) in enumerate(zip(pass_a, pass_b, strict=False)):
        if a.final_tick != b.final_tick:
            print(
                f"[xushi2] MISMATCH at episode={ep_idx}: "
                f"final_tick {a.final_tick} vs {b.final_tick}"
            )
            return 1
        if len(a.decision_hashes) != len(b.decision_hashes):
            print(
                f"[xushi2] MISMATCH at episode={ep_idx}: "
                f"decision count {len(a.decision_hashes)} vs {len(b.decision_hashes)}"
            )
            return 1
        for d_idx, (ha, hb) in enumerate(zip(a.decision_hashes, b.decision_hashes, strict=False)):
            if ha != hb:
                print(
                    f"[xushi2] MISMATCH at episode={ep_idx} decision={d_idx}: "
                    f"expected=0x{ha:016x} actual=0x{hb:016x}"
                )
                return 1
    return 0


def normalize_entry_config(config: dict) -> NormalizedEntryConfig:
    from train.phases import resolve_phase

    phase_int, phase_spec = resolve_phase(config)
    env_cfg = dict(config.get("env", {}))
    sim_cfg = dict(config.get("sim", {}))
    if phase_int in (2, 3, 4):
        sim_cfg = dict(env_cfg.get("sim", {}))
    run_cfg = dict(config.get("run", {}))
    base_seed = int(env_cfg.get("seed_base", sim_cfg.get("seed", 0)))
    return NormalizedEntryConfig(
        phase_int=phase_int,
        phase_label=str(phase_spec["label"]),
        sim_cfg=sim_cfg,
        env_cfg=env_cfg,
        run_cfg=run_cfg,
        base_seed=base_seed,
    )


def format_phase_banner(normalized: NormalizedEntryConfig, phase_raw: object) -> str:
    episodes = int(normalized.run_cfg.get("episodes", 4))
    bot_a = str(normalized.run_cfg.get("team_a_bot", "basic"))
    bot_b = str(normalized.run_cfg.get("team_b_bot", "basic"))

    phase_groups: list[tuple[set[int], Callable[[], str]]] = [
        (
            {11},
            lambda: (
                f"[xushi2] phase={phase_raw} mappo match_type=current "
                f"learner_team=both base_seed=0x{normalized.base_seed:x}"
            ),
        ),
        (
            {4, 5, 6, 7, 8, 9, 10},
            lambda: (
                f"[xushi2] phase={phase_raw} mappo "
                f"opponent={normalized.env_cfg.get('opponent_bot', '?')} "
                f"learner_team={normalized.env_cfg.get('learner_team', 'A')} "
                f"base_seed=0x{normalized.base_seed:x}"
            ),
        ),
        (
            {3},
            lambda: (
                f"[xushi2] phase={phase_raw} "
                f"opponent={normalized.env_cfg.get('opponent_bot', '?')} "
                f"learner_team={normalized.env_cfg.get('learner_team', 'A')} "
                f"base_seed=0x{normalized.base_seed:x}"
            ),
        ),
        (
            {2},
            lambda: f"[xushi2] phase={phase_raw} memory_toy base_seed=0x{normalized.base_seed:x}",
        ),
    ]
    for phases, renderer in phase_groups:
        if normalized.phase_int in phases:
            return renderer()
    return (
        f"[xushi2] phase={phase_raw} episodes={episodes} "
        f"bots={bot_a} vs {bot_b} base_seed=0x{normalized.base_seed:x}"
    )


def run_phase(normalized: NormalizedEntryConfig, full_config: dict) -> int:
    phase_int = normalized.phase_int
    phase_label = full_config.get("phase", "unknown")

    if phase_int == 0:
        episodes = int(normalized.run_cfg.get("episodes", 4))
        bot_a = str(normalized.run_cfg.get("team_a_bot", "basic"))
        bot_b = str(normalized.run_cfg.get("team_b_bot", "basic"))
        assert_determinism = bool(normalized.run_cfg.get("assert_determinism", True))
        if not assert_determinism:
            print(f"[xushi2] phase {phase_label} not yet supported by this entrypoint")
            return 2

        pass_a = _run_pass(normalized.sim_cfg, bot_a, bot_b, episodes, normalized.base_seed)
        pass_b = _run_pass(normalized.sim_cfg, bot_a, bot_b, episodes, normalized.base_seed)
        rc = _assert_identical(pass_a, pass_b)
        if rc == 0:
            total = sum(len(r.decision_hashes) for r in pass_a)
            per_ep = len(pass_a[0].decision_hashes) if pass_a else 0
            print(
                f"[xushi2] OK: {episodes} episodes x {per_ep} decisions "
                f"({total} hashes) all identical"
            )
        return rc

    if phase_int in (2, 3):
        from train.ppo_recurrent import train_from_config

        result = train_from_config(full_config)
        recurrent = float(result["recurrent"])
        from train.phases import resolve_phase

        if "feedforward" in resolve_phase(full_config)[1].get("training_variants", ()):
            feedforward = float(result["feedforward"])
            gap = recurrent - feedforward
            print(
                f"[{normalized.phase_label}] recurrent_final={recurrent:.3f} "
                f"feedforward_final={feedforward:.3f} gap={gap:.3f}"
            )
        else:
            print(f"[{normalized.phase_label}] recurrent_final={recurrent:.3f}")
        return 0

    if phase_int in (4, 5, 6, 7, 8, 9, 10, 11):
        from train.mappo_eval_checkpoint import train_phase4_from_config

        result = train_phase4_from_config(full_config)
        print(f"[phase{phase_int}] mappo_final={float(result['mappo']):.3f}")
        return 0

    print(f"[xushi2] unsupported phase/config shape: phase={phase_label!r}")
    return 2


def main() -> int:
    _enable_usr1_traceback_dump()
    parser = argparse.ArgumentParser(description="xushi2 training entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training config YAML under experiments/configs/",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        normalized = normalize_entry_config(config)
    except ValueError as exc:
        print(f"[xushi2] {exc}")
        return 2

    phase_raw = config.get("phase", "unknown")
    print(format_phase_banner(normalized, phase_raw), flush=True)
    return run_phase(normalized, config)


if __name__ == "__main__":
    raise SystemExit(main())
