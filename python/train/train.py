"""Phase-0/Phase-2 training entrypoint.

Phase 0:
    Determinism harness over scripted bot rollouts in the C++ simulator.
Phase 2:
    Memory-toy recurrent PPO training + feedforward baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml



def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _phase0_runner_imports():
    from xushi2.runner import run_episode

    return run_episode


def _run_pass(sim_cfg: dict, bot_a: str, bot_b: str, episodes: int,
              base_seed: int):
    run_episode = _phase0_runner_imports()
    results = []
    for i in range(episodes):
        results.append(run_episode(sim_cfg, bot_a, bot_b, seed_override=base_seed + i))
    return results


def _assert_identical(pass_a, pass_b) -> int:
    """Return 0 on full match, 1 on first divergence (and print it)."""
    if len(pass_a) != len(pass_b):
        print(f"[xushi2] MISMATCH: episode count {len(pass_a)} vs {len(pass_b)}")
        return 1
    for ep_idx, (a, b) in enumerate(zip(pass_a, pass_b)):
        if a.final_tick != b.final_tick:
            print(f"[xushi2] MISMATCH at episode={ep_idx}: "
                  f"final_tick {a.final_tick} vs {b.final_tick}")
            return 1
        if len(a.decision_hashes) != len(b.decision_hashes):
            print(f"[xushi2] MISMATCH at episode={ep_idx}: "
                  f"decision count {len(a.decision_hashes)} vs {len(b.decision_hashes)}")
            return 1
        for d_idx, (ha, hb) in enumerate(zip(a.decision_hashes, b.decision_hashes)):
            if ha != hb:
                print(f"[xushi2] MISMATCH at episode={ep_idx} decision={d_idx}: "
                      f"expected=0x{ha:016x} actual=0x{hb:016x}")
                return 1
    return 0


def _run_phase0(config: dict) -> int:
    sim_cfg = config.get("sim", {})
    run_cfg = config.get("run", {})

    episodes = int(run_cfg.get("episodes", 4))
    bot_a = str(run_cfg.get("team_a_bot", "basic"))
    bot_b = str(run_cfg.get("team_b_bot", "basic"))
    base_seed = int(sim_cfg.get("seed", 0))

    print(f"[xushi2] phase=0 episodes={episodes} "
          f"bots={bot_a} vs {bot_b} base_seed=0x{base_seed:x}")

    pass_a = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)
    pass_b = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)

    rc = _assert_identical(pass_a, pass_b)
    if rc == 0:
        total = sum(len(r.decision_hashes) for r in pass_a)
        per_ep = len(pass_a[0].decision_hashes) if pass_a else 0
        print(f"[xushi2] OK: {episodes} episodes × {per_ep} decisions "
              f"({total} hashes) all identical")
    return rc


def _run_phase2(config: dict) -> int:
    from train.ppo_recurrent import train_from_config

    result = train_from_config(config)
    recurrent = float(result["recurrent"])
    feedforward = float(result["feedforward"])
    gap = recurrent - feedforward
    print(
        f"[phase2] recurrent_final={recurrent:.3f} "
        f"feedforward_final={feedforward:.3f} "
        f"gap={gap:.3f}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 training entrypoint")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training config YAML under experiments/configs/",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    phase = int(config.get("phase", -1))
    run_cfg = config.get("run", {})
    assert_determinism = bool(run_cfg.get("assert_determinism", True))

    if phase == 0 and assert_determinism:
        return _run_phase0(config)

    if phase == 2:
        return _run_phase2(config)

    print(f"[xushi2] phase {phase} not yet supported by this entrypoint")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
