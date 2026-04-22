"""Phase-0 acceptance harness.

Loads a YAML config, runs scripted-bot-vs-scripted-bot episodes, runs the
same sequence a second time, and asserts every per-decision state_hash
matches bit-identically. Exits 0 on full match, 1 on first divergence.

Usage:
    python -m train.train --config experiments/configs/phase0_determinism.yaml

Phase 1+ (PPO, flat obs) will reuse this entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from xushi2.runner import EpisodeResult, run_episode


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _run_pass(sim_cfg: dict, bot_a: str, bot_b: str, episodes: int,
              base_seed: int) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for i in range(episodes):
        results.append(run_episode(sim_cfg, bot_a, bot_b, seed_override=base_seed + i))
    return results


def _assert_identical(pass_a: list[EpisodeResult], pass_b: list[EpisodeResult]) -> int:
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
    phase = config.get("phase", "unknown")
    sim_cfg = config.get("sim", {})
    run_cfg = config.get("run", {})

    episodes = int(run_cfg.get("episodes", 4))
    bot_a = str(run_cfg.get("team_a_bot", "basic"))
    bot_b = str(run_cfg.get("team_b_bot", "basic"))
    assert_determinism = bool(run_cfg.get("assert_determinism", True))
    base_seed = int(sim_cfg.get("seed", 0))

    print(f"[xushi2] phase={phase} episodes={episodes} "
          f"bots={bot_a} vs {bot_b} base_seed=0x{base_seed:x}")

    if phase == 0:
        if not assert_determinism:
            # Later phases will slot in here. For now the harness is Phase-0-only.
            print(f"[xushi2] phase {phase} not yet supported by this entrypoint")
            return 2

        pass_a = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)
        pass_b = _run_pass(sim_cfg, bot_a, bot_b, episodes, base_seed)

        rc = _assert_identical(pass_a, pass_b)
        if rc == 0:
            total = sum(len(r.decision_hashes) for r in pass_a)
            per_ep = len(pass_a[0].decision_hashes) if pass_a else 0
            print(f"[xushi2] OK: {episodes} episodes × {per_ep} decisions "
                  f"({total} hashes) all identical")
        return rc

    if phase == 2:
        from train.ppo_recurrent import train_from_config

        result = train_from_config(config)
        recurrent = float(result["recurrent"])
        feedforward = float(result["feedforward"])
        gap = recurrent - feedforward
        print(f"[phase2] recurrent_final={recurrent:.3f} "
              f"feedforward_final={feedforward:.3f} gap={gap:.3f}")
        return 0

    # Later phases will slot in here. For now the harness is Phase-0-only.
    print(f"[xushi2] phase {phase} not yet supported by this entrypoint")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
