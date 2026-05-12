"""Phase-2 memory-toy ablation gate CLI."""

from __future__ import annotations

import argparse

from eval.memory_toy_gate import evaluate_memory_toy_gate, format_result_table, load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="MemoryToy ablation gate")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0x4D454D54)
    args = parser.parse_args()

    model, config = load_checkpoint(args.checkpoint)
    result = evaluate_memory_toy_gate(
        model=model,
        config=config,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    print(format_result_table(result.per_mode))
    print(f"\ngap (normal - zero): {result.gap_normal_minus_zero:+.3f}")

    if result.passed:
        print("PHASE 2 GATE: PASS")
        return 0

    print("PHASE 2 GATE: FAIL")
    for msg in result.failure_reasons:
        print(f" - {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
