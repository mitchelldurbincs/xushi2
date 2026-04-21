"""Evaluation entrypoint.

Phase 0: runs one scripted-vs-scripted match and prints a short summary.
With --dump-golden, writes the per-decision hash trajectory to stdout (one
hex hash per line) for committing as a golden-replay artifact.
"""

from __future__ import annotations

import argparse

from xushi2.runner import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="xushi2 evaluation entrypoint")
    parser.add_argument("--policy", type=str, default=None,
                        help="Policy checkpoint path (unused in Phase 0)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xD1CEDA7A)
    parser.add_argument("--bot-a", type=str, default="basic")
    parser.add_argument("--bot-b", type=str, default="basic")
    parser.add_argument("--round-length-seconds", type=int, default=30)
    parser.add_argument("--dump-golden", action="store_true",
                        help="Print hash trajectory (one hex per line) to stdout")
    # Phase-1 mechanics. Required — no defaults. Match the values in
    # experiments/configs/phase0_determinism.yaml to reproduce its golden.
    parser.add_argument("--revolver-damage-centi-hp", type=int, required=True)
    parser.add_argument("--revolver-fire-cooldown-ticks", type=int, required=True)
    parser.add_argument("--revolver-hitbox-radius", type=float, required=True)
    parser.add_argument("--respawn-ticks", type=int, required=True)
    args = parser.parse_args()

    sim_cfg = {
        "seed": args.seed,
        "round_length_seconds": args.round_length_seconds,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "mechanics": {
            "revolver_damage_centi_hp": args.revolver_damage_centi_hp,
            "revolver_fire_cooldown_ticks": args.revolver_fire_cooldown_ticks,
            "revolver_hitbox_radius": args.revolver_hitbox_radius,
            "respawn_ticks": args.respawn_ticks,
        },
    }

    for ep_idx in range(args.episodes):
        r = run_episode(sim_cfg, args.bot_a, args.bot_b,
                        seed_override=args.seed + ep_idx)
        if args.dump_golden:
            for h in r.decision_hashes:
                print(f"{h:016x}")
        else:
            winner_str = {0: "draw", 1: "A", 2: "B"}.get(r.winner, "?")
            print(f"episode={ep_idx} seed=0x{args.seed + ep_idx:x} "
                  f"decisions={len(r.decision_hashes)} final_tick={r.final_tick} "
                  f"kills=A{r.team_a_kills}/B{r.team_b_kills} winner={winner_str} "
                  f"first_hash=0x{r.decision_hashes[0]:016x} "
                  f"last_hash=0x{r.decision_hashes[-1]:016x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
