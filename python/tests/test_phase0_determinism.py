"""Phase-0 determinism test.

Same seed, same bots → identical per-decision state_hash trajectory.
Also sanity-checks that the sim actually advances (first hash ≠ last).
"""

from __future__ import annotations

import pytest

from xushi2.runner import run_episode

SIM_CFG = {
    "seed": 0xD1CEDA7A,
    "round_length_seconds": 30,
    "fog_of_war_enabled": False,
    "randomize_map": False,
    # Must match experiments/configs/phase0_determinism.yaml.
    "mechanics": {
        "revolver_damage_centi_hp": 7500,
        "revolver_fire_cooldown_ticks": 15,
        "revolver_hitbox_radius": 0.75,
        "respawn_ticks": 240,
    },
}
# round_length 30s × 30 Hz / action_repeat 3 = 300 decisions.
EXPECTED_DECISIONS = 300


@pytest.mark.parametrize("bot_a,bot_b", [
    ("basic", "basic"),
    ("walk_to_objective", "walk_to_objective"),
    ("hold_and_shoot", "hold_and_shoot"),
    ("noop", "noop"),
])
def test_intra_process_determinism(bot_a: str, bot_b: str) -> None:
    a = run_episode(SIM_CFG, bot_a, bot_b)
    b = run_episode(SIM_CFG, bot_a, bot_b)
    assert a.decision_hashes == b.decision_hashes
    assert a.final_tick == b.final_tick
    assert len(a.decision_hashes) == EXPECTED_DECISIONS


def test_basic_bot_evolves_state() -> None:
    r = run_episode(SIM_CFG, "basic", "basic")
    # Sim must actually change state — first hash differs from last.
    assert r.decision_hashes[0] != r.decision_hashes[-1]


def test_basic_differs_from_noop() -> None:
    basic = run_episode(SIM_CFG, "basic", "basic")
    noop = run_episode(SIM_CFG, "noop", "noop")
    # Must diverge somewhere; bots should actually do something different.
    assert basic.decision_hashes != noop.decision_hashes


def test_different_seeds_diverge() -> None:
    a = run_episode(SIM_CFG, "basic", "basic", seed_override=1)
    b = run_episode(SIM_CFG, "basic", "basic", seed_override=2)
    assert a.decision_hashes != b.decision_hashes


def test_invalid_bot_name_rejected() -> None:
    with pytest.raises(ValueError):
        run_episode(SIM_CFG, "not_a_real_bot", "basic")
