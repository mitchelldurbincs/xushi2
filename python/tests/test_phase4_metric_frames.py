"""Frame-consistency tests for the Phase 4 diagnostic metrics.

The critic tensor stores own-team slots as map-normalized team-frame
coordinates (they are actor-observation mirrors) and enemy slots in raw
world units. ``Phase4MappoEnv._slot_position`` used to return both without
converting, so every bearing and distance derived from it mixed percentages
with metres.

That mattered beyond reporting: ``team_a_aim_error_rad`` is derived from
``_nearest_visible_target`` and is used as a hard gate in
``composition_rehearsal``, where failing it sets ``total_updates = 0`` and
turns a training run into a no-op.

These tests pin the conversion against hand-computed geometry, and include
the discriminating case that the mixed-frame version got wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from xushi2.multi_enemy_obs import normalize_world_for_team
from xushi2.obs_manifest import CRITIC_DIM, critic_field_slice

# Default arena; centre is (25, 25).
_MAP = {"min_x": 0.0, "min_y": 0.0, "max_x": 50.0, "max_y": 50.0}
_CENTRE = (25.0, 25.0)


def _sim_cfg(round_length: int = 60) -> dict:
    return {
        "seed": 7,
        "round_length_seconds": round_length,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


@pytest.fixture()
def env():
    e = Phase4MappoEnv(_sim_cfg(), opponent_bot="noop")
    e.reset(seed=7)
    yield e
    e.close()


def _synthetic_critic(
    *,
    own_world: tuple[float, float],
    own_aim_rad: float,
    enemies_world: dict[int, tuple[float, float]],
) -> np.ndarray:
    """Build a critic tensor with hand-placed geometry.

    Own-team slots are written normalized (as the real builder does); enemy
    slots are written in world units (as the real builder does).
    """
    critic = np.zeros(CRITIC_DIM, dtype=np.float32)

    critic[critic_field_slice("slot0/own_hp")] = 1.0
    critic[critic_field_slice("slot0/own_position")] = normalize_world_for_team(
        np.array(own_world, dtype=np.float32), _MAP, team_b_view=False
    )
    # own_aim_unit is (sin, cos) per the manifest.
    critic[critic_field_slice("slot0/own_aim_unit")] = np.array(
        [math.sin(own_aim_rad), math.cos(own_aim_rad)], dtype=np.float32
    )

    for enemy_idx, world in enemies_world.items():
        critic[critic_field_slice(f"enemy{enemy_idx}/alive_flag")] = 1.0
        critic[critic_field_slice(f"enemy{enemy_idx}/world_position")] = np.array(
            world, dtype=np.float32
        )
    return critic


def test_slot_position_world_round_trips_own_team(env) -> None:
    world = (12.0, 37.0)
    critic = _synthetic_critic(own_world=world, own_aim_rad=0.0, enemies_world={})
    got = env._slot_position_world(critic, 0)
    np.testing.assert_allclose(got, world, atol=1e-4)


def test_slot_position_world_passes_enemy_block_through(env) -> None:
    critic = _synthetic_critic(own_world=_CENTRE, own_aim_rad=0.0, enemies_world={0: (3.5, 44.25)})
    got = env._slot_position_world(critic, 3)
    np.testing.assert_allclose(got, (3.5, 44.25), atol=1e-5)


def test_aim_error_is_zero_when_aimed_exactly_at_enemy(env) -> None:
    # Agent at centre, enemy 15u straight up (+y). Bearing is pi/2.
    critic = _synthetic_critic(
        own_world=_CENTRE,
        own_aim_rad=math.pi / 2.0,
        enemies_world={0: (25.0, 40.0)},
    )
    target_slot, aim_error = env._nearest_visible_target(critic, 0)
    assert target_slot == 3
    assert aim_error == pytest.approx(0.0, abs=1e-5)


def test_mixed_frame_regression_case(env) -> None:
    """The exact case the old two-frame accessor got wrong.

    Own world (25, 25) normalizes to (0, 0). The old code subtracted that
    from the enemy's raw world (25, 40), giving rel = (25, 40) and a bearing
    of atan2(40, 25) = 1.0122 rad instead of the true pi/2. Aiming exactly at
    the enemy therefore reported ~0.559 rad of error instead of zero.
    """
    critic = _synthetic_critic(
        own_world=_CENTRE,
        own_aim_rad=math.pi / 2.0,
        enemies_world={0: (25.0, 40.0)},
    )
    _slot, aim_error = env._nearest_visible_target(critic, 0)

    mixed_frame_bearing = math.atan2(40.0, 25.0)
    mixed_frame_error = abs(math.pi / 2.0 - mixed_frame_bearing)
    assert mixed_frame_error == pytest.approx(0.5586, abs=1e-3)  # pin the old value
    assert aim_error == pytest.approx(0.0, abs=1e-5)
    assert abs(aim_error - mixed_frame_error) > 0.5


def test_nearest_visible_target_picks_smallest_aim_error(env) -> None:
    # Aim along +x. enemy1 is nearly on that bearing; the others are not.
    critic = _synthetic_critic(
        own_world=_CENTRE,
        own_aim_rad=0.0,
        enemies_world={
            0: (25.0, 40.0),  # +y  -> pi/2 off
            1: (40.0, 25.5),  # +x  -> nearly aligned
            2: (10.0, 25.0),  # -x  -> pi off
        },
    )
    target_slot, aim_error = env._nearest_visible_target(critic, 0)
    assert target_slot == 4  # enemy index 1 -> slot 4
    assert aim_error < 0.1


def test_distances_are_world_scale_not_normalized(env) -> None:
    """A 15u separation must read as ~15, not ~0.6 (normalized) or ~47 (mixed)."""
    critic = _synthetic_critic(
        own_world=_CENTRE, own_aim_rad=math.pi / 2.0, enemies_world={0: (25.0, 40.0)}
    )
    own = env._slot_position_world(critic, 0)
    enemy = env._slot_position_world(critic, 3)
    assert float(np.linalg.norm(enemy - own)) == pytest.approx(15.0, abs=1e-4)


def test_on_point_nearest_distance_equals_true_minimum() -> None:
    """The reported nearest distance must be the minimum over the enemy team.

    The old loop fused "count an enemy in LoS" and "find the nearest enemy"
    into one `break`, so the scan stopped at the first visible enemy and
    reported the minimum over a prefix instead.
    """
    e = Phase4MappoEnv(_sim_cfg(), opponent_bot="walk_to_objective")
    try:
        e.reset(seed=7)
        walk_in = np.zeros((3, 6), dtype=np.float32)
        walk_in[:, 1] = 1.0  # team-relative +y drives both sides toward centre

        for _ in range(120):
            e.step(walk_in)
            metrics = e._objective_metrics_after_step(e._objective_snapshot())
            if metrics["on_point_nearest_enemy_distance_count_a"] > 0:
                break
        else:
            pytest.skip("never established on-point contact within the step budget")

        reported = (
            metrics["on_point_nearest_enemy_distance_sum_a"]
            / metrics["on_point_nearest_enemy_distance_count_a"]
        )

        # Recompute the truth independently, in world units, over all alive
        # enemies -- not just the ones up to the first visible.
        critic = np.zeros(CRITIC_DIM, dtype=np.float32)
        e.build_critic_obs(critic)
        best_per_agent = []
        for own_slot in range(0, 3):
            if not e._slot_alive(critic, own_slot):
                continue
            own = e._slot_position_world(critic, own_slot)
            dists = [
                float(np.linalg.norm(e._slot_position_world(critic, enemy) - own))
                for enemy in range(3, 6)
                if e._slot_alive(critic, enemy)
            ]
            if dists:
                best_per_agent.append(min(dists))
        assert best_per_agent, "expected at least one alive Team A agent"

        # The reported mean is over on-point agents only, so it must be at
        # least the smallest true nearest-distance and no more than the largest.
        assert min(best_per_agent) - 1e-3 <= reported <= max(best_per_agent) + 1e-3
    finally:
        e.close()


def test_contested_majority_is_pure_and_requires_contest() -> None:
    """Majority requires both teams present; ties and empty sides are None."""
    make = dict.fromkeys(
        [
            "tick",
            "team_a_score_ticks",
            "team_b_score_ticks",
            "cap_progress_ticks",
            "alive_a",
            "alive_b",
        ],
        0,
    )

    def snap(on_a: int, on_b: int) -> dict:
        return {**make, "alive_on_point_a": on_a, "alive_on_point_b": on_b}

    fn = Phase4MappoEnv._contested_majority_team
    assert fn(snap(2, 1)) == "A"
    assert fn(snap(1, 2)) == "B"
    assert fn(snap(1, 1)) is None  # contested but tied
    assert fn(snap(3, 0)) is None  # uncontested is not a contested majority
    assert fn(snap(0, 0)) is None


def test_fire_and_damage_share_one_contest_state(monkeypatch) -> None:
    """Both metric halves must be attributed to the same decision window.

    _attach_damage_metrics runs after step_decision. It used to re-derive the
    majority from post-step state while _combat_metrics_before_step derived it
    from pre-step state, so a fight that flipped the majority mid-window was
    counted under two different definitions.
    """
    e = Phase4MappoEnv(_sim_cfg(), opponent_bot="noop")
    try:
        e.reset(seed=7)
        seen: list[str | None] = []
        real = Phase4MappoEnv._contested_majority_team

        def spy(snapshot):
            value = real(snapshot)
            seen.append(value)
            return value

        monkeypatch.setattr(Phase4MappoEnv, "_contested_majority_team", staticmethod(spy))

        act = np.zeros((3, 6), dtype=np.float32)
        act[:, 3] = 1.0  # fire, so the combat-metric path runs
        e.step(act)

        # Exactly one derivation per step, shared by both consumers.
        assert len(seen) == 1
    finally:
        e.close()
