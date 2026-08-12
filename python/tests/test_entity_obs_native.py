"""Native entity-obs path: config mapping, snapshot semantics, and the
end-to-end leak seal.

The native-vs-legacy parity suite that guarded the Phase-2 migration was
deleted with the legacy Python obs assembly at the Phase-3 cutover; the
survivors that don't depend on the legacy path live here. The heavy
counterfactual leak matrix is C++ (tests/observations/test_entity_obs_leak.cpp);
`test_hidden_enemy_state_never_reaches_env_step_obs` below is the single
Python-side integration seal over the FFI.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.entity_obs_native import (
    make_obs_config,
    phase4_multi_enemy_obs_config,
    snapshot_obs_config,
)
from xushi2.multi_enemy_obs import MULTI_ENEMY_ENTITY_GRID_OBS_DIM


def test_snapshot_obs_config_mapping() -> None:
    # Phase >= 7 checkpoints: the checkpoint's stored fog semantics, with
    # last-seen markers ON (fixing the legacy no-last-seen serving skew).
    cfg = snapshot_obs_config(11, {"fog_mode": "per_agent", "visible_radius": 0.5})
    assert cfg.fog_mode == _cpp.FogMode.PerAgent
    assert cfg.visible_radius == pytest.approx(0.5)
    assert cfg.last_seen_enabled is True
    assert cfg.zero_hidden_token_markers is False

    # Missing fog keys fall back to the legacy serving defaults.
    cfg = snapshot_obs_config(8, {})
    assert cfg.fog_mode == _cpp.FogMode.TeamShared
    assert cfg.visible_radius == pytest.approx(0.65)

    # Phase < 7 multi-enemy checkpoints: training-time dup-C semantics —
    # native LoS only, no radius, no last-seen, hidden tokens zeroed.
    cfg = snapshot_obs_config(4, {"fog_mode": "team_shared", "visible_radius": 0.65})
    assert cfg.fog_mode == _cpp.FogMode.PerAgent
    assert np.isnan(cfg.visible_radius)
    assert cfg.last_seen_enabled is False
    assert cfg.zero_hidden_token_markers is True


def test_make_obs_config_rejects_unknown_fog_mode() -> None:
    with pytest.raises(ValueError, match="unknown fog_mode"):
        make_obs_config(
            fog_mode="everything_visible",
            visible_radius=None,
            last_seen_enabled=False,
        )


def test_snapshot_engine_matches_training_env_for_dup_c_checkpoints() -> None:
    # A dup-C (phase < 7) snapshot's engine must produce, for any slot,
    # exactly what the Phase-4 multi-enemy training env's engine produces —
    # training semantics, per checkpoint. Both engines are stateless here
    # (no last-seen), so they can be driven side by side on one sim.
    cfg = _cpp.MatchConfig()
    cfg.seed = 77
    cfg.round_length_seconds = 20
    cfg.fog_of_war_enabled = True
    cfg.team_size = 3
    cover = _cpp.CoverCircle()
    cover.center.x = 20.0
    cover.center.y = 15.0
    cover.radius = 4.0
    cfg.cover_circles = [cover]
    mech = _cpp.Phase1MechanicsConfig()
    mech.revolver_damage_centi_hp = 7500
    mech.revolver_fire_cooldown_ticks = 15
    mech.revolver_hitbox_radius = 0.75
    mech.respawn_ticks = 240
    cfg.mechanics = mech
    sim = _cpp.Sim(cfg)

    snapshot_engine = _cpp.ObservationEngine(snapshot_obs_config(4, {}))
    env_engine = _cpp.ObservationEngine(phase4_multi_enemy_obs_config())

    buf_a = np.zeros(MULTI_ENEMY_ENTITY_GRID_OBS_DIM, dtype=np.float32)
    buf_b = np.zeros(MULTI_ENEMY_ENTITY_GRID_OBS_DIM, dtype=np.float32)
    for step in range(30):
        actions = []
        for slot in range(6):
            a = _cpp.Action()
            a.move_y = 1.0 if slot < 3 else -1.0
            a.aim_delta = 0.05 * (slot + 1)
            actions.append(a)
        sim.step_decision(actions)
        for slot in range(6):
            snapshot_engine.build_entity_obs(sim, slot, buf_a)
            env_engine.build_entity_obs(sim, slot, buf_b)
            np.testing.assert_array_equal(
                buf_a, buf_b, err_msg=f"slot {slot} step {step}"
            )


def test_hidden_enemy_state_never_reaches_env_step_obs() -> None:
    # The end-to-end FFI seal: two live phase11 envs whose action streams
    # diverge ONLY in enemy state hidden from the Team-A viewers (a small
    # visibility radius keeps every enemy out of range; the divergence is
    # aim-spin, which never moves anyone). Rows 0-2 of env.step's obs must
    # stay byte-identical; rows 3-5 (the divergent team's own view of
    # itself) MUST differ — the positive control proving the assertion is
    # not vacuous.
    def make_env() -> Phase11CurrentSelfplayMappoEnv:
        return Phase11CurrentSelfplayMappoEnv(
            {
                "round_length_seconds": 10,
                "action_repeat": 3,
                "seed": 991,
                "fog_of_war_enabled": True,
                "mechanics": {
                    "revolver_damage_centi_hp": 7500,
                    "revolver_fire_cooldown_ticks": 15,
                    "revolver_hitbox_radius": 0.75,
                    "respawn_ticks": 240,
                },
            },
            reward_cfg={},
            fog_mode="per_agent",
            visible_radius=0.3,
            map_randomization={},
        )

    env_idle = make_env()
    env_spin = make_env()
    saw_b_divergence = False
    try:
        obs_idle, _ = env_idle.reset(seed=42)
        obs_spin, _ = env_spin.reset(seed=42)
        np.testing.assert_array_equal(obs_idle, obs_spin)

        idle = np.zeros((6, 6), dtype=np.float32)
        spin = np.zeros((6, 6), dtype=np.float32)
        spin[3:6, 2] = 0.7  # Team B twirls its aim; nobody moves

        for step in range(15):
            obs_idle, _r, _t, _tr, _ = env_idle.step(idle)
            obs_spin, _r, _t, _tr, _ = env_spin.step(spin)
            for viewer in range(3):
                assert not any(
                    env_spin._obs_engine.visible_enemies(env_spin._sim, viewer)
                ), f"precondition: enemies must stay outside the radius (step {step})"
            np.testing.assert_array_equal(
                obs_idle[:3],
                obs_spin[:3],
                err_msg=(
                    f"hidden enemy aim reached a Team-A observation at step {step}"
                ),
            )
            saw_b_divergence |= not np.array_equal(obs_idle[3:], obs_spin[3:])
    finally:
        env_idle.close()
        env_spin.close()
    assert saw_b_divergence, (
        "positive control: Team B's own rows must reflect its aim divergence"
    )
