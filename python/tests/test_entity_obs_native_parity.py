"""Parity: native ObservationEngine vs the legacy Python entity-obs assembly.

The migration's safety net (plan: native obs ownership, Phase 2). Both paths
are driven from the same live env step stream; every observation must agree
to 1e-6 over all rows × 3167 floats. Scenarios are scripted so the fog
branches actually fire: teams approach (enemies become visible), then
retreat (visibility drops, last-seen stale markers appear).

Once the legacy path is deleted (Phase 3), the golden fixtures written by
python/tools/dump_entity_obs_golden.py take over regression duty.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv
from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv
from xushi2 import xushi2_cpp as _cpp
from xushi2.entity_obs_native import (
    make_obs_config,
    phase4_multi_enemy_obs_config,
    snapshot_obs_config,
)
from xushi2.multi_enemy_obs import (
    ENTITY_TOKEN_DIM,
    MULTI_ENEMY_ENTITY_GRID_OBS_DIM,
    MULTI_ENEMY_TOKEN_COUNT,
)

_TOLERANCE = 1e-6


def _phase11_sim_cfg() -> dict:
    return {
        "round_length_seconds": 20,
        "action_repeat": 3,
        "seed": 991,
        # Native sim fog ON so the LoS component of the visibility rule is
        # live (walls/covers from map randomization occlude).
        "fog_of_war_enabled": True,
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }


def _phase11_env(fog_mode: str, native: bool) -> Phase11CurrentSelfplayMappoEnv:
    return Phase11CurrentSelfplayMappoEnv(
        _phase11_sim_cfg(),
        reward_cfg={},
        fog_mode=fog_mode,
        visible_radius=0.65,
        # Defaults place 4 cover pillars + 2 walls per episode, so occlusion
        # is exercised without bespoke geometry.
        map_randomization={},
        native_entity_obs=native,
    )


def _approach_retreat_action(step: int) -> np.ndarray:
    """Teams close on each other, then split — flips visibility both ways."""
    act = np.zeros((6, 6), dtype=np.float32)
    # 4.2 u/s × 0.1 s/decision = 0.42 u/decision/team; the spawn gap is 40 u
    # and the 0.65 radius needs ≲16 u, so ~40 approach decisions close it.
    # Actions are team-relative (Team B's frame is mirrored), so move_y=+1
    # walks BOTH teams toward each other.
    direction = 1.0 if step < 42 else -1.0
    act[0:3, 1] = direction
    act[3:6, 1] = direction
    act[0, 0] = 0.4           # some lateral drift + aim motion for variety
    act[4, 0] = -0.3
    act[:, 2] = 0.1
    act[1, 3] = 1.0           # a bit of gunfire
    act[5, 3] = 1.0
    return act


def _split_tokens(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = obs.reshape(-1, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
    token_width = MULTI_ENEMY_TOKEN_COUNT * ENTITY_TOKEN_DIM
    tokens = flat[:, :token_width].reshape(
        -1, MULTI_ENEMY_TOKEN_COUNT, ENTITY_TOKEN_DIM
    )
    mask = flat[:, token_width : token_width + MULTI_ENEMY_TOKEN_COUNT]
    return tokens, mask


_GRID_FLAT = 3 * 32 * 32


def _dilate(grid: np.ndarray) -> np.ndarray:
    """3x3 max-dilation per channel (channels, 32, 32)."""
    padded = np.pad(grid, ((0, 0), (1, 1), (1, 1)), mode="constant")
    out = grid.copy()
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out = np.maximum(out, padded[:, dy : dy + 32, dx : dx + 32])
    return out


def _assert_obs_parity(obs_n: np.ndarray, obs_l: np.ndarray, context: str) -> None:
    """Tokens and mask must agree to 1e-6. Grid marks may shift by ONE pixel.

    The legacy adapter normalizes enemy positions with double-precision map
    bounds; the native path uses the sim's float32 bounds. The ~1-ulp
    difference is far inside the 1e-6 token tolerance, but the discrete
    grid rounding amplifies it to a full pixel when a mark lands within
    ~1e-7 of a cell boundary. Accept only that artifact: rare, equal-valued
    marks at 8-neighbor positions.
    """
    flat_n = obs_n.reshape(-1, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
    flat_l = obs_l.reshape(-1, MULTI_ENEMY_ENTITY_GRID_OBS_DIM)
    np.testing.assert_allclose(
        flat_n[:, :-_GRID_FLAT], flat_l[:, :-_GRID_FLAT],
        atol=_TOLERANCE, rtol=0, err_msg=f"tokens/mask diverged {context}",
    )
    for row in range(flat_n.shape[0]):
        g_n = flat_n[row, -_GRID_FLAT:].reshape(3, 32, 32)
        g_l = flat_l[row, -_GRID_FLAT:].reshape(3, 32, 32)
        mismatched = np.abs(g_n - g_l) > _TOLERANCE
        if not mismatched.any():
            continue
        assert mismatched.sum() <= 4, (
            f"{int(mismatched.sum())} grid cells diverged (row {row}, "
            f"{context}) — more than a boundary-rounding artifact"
        )
        assert np.all(g_n <= _dilate(g_l) + _TOLERANCE) and np.all(
            g_l <= _dilate(g_n) + _TOLERANCE
        ), (
            f"grid marks diverged by more than one pixel (row {row}, "
            f"{context})"
        )


@pytest.mark.parametrize("fog_mode", ["team_shared", "per_agent"])
@pytest.mark.parametrize("seed", [7, 4242])
def test_phase11_native_parity(fog_mode: str, seed: int) -> None:
    legacy = _phase11_env(fog_mode, native=False)
    native = _phase11_env(fog_mode, native=True)
    saw_visible = False
    saw_stale = False
    try:
        obs_l, _ = legacy.reset(seed=seed)
        obs_n, _ = native.reset(seed=seed)
        _assert_obs_parity(obs_n, obs_l, "at reset")

        for step in range(70):
            act = _approach_retreat_action(step)
            obs_l, rew_l, term_l, trunc_l, _ = legacy.step(act)
            obs_n, rew_n, term_n, trunc_n, _ = native.step(act)
            assert (term_l, trunc_l) == (term_n, trunc_n)
            np.testing.assert_allclose(rew_n, rew_l, atol=0, rtol=0)
            _assert_obs_parity(obs_n, obs_l, f"at step {step}")
            tokens, mask = _split_tokens(obs_n)
            enemy_tokens = tokens[:, 1:4, :]
            enemy_mask = mask[:, 1:4]
            saw_visible |= bool(
                ((enemy_mask > 0.5) & (enemy_tokens[:, :, 7] > 0.5)).any()
            )
            saw_stale |= bool(
                ((enemy_mask > 0.5) & (enemy_tokens[:, :, 17] == 0.5)).any()
            )
            if term_l or trunc_l:
                break
    finally:
        legacy.close()
        native.close()
    assert saw_visible, "scenario never made an enemy visible — parity vacuous"
    assert saw_stale, "scenario never produced a stale marker — parity vacuous"


@pytest.mark.parametrize("seed", [3, 991])
def test_phase4_multi_enemy_native_parity(seed: int) -> None:
    sim_cfg = {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 20,
        "fog_of_war_enabled": True,
        "randomize_map": False,
        "action_repeat": 3,
        # Same disc the C++ leak tests use: blocks slot 0's diagonals,
        # leaves the x=25 vertical clear.
        "cover_circles": [{"x": 20.0, "y": 15.0, "radius": 4.0}],
        "mechanics": {
            "revolver_damage_centi_hp": 7500,
            "revolver_fire_cooldown_ticks": 15,
            "revolver_hitbox_radius": 0.75,
            "respawn_ticks": 240,
        },
    }
    legacy = Phase4MultiEnemyMappoEnv(
        dict(sim_cfg), opponent_bot="basic", native_entity_obs=False
    )
    native = Phase4MultiEnemyMappoEnv(
        dict(sim_cfg), opponent_bot="basic", native_entity_obs=True
    )
    try:
        obs_l, _ = legacy.reset(seed=seed)
        obs_n, _ = native.reset(seed=seed)
        np.testing.assert_allclose(obs_n, obs_l, atol=_TOLERANCE, rtol=0)

        rng = np.random.default_rng(seed)
        for step in range(40):
            act = rng.uniform(-1.0, 1.0, size=(3, 6)).astype(np.float32)
            act[:, 1] = 1.0  # push learners toward the enemy side
            obs_l, _rew_l, term_l, trunc_l, _ = legacy.step(act)
            obs_n, _rew_n, term_n, trunc_n, _ = native.step(act)
            assert (term_l, trunc_l) == (term_n, trunc_n)
            _assert_obs_parity(obs_n, obs_l, f"at step {step}")
            if term_l or trunc_l:
                break
    finally:
        legacy.close()
        native.close()


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
    # A dup-C (phase < 7) snapshot's engine must produce, for the OPPOSITE
    # team's slots, exactly what the Phase-4 multi-enemy training env's
    # native path produces for its learners — training semantics, per
    # checkpoint. Both engines are stateless here (no last-seen), so they
    # can be driven side by side on one sim.
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


def test_snapshot_policy_native_flag(tmp_path) -> None:
    # Fabricated multi-enemy checkpoint (same recipe as test_phase9_snapshot):
    # native flag on must build a per-checkpoint engine and act() must run
    # end-to-end; flag off keeps the legacy conversion.
    import torch
    import yaml

    from train.mappo import MappoActorCritic, make_mappo_config
    from train.phases import resolve_phase
    from xushi2.runner import _build_config
    from xushi2.snapshot_policy import SnapshotPolicy
    from _paths import config_path

    with open(
        config_path("phase4/probe/phase4_mappo_multi_enemy_actor_obs_v1.yaml"),
        encoding="utf-8",
    ) as fh:
        config = yaml.safe_load(fh)
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    _phase, spec = resolve_phase(config)
    _env_fn, ckpt_env_cfg, _seed = spec["env_bundle"](config)
    ckpt_path = tmp_path / "snapshot.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"phase": 8, "env": ckpt_env_cfg, "mappo": cfg.__dict__},
        },
        ckpt_path,
    )

    sim_cfg = dict(ckpt_env_cfg.get("sim", {}))
    match_cfg = _build_config(sim_cfg)
    match_cfg.team_size = 3
    sim = _cpp.Sim(match_cfg)

    legacy = SnapshotPolicy(ckpt_path)
    assert legacy._obs_engine is None
    native = SnapshotPolicy(ckpt_path, native_entity_obs=True)
    assert native._obs_engine is not None
    # Phase >= 7 checkpoints serve with their stored fog semantics and
    # last-seen ON (the skew fix).
    engine_cfg = snapshot_obs_config(8, ckpt_env_cfg)
    assert engine_cfg.last_seen_enabled is True

    native.reset(batch_size=3)
    actions = native.act(sim, (3, 4, 5))
    assert actions.shape[0] == 3
    assert actions.shape[1] >= 6
    assert np.all(np.isfinite(actions))
