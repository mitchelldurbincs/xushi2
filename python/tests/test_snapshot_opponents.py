"""Tests for 3-agent snapshot opponents: SnapshotPolicy obs routing for
phase-4 multi-enemy checkpoints, the snapshot:<path> opponent-mix entries,
and the factory wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.phase4_mappo import Phase4MappoEnv
from envs.runtime_factory import mappo_env_fn_from_config
from train.opponent_mix import parse_opponent_bot_mix
from xushi2.snapshot_policy import SnapshotPolicy

_CKPT = Path(__file__).resolve().parents[2] / (
    "data/checkpoints/phase4_v5_upd300_stochastic_600t_converter.pt"
)

pytestmark = pytest.mark.skipif(
    not _CKPT.exists(), reason="reference checkpoint not present"
)


def _sim_cfg() -> dict:
    return {
        "seed": 0xD1CEDA7A,
        "round_length_seconds": 5,
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


def test_snapshot_policy_routes_phase4_multi_enemy_obs():
    policy = SnapshotPolicy(_CKPT)
    assert policy.cfg.entity_token_count > 3
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="noop")
    env.reset(seed=5)
    actions = policy.act(env._sim, [3, 4, 5])
    env.close()
    assert actions.shape[0] == 3
    assert actions.shape[1] >= 6
    assert np.isfinite(actions).all()


def test_set_opponent_bot_snapshot_applies_at_reset():
    env = Phase4MappoEnv(sim_cfg=_sim_cfg(), opponent_bot="weak_basic_v2")
    env.set_opponent_bot(f"snapshot:{_CKPT}")
    env.reset(seed=7)
    assert env._opponent_bot == "snapshot"
    assert env._opponent_policy is not None
    obs, _r, _te, _tr, info = env.step(
        np.zeros((env.n_agents, env.action_dim), dtype=np.float32)
    )
    assert np.isfinite(obs).all()
    # Switching back to a scripted bot clears the frozen policy.
    env.set_opponent_bot("weak_basic_v2")
    env.reset(seed=8)
    assert env._opponent_bot == "weak_basic_v2"
    assert env._opponent_policy is None
    env.close()


def test_mix_accepts_snapshot_entries():
    mix = parse_opponent_bot_mix(
        {f"snapshot:{_CKPT}": 0.5, "weak_basic_v2": 0.5}
    )
    assert len(mix) == 2
    with pytest.raises(ValueError, match="requires a checkpoint path"):
        parse_opponent_bot_mix({"snapshot:": 1.0})


def test_factory_builds_snapshot_opponent_env():
    env_fn = mappo_env_fn_from_config(
        {
            "sim": _sim_cfg(),
            "opponent_bot": "snapshot",
            "snapshot_paths": [str(_CKPT)],
            "learner_team": "A",
            "actor_obs": "multi_enemy_entity_grid",
        }
    )
    env = env_fn()
    env.reset(seed=9)
    obs, _r, _te, _tr, _info = env.step(
        np.zeros((3, 6), dtype=np.float32)
    )
    assert np.isfinite(obs).all()
    env.close()
