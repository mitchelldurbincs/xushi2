from __future__ import annotations

import numpy as np

from envs.phase4_mappo import Phase4MappoEnv
from envs.phase5_entity_mappo import Phase5EntityMappoEnv
from envs.phase6_grid_mappo import Phase6GridMappoEnv
from envs.phase7_fog_mappo import Phase7FogMappoEnv
from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv
from envs.phase10_target_slot_mappo import Phase10TargetSlotMappoEnv


def _cfg() -> dict:
    return {
        "seed": 123,
        "round_length_seconds": 4,
        "fog_of_war_enabled": False,
        "randomize_map": False,
        "action_repeat": 3,
        "mechanics": {"respawn_ticks": 120},
    }


def _envs():
    cfg = _cfg()
    return [
        Phase4MappoEnv(cfg, opponent_bot="noop"),
        Phase5EntityMappoEnv(cfg, opponent_bot="noop"),
        Phase6GridMappoEnv(cfg, opponent_bot="noop"),
        Phase7FogMappoEnv(cfg, opponent_bot="noop"),
        Phase8RandomMapMappoEnv(cfg, opponent_bot="noop"),
        Phase10TargetSlotMappoEnv(cfg, opponent_bot="noop"),
    ]


def test_reset_step_shape_parity_and_replay_fields() -> None:
    for env in _envs():
        obs, info = env.reset(seed=9)
        assert obs.shape == env.observation_space.shape
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, reward, *_rest, step_info = env.step(action)
        assert next_obs.shape == env.observation_space.shape
        assert reward.shape == (3,)
        assert "state_hash" in info and "tick" in info
        assert "state_hash" in step_info and "tick" in step_info
        env.close()


def test_seed_determinism_state_hash_parity() -> None:
    for env_a in _envs():
        env_b = env_a.__class__(_cfg(), opponent_bot="noop")
        try:
            _, info_a = env_a.reset(seed=77)
            _, info_b = env_b.reset(seed=77)
            assert info_a["state_hash"] == info_b["state_hash"]
            action = np.zeros(env_a.action_space.shape, dtype=np.float32)
            for _ in range(3):
                _, _, _, _, ia = env_a.step(action)
                _, _, _, _, ib = env_b.step(action)
                assert ia["state_hash"] == ib["state_hash"]
        finally:
            env_a.close()
            env_b.close()
