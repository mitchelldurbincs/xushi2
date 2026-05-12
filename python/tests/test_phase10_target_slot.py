from __future__ import annotations

import numpy as np
import torch
import yaml

from envs.phase10_target_slot_mappo import (
    PHASE10_TARGET_OBS_DIM,
    TARGET_SLOT_MASK_DIM,
    Phase10TargetSlotMappoEnv,
)
from train.mappo import MappoActorCritic, make_mappo_config
from xushi2.obs_manifest import CRITIC_DIM
from _paths import config_path


def _load_config() -> dict:
    with open(
        config_path("phase10_target_slot_probe.yaml"),
        encoding="utf-8",
    ) as fh:
        return yaml.safe_load(fh)


def test_phase10_env_accepts_target_slot_and_forwards_controls() -> None:
    config = _load_config()
    env_cfg = config["env"]
    env = Phase10TargetSlotMappoEnv(
        env_cfg["sim"],
        opponent_bot=env_cfg["opponent_bot"],
        learner_team=env_cfg["learner_team"],
        reward_cfg=env_cfg["reward"],
        fog_mode=env_cfg["fog_mode"],
        visible_radius=float(env_cfg["visible_radius"]),
        map_randomization=env_cfg["map_randomization"],
    )
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (3, PHASE10_TARGET_OBS_DIM)
        assert env.action_space.shape == (3, 7)
        assert info["target_slots"].tolist() == [0, 0, 0]
        assert info["target_slot_mask"].shape == (3, TARGET_SLOT_MASK_DIM)
        assert np.array_equal(obs[:, -TARGET_SLOT_MASK_DIM:], info["target_slot_mask"])

        action = np.zeros((3, 7), dtype=np.float32)
        action[:, 6] = np.array([-1.0, 1.2, 8.0], dtype=np.float32)
        next_obs, reward, term, trunc, info = env.step(action)
        assert next_obs.shape == (3, PHASE10_TARGET_OBS_DIM)
        assert reward.shape == (3,)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert info["target_slots"].tolist() == [0, 1, 4]
        assert np.array_equal(
            next_obs[:, -TARGET_SLOT_MASK_DIM:],
            info["target_slot_mask"],
        )

        critic_obs = np.zeros(CRITIC_DIM, dtype=np.float32)
        env.build_critic_obs(critic_obs)
        assert np.all(np.isfinite(critic_obs))
    finally:
        env.close()


def test_phase10_actor_samples_target_slot_factor() -> None:
    config = _load_config()
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    obs = torch.zeros(cfg.n_agents, cfg.obs_dim)
    token_width = cfg.entity_token_count * cfg.entity_token_dim
    obs[:, token_width : token_width + cfg.entity_token_count] = 1.0
    obs[:, -cfg.target_action_dim :] = 1.0
    h = model.init_hidden(cfg.n_agents)

    action, logprob, h_next = model.sample_action(obs, h)
    assert action.shape == (cfg.n_agents, 7)
    assert logprob.shape == (cfg.n_agents,)
    assert h_next.shape == (cfg.n_agents, cfg.gru_hidden)
    target = action[:, 6]
    assert torch.all(target >= 0)
    assert torch.all(target < cfg.target_action_dim)

    greedy, _ = model.greedy_action(obs, h)
    assert greedy.shape == (cfg.n_agents, 7)


def test_phase10_actor_respects_target_slot_mask() -> None:
    config = _load_config()
    cfg = make_mappo_config(config)
    model = MappoActorCritic(cfg)
    obs = torch.zeros(cfg.n_agents, cfg.obs_dim)
    obs[:, -cfg.target_action_dim :] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    h = model.init_hidden(cfg.n_agents)

    greedy, _ = model.greedy_action(obs, h)
    assert greedy[:, 6].tolist() == [0.0, 1.0, 4.0]
    for _ in range(20):
        action, _logprob, _ = model.sample_action(obs, h)
        assert action[:, 6].tolist() == [0.0, 1.0, 4.0]
