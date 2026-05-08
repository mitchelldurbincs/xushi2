"""Phase registry shared across training entrypoints."""

from __future__ import annotations

from functools import partial
from typing import Callable

import gymnasium as gym


def _make_phase2_env(episode_length: int, cue_visible_ticks: int):
    from envs.memory_toy import MemoryToyEnv

    return MemoryToyEnv(
        episode_length=episode_length,
        cue_visible_ticks=cue_visible_ticks,
    )


def _make_phase3_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase3_ranger import Phase3RangerEnv

    return Phase3RangerEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _make_phase4_env(
    sim_cfg: dict,
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict,
):
    from envs.phase4_mappo import Phase4MappoEnv

    return Phase4MappoEnv(
        sim_cfg,
        opponent_bot=opponent_bot,
        learner_team=learner_team,
        reward_cfg=reward_cfg,
    )


def _phase2_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    ep_len = int(env_cfg.get("episode_length", 64))
    cue_ticks = int(env_cfg.get("cue_visible_ticks", 4))
    return (
        partial(_make_phase2_env, ep_len, cue_ticks),
        {"episode_length": ep_len, "cue_visible_ticks": cue_ticks},
        int(env_cfg.get("seed_base", 0)),
    )


def _phase3_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    sim_cfg = dict(env_cfg.get("sim", {}))
    opponent_bot = str(env_cfg.get("opponent_bot", "basic"))
    learner_team = str(env_cfg.get("learner_team", "A"))
    reward_cfg = dict(env_cfg.get("reward", {}))
    return (
        partial(_make_phase3_env, sim_cfg, opponent_bot, learner_team, reward_cfg),
        {
            "sim": sim_cfg,
            "opponent_bot": opponent_bot,
            "learner_team": learner_team,
            "reward": reward_cfg,
        },
        int(env_cfg.get("seed_base", sim_cfg.get("seed", 0))),
    )


def _phase4_env_bundle(config: dict) -> tuple[Callable[[], gym.Env], dict, int]:
    env_cfg = config.get("env", {})
    sim_cfg = dict(env_cfg.get("sim", {}))
    opponent_bot = str(env_cfg.get("opponent_bot", "basic"))
    learner_team = str(env_cfg.get("learner_team", "A"))
    reward_cfg = dict(env_cfg.get("reward", {}))
    return (
        partial(_make_phase4_env, sim_cfg, opponent_bot, learner_team, reward_cfg),
        {
            "sim": sim_cfg,
            "opponent_bot": opponent_bot,
            "learner_team": learner_team,
            "reward": reward_cfg,
        },
        int(env_cfg.get("seed_base", sim_cfg.get("seed", 0))),
    )


def _phase0_seed(config: dict) -> int:
    env_cfg = config.get("env", {})
    sim_cfg = config.get("sim", {})
    return int(env_cfg.get("seed_base", sim_cfg.get("seed", 0)))


PHASE_REGISTRY: dict[int, dict] = {
    0: {
        "label": "phase0",
        "training_variants": (),
        "seed_deriver": _phase0_seed,
    },
    2: {
        "label": "phase2",
        "obs_dim": 3,
        "action_dim": 2,
        "continuous_action_dim": 2,
        "binary_action_dim": 0,
        "training_variants": ("recurrent", "feedforward"),
        "env_bundle": _phase2_env_bundle,
    },
    3: {
        "label": "phase3",
        "obs_dim": 31,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("recurrent",),
        "env_bundle": _phase3_env_bundle,
    },
    4: {
        "label": "phase4",
        "obs_dim": 31,
        "critic_obs_dim": 135,
        "n_agents": 3,
        "action_dim": 6,
        "continuous_action_dim": 3,
        "binary_action_dim": 3,
        "training_variants": ("mappo",),
        "env_bundle": _phase4_env_bundle,
    },
}


def resolve_phase(config: dict) -> tuple[int, dict]:
    raw_phase = config.get("phase", 2)
    try:
        phase = int(raw_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported phase/config shape: phase={raw_phase!r}") from exc

    spec = PHASE_REGISTRY.get(phase)
    if spec is None:
        raise ValueError(f"unsupported phase/config shape: phase={raw_phase!r}")
    return phase, spec
