"""Neutral public environment factory for runtime specs.

This module is the env-layer entrypoint used by training code. Phase-named env
classes remain as compatibility implementations, but callers select behavior
through explicit runtime capabilities.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym


def make_memory_toy_env(episode_length: int, cue_visible_ticks: int) -> gym.Env:
    from envs.memory_toy import MemoryToyEnv

    return MemoryToyEnv(
        episode_length=int(episode_length),
        cue_visible_ticks=int(cue_visible_ticks),
    )


def make_ranger_duel_env(
    sim_cfg: dict[str, Any],
    opponent_bot: str,
    learner_team: str,
    reward_cfg: dict[str, Any],
) -> gym.Env:
    from envs.phase3_ranger import Phase3RangerEnv

    return Phase3RangerEnv(
        dict(sim_cfg),
        opponent_bot=str(opponent_bot),
        learner_team=str(learner_team),
        reward_cfg=dict(reward_cfg),
    )


def make_mappo_match_env(
    *,
    sim_cfg: dict[str, Any],
    opponent_bot: str = "basic",
    learner_team: str = "A",
    reward_cfg: dict[str, Any] | None = None,
    actor_obs: str = "flat",
    fog_mode: str = "none",
    visible_radius: float = 0.65,
    map_randomization: dict[str, Any] | None = None,
    mini_game: str | None = None,
    mini_game_cfg: dict[str, Any] | None = None,
    self_play: bool = False,
    self_play_schedule: dict[str, Any] | None = None,
    snapshot_paths: tuple[str, ...] = (),
    snapshot_league: dict[str, Any] | None = None,
    target_slot: bool = False,
    n_agents: int = 3,
) -> gym.Env:
    reward = dict(reward_cfg or {})
    mini_cfg = dict(mini_game_cfg or {})
    fog = None if fog_mode in ("", "none", None) else str(fog_mode)
    map_cfg = dict(map_randomization or {})
    league_cfg = dict(snapshot_league or {})

    if mini_game == "cap_duel":
        from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv

        return Phase4CapDuelMappoEnv(
            **mini_cfg,
            self_play_schedule=(self_play_schedule if self_play else None),
            snapshot_league=(snapshot_league if self_play else None),
        )

    if self_play:
        if mini_game not in (None, ""):
            raise ValueError("current self-play can only be combined with cap_duel mini_game")
        from envs.phase4_selfplay_mappo import Phase4CurrentSelfplayMappoEnv

        return Phase4CurrentSelfplayMappoEnv(
            dict(sim_cfg),
            reward_cfg=reward,
            self_play_schedule=self_play_schedule,
            snapshot_league=snapshot_league,
        )
    if int(n_agents) == 6:
        from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv

        return Phase11CurrentSelfplayMappoEnv(
            dict(sim_cfg),
            reward_cfg=reward,
            fog_mode=fog or "team_shared",
            visible_radius=float(visible_radius),
            map_randomization=map_cfg,
            self_play_schedule=self_play_schedule,
            snapshot_league=league_cfg,
        )

    if mini_game == "aim_only":
        from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv

        return Phase4AimOnlyMappoEnv(**mini_cfg)
    if mini_game == "combat_1v1":
        from envs.phase4_combat_1v1_mappo import Phase4Combat1v1MappoEnv

        return Phase4Combat1v1MappoEnv(**mini_cfg)
    if mini_game not in (None, ""):
        raise ValueError(f"unknown mappo mini_game {mini_game!r}")

    if target_slot:
        from envs.phase10_target_slot_mappo import Phase10TargetSlotMappoEnv

        return Phase10TargetSlotMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
            fog_mode=fog or "team_shared",
            visible_radius=float(visible_radius),
            map_randomization=map_cfg,
        )
    if tuple(snapshot_paths):
        from envs.phase9_snapshot_mappo import Phase9SnapshotMappoEnv

        return Phase9SnapshotMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
            fog_mode=fog or "team_shared",
            visible_radius=float(visible_radius),
            map_randomization=map_cfg,
            snapshot_paths=list(snapshot_paths),
            snapshot_league=league_cfg,
        )
    if map_cfg:
        from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv

        return Phase8RandomMapMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
            fog_mode=fog or "team_shared",
            visible_radius=float(visible_radius),
            map_randomization=map_cfg,
        )
    if fog is not None:
        from envs.phase7_fog_mappo import Phase7FogMappoEnv

        return Phase7FogMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
            fog_mode=fog,
            visible_radius=float(visible_radius),
        )
    if actor_obs == "entity_grid":
        from envs.phase6_grid_mappo import Phase6GridMappoEnv

        return Phase6GridMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
        )
    if actor_obs == "entity":
        from envs.phase5_entity_mappo import Phase5EntityMappoEnv

        return Phase5EntityMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
        )
    if actor_obs != "flat":
        raise ValueError(f"unknown mappo actor_obs {actor_obs!r}")

    from envs.phase4_mappo import Phase4MappoEnv

    return Phase4MappoEnv(
        dict(sim_cfg),
        opponent_bot=str(opponent_bot),
        learner_team=str(learner_team),
        reward_cfg=reward,
    )


def mappo_env_fn_from_config(env_cfg: dict[str, Any]) -> Callable[[], gym.Env]:
    cfg = dict(env_cfg)
    sim_cfg = dict(cfg.get("sim", {}))
    reward_cfg = dict(cfg.get("reward", {}))
    self_play_cfg = dict(cfg.get("self_play", {}))
    return partial(
        make_mappo_match_env,
        sim_cfg=sim_cfg,
        opponent_bot=str(cfg.get("opponent_bot", cfg.get("opponent", {}).get("kind", "basic"))),
        learner_team=str(cfg.get("learner_team", "A")),
        reward_cfg=reward_cfg,
        actor_obs=str(cfg.get("actor_obs", "flat")),
        fog_mode=str(cfg.get("fog_mode", cfg.get("features", {}).get("fog", "none"))),
        visible_radius=float(cfg.get("visible_radius", 0.65)),
        map_randomization=dict(cfg.get("map_randomization", {})),
        mini_game=(None if cfg.get("mini_game") is None else str(cfg.get("mini_game"))),
        mini_game_cfg=dict(cfg.get("mini_game_config", {})),
        self_play=bool(self_play_cfg.get("enabled", cfg.get("current_selfplay", False))),
        self_play_schedule=(
            dict(cfg.get("self_play_schedule", {}))
            if "self_play_schedule" in cfg
            else None
        ),
        snapshot_paths=tuple(str(p) for p in cfg.get("snapshot_paths", ())),
        snapshot_league=(
            dict(cfg.get("snapshot_league", {})) if "snapshot_league" in cfg else None
        ),
        target_slot=bool(cfg.get("target_slot", cfg.get("features", {}).get("target_slot", False))),
        n_agents=int(cfg.get("n_agents", cfg.get("team_size", 3))),
    )
