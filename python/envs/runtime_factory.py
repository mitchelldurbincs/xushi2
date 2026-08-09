"""Neutral public environment factory for runtime specs.

This module is the env-layer entrypoint used by training code. Phase-named env
classes remain as compatibility implementations, but callers select behavior
through explicit runtime capabilities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym

_LOGGER = logging.getLogger(__name__)

# Which parameters each env variant actually honors. Anything else the caller
# supplies is dropped, and dropping used to be silent: a config asking for
# team-shared fog on a self-play run produced a full run with fog off and
# metrics that looked entirely plausible.
#
# Two grades of dropped parameter, because they are not equally serious:
#
#  - Semantic ones change what is being simulated or what shape the policy
#    sees. Dropping them invalidates the experiment, so they raise.
#  - The rest only make the config a misleading description of the run (a
#    mini-game that ignores opponent_bot still runs the intended mini-game), so
#    they warn. Several committed configs carry these, and failing them would
#    block work without protecting a result.
_COMMON_PARAMS = frozenset({"sim_cfg", "reward_cfg", "opponent_bot", "learner_team"})
_VARIANT_PARAMS: dict[str, frozenset[str]] = {
    # cap_duel is fully self-contained; it takes only its own mini-game config
    # and the self-play schedule.
    "cap_duel": frozenset(
        {"mini_game", "mini_game_cfg", "self_play", "self_play_schedule", "snapshot_league"}
    ),
    # n_agents is honored in the sense that this env is inherently six-agent,
    # so a caller passing 6 is agreeing with it rather than being overridden.
    "self_play": frozenset(
        {"sim_cfg", "reward_cfg", "self_play", "self_play_schedule", "snapshot_league", "n_agents"}
    ),
    "phase11": frozenset(
        {"sim_cfg", "reward_cfg", "fog_mode", "visible_radius", "map_randomization",
         "self_play_schedule", "snapshot_league", "n_agents", "actor_obs"}
    ),
    "aim_only": frozenset({"mini_game", "mini_game_cfg"}),
    "combat_1v1": frozenset({"mini_game", "mini_game_cfg"}),
    "multi_enemy": _COMMON_PARAMS | frozenset({"actor_obs"}),
    "flat": _COMMON_PARAMS,
}

# Dropping one of these changes the simulation or the observation shape.
_SEMANTIC_PARAMS = frozenset({"fog_mode", "visible_radius", "map_randomization", "actor_obs"})

# Defaults used to decide whether the caller actually asked for something. A
# value equal to its default carries no intent, so it is never a conflict.
_PARAM_DEFAULTS: dict[str, Any] = {
    "opponent_bot": "basic",
    "learner_team": "A",
    "actor_obs": "flat",
    "fog_mode": "none",
    "visible_radius": 0.65,
    "map_randomization": {},
    "mini_game": None,
    "mini_game_cfg": {},
    "self_play": False,
    "self_play_schedule": None,
    "snapshot_league": None,
    "n_agents": 3,
    "reward_cfg": {},
}


def _check_ignored_params(variant: str, supplied: dict[str, Any]) -> None:
    """Raise on semantic drops, warn on merely-misleading ones."""
    honored = _VARIANT_PARAMS[variant]
    ignored = sorted(
        name
        for name, value in supplied.items()
        if name not in honored
        and name in _PARAM_DEFAULTS
        and value != _PARAM_DEFAULTS[name]
    )
    if not ignored:
        return
    semantic = [name for name in ignored if name in _SEMANTIC_PARAMS]
    if semantic:
        raise ValueError(
            f"env variant {variant!r} cannot honor {semantic}, and dropping "
            "them changes what is simulated or what the policy observes. "
            "Running anyway would produce a result whose config does not "
            "describe it. Remove them, or pick a variant that supports them "
            f"(this variant honors: {sorted(honored)})."
        )
    cosmetic = [name for name in ignored if name not in _SEMANTIC_PARAMS]
    _LOGGER.warning(
        "env variant %r ignores %s from the config; the run is unaffected but "
        "the config does not describe it accurately",
        variant,
        cosmetic,
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
    snapshot_league: dict[str, Any] | None = None,
    n_agents: int = 3,
    opponent_snapshot_stochastic: bool = False,
) -> gym.Env:
    reward = dict(reward_cfg or {})
    mini_cfg = dict(mini_game_cfg or {})
    fog = None if fog_mode in ("", "none", None) else str(fog_mode)
    map_cfg = dict(map_randomization or {})
    league_cfg = dict(snapshot_league or {})

    supplied = {
        "opponent_bot": opponent_bot,
        "learner_team": learner_team,
        "reward_cfg": reward,
        "actor_obs": actor_obs,
        "fog_mode": fog_mode,
        "visible_radius": visible_radius,
        "map_randomization": map_cfg,
        "mini_game": mini_game,
        "mini_game_cfg": mini_cfg,
        "self_play": self_play,
        "self_play_schedule": self_play_schedule,
        "snapshot_league": snapshot_league,
        "n_agents": n_agents,
    }

    if mini_game == "cap_duel":
        _check_ignored_params("cap_duel", supplied)
        from envs.phase4_cap_duel_mappo import Phase4CapDuelMappoEnv

        return Phase4CapDuelMappoEnv(
            **mini_cfg,
            self_play_schedule=(self_play_schedule if self_play else None),
            snapshot_league=(snapshot_league if self_play else None),
        )

    if self_play:
        if mini_game not in (None, ""):
            raise ValueError("current self-play can only be combined with cap_duel mini_game")
        # Checked before phase11 so `self_play + n_agents: 6` resolves here, as
        # it always has. The rejection makes the precedence visible instead of
        # letting the losing branch's parameters vanish.
        _check_ignored_params("self_play", supplied)
        from envs.phase4_selfplay_mappo import Phase4CurrentSelfplayMappoEnv

        return Phase4CurrentSelfplayMappoEnv(
            dict(sim_cfg),
            reward_cfg=reward,
            self_play_schedule=self_play_schedule,
            snapshot_league=snapshot_league,
        )
    if int(n_agents) == 6:
        _check_ignored_params("phase11", supplied)
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
        _check_ignored_params("aim_only", supplied)
        from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv

        return Phase4AimOnlyMappoEnv(**mini_cfg)
    if mini_game == "combat_1v1":
        _check_ignored_params("combat_1v1", supplied)
        from envs.phase4_combat_1v1_mappo import Phase4Combat1v1MappoEnv

        return Phase4Combat1v1MappoEnv(**mini_cfg)
    if mini_game not in (None, ""):
        raise ValueError(f"unknown mappo mini_game {mini_game!r}")

    # 3-agent snapshot opponent: a frozen checkpoint drives the enemy team.
    # Seeds from snapshot_paths (or snapshot_league.latest); episode-level
    # snapshot mixing is handled by opponent_bot_mix "snapshot:<path>"
    # entries through the runtime setters, not here.
    opponent_policy = None
    if str(opponent_bot) == "snapshot":
        from xushi2.snapshot_policy import SnapshotPolicy

        pool = tuple(snapshot_paths) or tuple(league_cfg.get("latest", ()))
        if not pool:
            raise ValueError(
                "opponent_bot 'snapshot' requires snapshot_paths or "
                "snapshot_league.latest"
            )
        opponent_policy = SnapshotPolicy(
            pool[0], stochastic=bool(opponent_snapshot_stochastic)
        )

    if actor_obs == "multi_enemy_entity_grid":
        _check_ignored_params("multi_enemy", supplied)
        from envs.phase4_multi_enemy_mappo import Phase4MultiEnemyMappoEnv

        return Phase4MultiEnemyMappoEnv(
            dict(sim_cfg),
            opponent_bot=str(opponent_bot),
            learner_team=str(learner_team),
            reward_cfg=reward,
            opponent_policy=opponent_policy,
            opponent_snapshot_stochastic=bool(opponent_snapshot_stochastic),
        )
    if actor_obs != "flat":
        raise ValueError(f"unknown mappo actor_obs {actor_obs!r}")

    _check_ignored_params("flat", supplied)
    from envs.phase4_mappo import Phase4MappoEnv

    return Phase4MappoEnv(
        dict(sim_cfg),
        opponent_bot=str(opponent_bot),
        learner_team=str(learner_team),
        reward_cfg=reward,
        opponent_policy=opponent_policy,
        opponent_snapshot_stochastic=bool(opponent_snapshot_stochastic),
    )


def mappo_env_fn_from_config(env_cfg: dict[str, Any]) -> Callable[[], gym.Env]:
    cfg = dict(env_cfg)
    sim_cfg = dict(cfg.get("sim", {}))
    reward_cfg = dict(cfg.get("reward", {}))
    self_play_cfg = dict(cfg.get("self_play", {}))
    # `snapshot_paths` and `target_slot` were accepted here and by
    # make_mappo_match_env, and referenced by neither. Reject them rather than
    # keeping a config surface that does nothing.
    # Only a key that actually asks for something is an error; several configs
    # carry an explicit `target_slot: false`, which requests nothing.
    features = dict(cfg.get("features", {}))
    for dead_key, replacement in (
        ("snapshot_paths", "snapshot_league"),
        ("target_slot", "ppo.target_selection_dim"),
    ):
        if cfg.get(dead_key) or features.get(dead_key):
            raise ValueError(
                f"env.{dead_key} is not implemented and was silently ignored; "
                f"use {replacement} instead"
            )
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
            dict(cfg.get("self_play_schedule", {})) if "self_play_schedule" in cfg else None
        ),
        snapshot_league=(
            dict(cfg.get("snapshot_league", {})) if "snapshot_league" in cfg else None
        ),
        n_agents=int(cfg.get("n_agents", cfg.get("team_size", 3))),
        opponent_snapshot_stochastic=bool(
            cfg.get("opponent_snapshot_stochastic", False)
        ),
    )
