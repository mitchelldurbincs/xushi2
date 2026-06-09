"""Public env interfaces and runtime factories.

Exports are lazy so importing the env package does not pull trainer-dependent
self-play helpers into unrelated training imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Phase4AimOnlyMappoEnv": ("envs.phase4_aim_only_mappo", "Phase4AimOnlyMappoEnv"),
    "Phase4CapDuelMappoEnv": ("envs.phase4_cap_duel_mappo", "Phase4CapDuelMappoEnv"),
    "Phase4Combat1v1MappoEnv": ("envs.phase4_combat_1v1_mappo", "Phase4Combat1v1MappoEnv"),
    "Phase4CurrentSelfplayMappoEnv": (
        "envs.phase4_selfplay_mappo",
        "Phase4CurrentSelfplayMappoEnv",
    ),
    "Phase4MappoEnv": ("envs.phase4_mappo", "Phase4MappoEnv"),
    "Phase11CurrentSelfplayMappoEnv": (
        "envs.phase11_current_selfplay_mappo",
        "Phase11CurrentSelfplayMappoEnv",
    ),
    "FlatRangerMappoMatchEnv": ("envs.phase4_mappo", "Phase4MappoEnv"),
    "CurrentSelfplayMappoMatchEnv": (
        "envs.phase4_selfplay_mappo",
        "Phase4CurrentSelfplayMappoEnv",
    ),
    "SixAgentCurrentSelfplayMappoEnv": (
        "envs.phase11_current_selfplay_mappo",
        "Phase11CurrentSelfplayMappoEnv",
    ),
    "make_mappo_match_env": ("envs.runtime_factory", "make_mappo_match_env"),
    "mappo_env_fn_from_config": ("envs.runtime_factory", "mappo_env_fn_from_config"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
