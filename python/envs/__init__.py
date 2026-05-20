"""Public env interfaces and runtime factories.

Exports are lazy so importing the env package does not pull trainer-dependent
self-play helpers into unrelated training imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MemoryToyEnv": ("envs.memory_toy", "MemoryToyEnv"),
    "Phase3RangerEnv": ("envs.phase3_ranger", "Phase3RangerEnv"),
    "Phase4AimOnlyMappoEnv": ("envs.phase4_aim_only_mappo", "Phase4AimOnlyMappoEnv"),
    "Phase4CapDuelMappoEnv": ("envs.phase4_cap_duel_mappo", "Phase4CapDuelMappoEnv"),
    "Phase4Combat1v1MappoEnv": ("envs.phase4_combat_1v1_mappo", "Phase4Combat1v1MappoEnv"),
    "Phase4CurrentSelfplayMappoEnv": (
        "envs.phase4_selfplay_mappo",
        "Phase4CurrentSelfplayMappoEnv",
    ),
    "Phase4MappoEnv": ("envs.phase4_mappo", "Phase4MappoEnv"),
    "Phase5EntityMappoEnv": ("envs.phase5_entity_mappo", "Phase5EntityMappoEnv"),
    "Phase6GridMappoEnv": ("envs.phase6_grid_mappo", "Phase6GridMappoEnv"),
    "Phase7FogMappoEnv": ("envs.phase7_fog_mappo", "Phase7FogMappoEnv"),
    "Phase8RandomMapMappoEnv": ("envs.phase8_random_map_mappo", "Phase8RandomMapMappoEnv"),
    "Phase9SnapshotMappoEnv": ("envs.phase9_snapshot_mappo", "Phase9SnapshotMappoEnv"),
    "Phase10TargetSlotMappoEnv": (
        "envs.phase10_target_slot_mappo",
        "Phase10TargetSlotMappoEnv",
    ),
    "Phase11CurrentSelfplayMappoEnv": (
        "envs.phase11_current_selfplay_mappo",
        "Phase11CurrentSelfplayMappoEnv",
    ),
    "FlatRangerMappoMatchEnv": ("envs.phase4_mappo", "Phase4MappoEnv"),
    "CurrentSelfplayMappoMatchEnv": (
        "envs.phase4_selfplay_mappo",
        "Phase4CurrentSelfplayMappoEnv",
    ),
    "EntityObsMappoEnv": ("envs.phase5_entity_mappo", "Phase5EntityMappoEnv"),
    "EntityGridObsMappoEnv": ("envs.phase6_grid_mappo", "Phase6GridMappoEnv"),
    "FogMappoMatchEnv": ("envs.phase7_fog_mappo", "Phase7FogMappoEnv"),
    "RandomizedMapMappoEnv": ("envs.phase8_random_map_mappo", "Phase8RandomMapMappoEnv"),
    "SnapshotOpponentMappoEnv": ("envs.phase9_snapshot_mappo", "Phase9SnapshotMappoEnv"),
    "TargetSlotMappoEnv": ("envs.phase10_target_slot_mappo", "Phase10TargetSlotMappoEnv"),
    "SixAgentCurrentSelfplayMappoEnv": (
        "envs.phase11_current_selfplay_mappo",
        "Phase11CurrentSelfplayMappoEnv",
    ),
    "make_mappo_match_env": ("envs.runtime_factory", "make_mappo_match_env"),
    "make_memory_toy_env": ("envs.runtime_factory", "make_memory_toy_env"),
    "make_ranger_duel_env": ("envs.runtime_factory", "make_ranger_duel_env"),
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
