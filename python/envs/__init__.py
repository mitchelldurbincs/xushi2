"""Public env interfaces and phase entrypoints."""

from __future__ import annotations

from envs.memory_toy import MemoryToyEnv
from envs.phase3_ranger import Phase3RangerEnv
from envs.phase4_aim_only_mappo import Phase4AimOnlyMappoEnv
from envs.phase4_combat_1v1_mappo import Phase4Combat1v1MappoEnv
from envs.phase4_mappo import Phase4MappoEnv
from envs.phase5_entity_mappo import Phase5EntityMappoEnv
from envs.phase6_grid_mappo import Phase6GridMappoEnv
from envs.phase7_fog_mappo import Phase7FogMappoEnv
from envs.phase8_random_map_mappo import Phase8RandomMapMappoEnv
from envs.phase9_snapshot_mappo import Phase9SnapshotMappoEnv
from envs.phase10_target_slot_mappo import Phase10TargetSlotMappoEnv
from envs.phase11_current_selfplay_mappo import Phase11CurrentSelfplayMappoEnv

__all__ = [
    "MemoryToyEnv",
    "Phase3RangerEnv",
    "Phase4AimOnlyMappoEnv",
    "Phase4Combat1v1MappoEnv",
    "Phase4MappoEnv",
    "Phase5EntityMappoEnv",
    "Phase6GridMappoEnv",
    "Phase7FogMappoEnv",
    "Phase8RandomMapMappoEnv",
    "Phase9SnapshotMappoEnv",
    "Phase10TargetSlotMappoEnv",
    "Phase11CurrentSelfplayMappoEnv",
]
