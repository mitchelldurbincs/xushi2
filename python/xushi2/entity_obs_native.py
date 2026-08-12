"""Native entity-observation config helpers (ObservationEngine adapters).

Single translation point from env/checkpoint config dictionaries to the
native ``_cpp.ObsConfig``. Every consumer of the native entity-obs path
(the Phase-11 env, the Phase-4 multi-enemy wrapper, and SnapshotPolicy)
builds its ObsConfig here so the fog-semantics mapping lives in one place.

See src/sim/include/xushi2/sim/obs_config.h for the semantics enshrined in
C++ (the Phase-11 visibility rule: alive AND radius AND native LoS, with
team-shared unioning radius and LoS independently across teammates).
"""

from __future__ import annotations

import math
from typing import Any

from xushi2 import xushi2_cpp as _cpp

__all__ = [
    "make_obs_config",
    "phase11_obs_config",
    "phase4_multi_enemy_obs_config",
    "snapshot_obs_config",
]

_FOG_MODES = {
    "none": _cpp.FogMode.NoFog,
    "team_shared": _cpp.FogMode.TeamShared,
    "per_agent": _cpp.FogMode.PerAgent,
}


def make_obs_config(
    *,
    fog_mode: str,
    visible_radius: float | None,
    last_seen_enabled: bool,
    zero_hidden_token_markers: bool = False,
) -> Any:
    """Build a native ObsConfig. ``visible_radius=None`` means no radius term."""
    try:
        mode = _FOG_MODES[str(fog_mode)]
    except KeyError:
        raise ValueError(
            f"unknown fog_mode {fog_mode!r}; expected one of {sorted(_FOG_MODES)}"
        ) from None
    cfg = _cpp.ObsConfig()
    cfg.fog_mode = mode
    cfg.visible_radius = math.nan if visible_radius is None else float(visible_radius)
    cfg.last_seen_enabled = bool(last_seen_enabled)
    cfg.zero_hidden_token_markers = bool(zero_hidden_token_markers)
    return cfg


def phase11_obs_config(fog_mode: str, visible_radius: float) -> Any:
    """The Phase-11 training rule: alive & radius & LoS, with last-seen markers."""
    return make_obs_config(
        fog_mode=fog_mode,
        visible_radius=float(visible_radius),
        last_seen_enabled=True,
    )


def phase4_multi_enemy_obs_config() -> Any:
    """The Phase-4 multi-enemy ablation rule: alive & native LoS only.

    No radius term (the legacy wrapper neutralized it), no last-seen
    markers, and hidden enemy tokens fully zeroed
    (``zero_masked_enemy_tokens`` semantics).
    """
    return make_obs_config(
        fog_mode="per_agent",
        visible_radius=None,
        last_seen_enabled=False,
        zero_hidden_token_markers=True,
    )


def snapshot_obs_config(phase: int, env_cfg: dict[str, Any]) -> Any:
    """Per-checkpoint ObsConfig for a frozen snapshot opponent.

    The decision here is TRAINING-time semantics, per checkpoint:

    - phase >= 7 multi-enemy checkpoints trained in the Phase-11 env: fog
      mode and radius from the checkpoint's stored env config, last-seen
      markers ON. (The legacy serving path never provided last-seen — a
      known train/serve skew this path fixes exactly.)
    - phase < 7 multi-enemy checkpoints trained in the Phase-4 wrapper:
      alive & native LoS only, hidden tokens zeroed. (The legacy serving
      path applied team-shared fog with a 0.65 radius and no token zeroing
      to these checkpoints — none of which their training had; that skew
      is also fixed by construction here.)
    """
    if int(phase) >= 7:
        return phase11_obs_config(
            str(env_cfg.get("fog_mode", "team_shared")),
            float(env_cfg.get("visible_radius", 0.65)),
        )
    return phase4_multi_enemy_obs_config()
