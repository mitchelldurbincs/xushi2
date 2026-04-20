"""xushi2 — Python interface to the C++ simulation.

The heavy lifting happens in the C++ extension `xushi2_cpp`, built by CMake
and placed alongside this file by src/python_bindings/CMakeLists.txt.

This module provides:
- a thin re-export of the C++ types
- a Gymnasium-style environment wrapper (see `env.py`, TBD)
- utilities for batched / vectorized environment stepping (see `vec_env.py`, TBD)
"""

from __future__ import annotations

__version__ = "0.0.1"

try:
    from . import xushi2_cpp as _cpp
except ImportError as exc:  # pragma: no cover - triggered before build
    raise ImportError(
        "xushi2_cpp extension not found. Build the C++ side with CMake first:\n"
        "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
        "    cmake --build build -j\n"
        "(from the repository root, not the python/ subdir)."
    ) from exc

Team = _cpp.Team
Role = _cpp.Role
HeroKind = _cpp.HeroKind
Action = _cpp.Action
MatchConfig = _cpp.MatchConfig
Sim = _cpp.Sim

TICK_HZ = _cpp.TICK_HZ
AGENTS_PER_MATCH = _cpp.AGENTS_PER_MATCH
TEAM_SIZE = _cpp.TEAM_SIZE

__all__ = [
    "Team",
    "Role",
    "HeroKind",
    "Action",
    "MatchConfig",
    "Sim",
    "TICK_HZ",
    "AGENTS_PER_MATCH",
    "TEAM_SIZE",
]
