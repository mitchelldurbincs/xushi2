"""Shared scripted-bot registry for Python entry points.

Keep this list in sync with bot names exposed by the C++ layer.
"""

from __future__ import annotations

VALID_SCRIPTED_BOTS: tuple[str, ...] = (
    "walk_to_objective",
    "hold_and_shoot",
    "basic",
    "weak_basic",
    "weak_basic_v2",
    "noop",
)

VALID_SCRIPTED_BOT_SET: frozenset[str] = frozenset(VALID_SCRIPTED_BOTS)

__all__ = ["VALID_SCRIPTED_BOTS", "VALID_SCRIPTED_BOT_SET"]
