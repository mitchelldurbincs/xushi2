"""Declared runtime-curriculum capabilities for MAPPO envs.

The vector wrappers push curriculum values (team-spirit ramp, reward alphas,
objective timing, respawn ticks) down to every wrapped env once per update.
Historically they discovered the setters with ``getattr(env, name, None)`` and
skipped envs that lacked them. That is silent by construction, and it has
already cost this project a set of runs: see the post-mortem comment in
``envs/phase4_multi_enemy_mappo.py``, where every multi-enemy run trained with
the objective-timing anneal, team-spirit ramp, and eval overrides dropped.

Silence is the problem, but so is unconditional strictness: the trainer pushes
team-spirit and the reward alphas on *every* update regardless of config, so an
env that legitimately cannot use a knob would fail every run.

The split this module draws:

- An env either implements a setter, or declares it unsupported via
  ``UNSUPPORTED_CURRICULUM_SETTERS``. Neither is an error by itself; failing to
  do either is, because it means nobody decided.
- Pushing a value to an env that declared the knob unsupported is a no-op.
- Requiring a knob that no env supports is a startup error, raised by
  ``require_curriculum_setters`` when the config actually turns that curriculum
  on. That is the case the 2026-06-10 runs needed and did not get.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CURRICULUM_SETTERS",
    "SupportsCurriculum",
    "UnsupportedCurriculumError",
    "declared_unsupported_setters",
    "require_curriculum_setters",
    "resolve_curriculum_setter",
    "supported_curriculum_setters",
]

# Canonical setter names. Adding a curriculum knob means adding it here; the
# vector wrappers dispatch generically off this tuple rather than carrying one
# hand-written method and one worker branch per knob.
CURRICULUM_SETTERS: tuple[str, ...] = (
    "set_team_spirit",
    "set_majority_on_point_alpha",
    "set_uncontested_on_point_alpha",
    "set_objective_timing_seconds",
    "set_respawn_ticks",
)

# Class attribute an env sets to declare, explicitly, that a knob does not
# apply to it. The value should be a mapping from setter name to the reason,
# so the decision is reviewable in the code rather than implied by absence.
_DECLARATION_ATTR = "UNSUPPORTED_CURRICULUM_SETTERS"


class UnsupportedCurriculumError(RuntimeError):
    """A configured curriculum has no env able to apply it."""


@runtime_checkable
class SupportsCurriculum(Protocol):
    """Structural type for an env that accepts every curriculum knob."""

    def set_team_spirit(self, value: float) -> None: ...

    def set_majority_on_point_alpha(self, value: float) -> None: ...

    def set_uncontested_on_point_alpha(self, value: float) -> None: ...

    def set_objective_timing_seconds(
        self, unlock_seconds: float, capture_seconds: float
    ) -> None: ...

    def set_respawn_ticks(self, respawn_ticks: int) -> None: ...


def declared_unsupported_setters(env: Any) -> dict[str, str]:
    """Setter names this env has explicitly declared it does not support."""
    declared = getattr(env, _DECLARATION_ATTR, None)
    if not declared:
        return {}
    if not isinstance(declared, dict):
        raise TypeError(
            f"{type(env).__name__}.{_DECLARATION_ATTR} must be a "
            f"{{setter_name: reason}} dict, got {type(declared).__name__}"
        )
    unknown = set(declared) - set(CURRICULUM_SETTERS)
    if unknown:
        raise ValueError(
            f"{type(env).__name__}.{_DECLARATION_ATTR} names unknown setters "
            f"{sorted(unknown)}; known: {list(CURRICULUM_SETTERS)}"
        )
    return dict(declared)


def resolve_curriculum_setter(env: Any, name: str) -> Callable[..., None] | None:
    """Return the bound setter, or ``None`` if the env opted out of it.

    Raises ``AttributeError`` when the env neither implements the setter nor
    declares it unsupported -- the case that used to be silently skipped.
    """
    if name not in CURRICULUM_SETTERS:
        raise ValueError(f"unknown curriculum setter {name!r}; known: {list(CURRICULUM_SETTERS)}")
    # An explicit declaration wins over a merely-present method. A subclass can
    # inherit a setter whose effect its own configuration makes inert -- e.g. a
    # scalar-reward env inheriting set_team_spirit, which only shapes the
    # per-agent path. Letting the inherited method win would report the knob as
    # supported and drop the curriculum anyway, which is the bug this module
    # exists to prevent.
    if name in declared_unsupported_setters(env):
        return None
    setter = getattr(env, name, None)
    if setter is not None:
        return setter
    raise AttributeError(
        f"{type(env).__name__} neither implements {name!r} nor declares it in "
        f"{_DECLARATION_ATTR}. Curriculum setters are pushed to every env once "
        "per update; an env that silently lacks one drops the curriculum "
        "without any signal. Implement the setter, or declare it unsupported "
        "with a reason."
    )


def supported_curriculum_setters(env: Any) -> frozenset[str]:
    """Setter names this env can actually apply.

    Also validates the env: every knob must be either implemented or declared,
    so calling this at construction turns an undeclared knob into a startup
    error rather than a silently-dropped curriculum thousands of updates later.
    """
    supported = {name for name in CURRICULUM_SETTERS if resolve_curriculum_setter(env, name)}
    return frozenset(supported)


def require_curriculum_setters(
    supported: frozenset[str],
    required: Iterable[str],
    *,
    context: str,
) -> None:
    """Raise if a configured curriculum cannot be applied by the envs."""
    missing = sorted(set(required) - supported)
    if not missing:
        return
    raise UnsupportedCurriculumError(
        f"{context} requires curriculum setter(s) {missing}, which the "
        f"environment does not support (supported: {sorted(supported)}). "
        "Either use an env that implements them or turn the curriculum off in "
        "the config -- running anyway would train against a curriculum that "
        "never actually applied."
    )
