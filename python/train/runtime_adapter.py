"""Public runtime/environment adapter for script entrypoints.

Entrypoints should go through this module instead of wiring runtime resolution
and env factory checks ad hoc.
"""

from __future__ import annotations

from typing import Any, Callable

from train.runtime_specs import RuntimeSpec, resolve_runtime_spec


def resolve_runtime_env_factory(
    config: dict[str, Any],
    *,
    require_learner: str | None = None,
    context: str = "entrypoint",
) -> tuple[RuntimeSpec, Callable[[], object], int]:
    """Resolve runtime and validated env factory for script entrypoints."""
    runtime = resolve_runtime_spec(config)
    if require_learner is not None and runtime.learner.kind != require_learner:
        raise ValueError(
            f"{context} requires learner.kind={require_learner!r}, got {runtime.learner.kind!r}"
        )
    if runtime.env_fn is None:
        raise ValueError(
            f"{context} requires an environment runtime, got env={runtime.env.kind!r}"
        )
    return runtime, runtime.env_fn, int(runtime.seed_base)
