"""Thin wandb wrapper that no-ops when disabled, missing, or failing.

Training code calls :func:`make_logger` once, then :meth:`WandbLogger.log`
repeatedly, then :meth:`WandbLogger.finish` in a finally block. If wandb is
unavailable or disabled, a null logger is returned and training continues
unchanged.

Disable wandb via ``WANDB_MODE=disabled`` or ``wandb.enabled: false`` in
the training config. Override the default project/entity via
``WANDB_PROJECT`` / ``WANDB_ENTITY`` env vars or the matching config fields.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

_LOGGER = logging.getLogger(__name__)

DEFAULT_PROJECT = "xushi2"


class WandbLogger(Protocol):
    enabled: bool

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def finish(self) -> None: ...


class _NullLogger:
    enabled = False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        return

    def finish(self) -> None:
        return


class _ActiveLogger:
    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        import wandb

        try:
            wandb.log(dict(metrics), step=step)
        except Exception as exc:
            _LOGGER.warning("wandb.log failed (%s); continuing", exc)

    def finish(self) -> None:
        import wandb

        try:
            wandb.finish()
        except Exception as exc:
            _LOGGER.warning("wandb.finish failed (%s)", exc)


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def make_logger(
    *,
    config: dict,
    run_name: str,
    run_config: Mapping[str, Any],
    tags: Sequence[str] = (),
) -> WandbLogger:
    """Construct an active wandb logger or a no-op fallback.

    Falls back to a null logger when wandb is disabled via ``WANDB_MODE``
    or ``wandb.enabled: false``, when the ``wandb`` package is missing, or
    when ``wandb.init`` raises (e.g. no API key, network unreachable).
    """
    wandb_cfg: dict = {}
    if isinstance(config, dict):
        cfg_section = config.get("wandb")
        if isinstance(cfg_section, dict):
            wandb_cfg = cfg_section

    project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project") or DEFAULT_PROJECT
    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    group = wandb_cfg.get("group")

    if not wandb_cfg.get("enabled", True):
        _LOGGER.info("W&B startup: project=%r entity=%r logger=null", project, entity)
        return _NullLogger()
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        _LOGGER.info("W&B startup: project=%r entity=%r logger=null", project, entity)
        return _NullLogger()

    try:
        import wandb
    except ImportError:
        _LOGGER.info("W&B startup: project=%r entity=%r logger=null", project, entity)
        _LOGGER.warning("wandb not installed; metrics will not be logged")
        return _NullLogger()

    enriched_config = dict(run_config)
    commit = _git_commit_short()
    if commit:
        enriched_config.setdefault("git_commit", commit)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=list(tags) if tags else None,
            config=enriched_config,
            reinit="finish_previous",
        )
    except Exception as exc:
        _LOGGER.info("W&B startup: project=%r entity=%r logger=null", project, entity)
        _LOGGER.warning("wandb.init failed (%s); metrics will not be logged", exc)
        return _NullLogger()

    _LOGGER.info("W&B startup: project=%r entity=%r logger=active", project, entity)
    return _ActiveLogger(run)
