"""Pytest configuration for xushi2 Python tests.

Forces ``WANDB_MODE=disabled`` so test runs that exercise the trainers do
not attempt to contact the wandb cloud or block on credential prompts.
Override with ``WANDB_MODE=online`` in the env if you intentionally want
to log a test run.
"""

from __future__ import annotations

import os

os.environ.setdefault("WANDB_MODE", "disabled")
