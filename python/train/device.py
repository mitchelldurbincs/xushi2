"""Device resolution shared by the MAPPO and recurrent-PPO trainers.

``resolve_device`` maps the string surfaced in configs (``"cpu"``,
``"cuda"``, ``"cuda:0"``, or the sentinel ``"auto"``) to a concrete
``torch.device``. ``"auto"`` picks CUDA when available and falls back to
CPU so the same config runs on either box without edits.
"""

from __future__ import annotations

import torch


def resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
