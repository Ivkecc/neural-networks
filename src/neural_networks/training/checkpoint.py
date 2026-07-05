from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
) -> None:
    model.load_state_dict(torch.load(path))