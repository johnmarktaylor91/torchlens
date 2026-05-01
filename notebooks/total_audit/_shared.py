"""Shared helpers for TorchLens Total Audit notebooks."""

from __future__ import annotations

import torch
from torch import nn


def tiny_model(seed: int = 0) -> nn.Module:
    """Return a deterministic three-layer MLP for audit notebooks.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    nn.Module
        Three-layer MLP with deterministic initial weights.
    """

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
