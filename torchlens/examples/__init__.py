"""Example-loading namespace for small TorchLens artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class _TinyLinear(nn.Module):
    """Minimal model used for vendored example logs."""

    def __init__(self) -> None:
        """Initialize the tiny example model."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny example model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x))


def load(name: str, *, cache_dir: str | Path | None = None) -> Any:
    """Load a minimal vendored TorchLens example artifact by name.

    Parameters
    ----------
    name:
        Example name. ``"name"``, ``"tiny_linear"``, and ``"linear"`` are
        accepted aliases for the built-in smoke artifact.
    cache_dir:
        Reserved for future Hub-backed example downloads. Currently unused.

    Returns
    -------
    Any
        A ``ModelLog`` for the requested example.

    Raises
    ------
    KeyError
        If ``name`` is unknown.
    """

    del cache_dir
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in {"name", "tiny_linear", "linear"}:
        raise KeyError(f"Unknown TorchLens example {name!r}.")

    from torchlens import log_forward_pass

    torch.manual_seed(0)
    model = _TinyLinear()
    x = torch.randn(2, 3)
    return log_forward_pass(model, x)


__all__ = ["load"]
