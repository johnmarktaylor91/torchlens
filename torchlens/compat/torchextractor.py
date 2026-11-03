"""Small torchextractor-compatible facade backed by TorchLens."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn


class Extractor:
    """Callable feature extractor with torchextractor-shaped ergonomics.

    Parameters
    ----------
    model:
        PyTorch model to execute.
    layers:
        Layer names accepted by ``torchlens.extract``.

    Examples
    --------
    >>> extractor = Extractor(model, {"stem": "0", "head": "2"})
    >>> features = extractor(x)
    >>> sorted((name, tuple(out.shape)) for name, out in features.items())
    [('head', (1, 2, 8, 8)), ('stem', (1, 4, 8, 8))]
    """

    def __init__(self, model: nn.Module, layers: Iterable[str] | Mapping[str, str]) -> None:
        """Initialize the facade.

        Parameters
        ----------
        model:
            PyTorch model to execute.
        layers:
            Layer names accepted by ``torchlens.extract``.
        """

        self.model = model
        self.layers = layers

    def __call__(self, x: Any) -> dict[str, torch.Tensor]:
        """Extract features for one model input.

        Parameters
        ----------
        x:
            Input passed to the wrapped model.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer names to outs.
        """

        from torchlens import extract

        return extract(self.model, x, self.layers)

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Extract features for one model input.

        Parameters
        ----------
        inputs:
            Input passed to the wrapped model.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from layer names to outs.
        """

        return self(inputs)


__all__ = ["Extractor"]
