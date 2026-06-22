"""DARTS ImageNet derived CNN, 2019.

Paper: DARTS: Differentiable Architecture Search (Liu, Simonyan, Yang; ICLR 2019).

Compact random-init reconstruction of the ImageNet evaluation form: the same
published DARTS_V2 normal/reduction genotype as the CIFAR derived network, but
with an ImageNet-style two-stage downsampling stem before the stacked cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from menagerie.classics.darts_cifar_derived_network import DARTSCell, DARTS_V2


class DARTSImageNetNetwork(nn.Module):
    """Compact ImageNet-flavored DARTS classifier."""

    def __init__(self, channels: int = 4, classes: int = 1000, layers: int = 3) -> None:
        """Create the ImageNet DARTS network.

        Parameters
        ----------
        channels
            Initial cell channel count.
        classes
            Number of output classes.
        layers
            Number of DARTS cells.
        """
        super().__init__()
        stem0_ch = channels
        stem1_ch = 3 * channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, stem0_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem0_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(stem0_ch, stem0_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem0_ch),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(stem0_ch, stem1_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem1_ch),
        )
        c_prev_prev = stem0_ch
        c_prev = stem1_ch
        c_curr = channels
        reduction_prev = True
        self.cells = nn.ModuleList()
        for i in range(layers):
            reduction = i in {layers // 3, 2 * layers // 3}
            if reduction:
                c_curr *= 2
            cell = DARTSCell(DARTS_V2, c_prev_prev, c_prev, c_curr, reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            c_prev_prev, c_prev = c_prev, len(DARTS_V2[1]) * c_curr
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an ImageNet-like image.

        Parameters
        ----------
        x
            Input image batch.

        Returns
        -------
        Tensor
            Class logits.
        """
        s0 = self.stem0(x)
        s1 = self.stem1(s0)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        return self.classifier(torch.flatten(self.pool(s1), 1))


def build() -> nn.Module:
    """Build the compact DARTS ImageNet derived network.

    Returns
    -------
    nn.Module
        Random-init DARTS ImageNet network.
    """
    return DARTSImageNetNetwork().eval()


def example_input() -> Tensor:
    """Return an ImageNet-like input.

    Returns
    -------
    Tensor
        Example image batch.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("DARTS ImageNet derived network", "build", "example_input", "2019", "E7"),
]
