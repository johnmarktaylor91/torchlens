"""DeepSORT AppearanceNet, 2017.

Paper: Simple Online and Realtime Tracking with a Deep Association Metric
(Wojke, Bewley, Paulus; ICIP 2017) plus Deep Cosine Metric Learning for
Person Re-Identification (Wojke, Bewley; WACV 2018).

Faithful compact random-init reconstruction of the dependency-gated appearance
descriptor: a pedestrian crop CNN produces a low-dimensional embedding, strips
the training classifier at inference time, and L2-normalizes the descriptor for
cosine nearest-neighbor association in DeepSORT.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    """Small residual unit used in the appearance CNN."""

    def __init__(self, channels: int) -> None:
        """Create a residual unit.

        Parameters
        ----------
        channels
            Input and output channel count.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual refinement.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Residual output.
        """
        return self.act(x + self.body(x))


class DownBlock(nn.Module):
    """Downsampling conv block with residual refinement."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Create a downsampling block.

        Parameters
        ----------
        in_ch
            Input channel count.
        out_ch
            Output channel count.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )
        self.res = ResidualUnit(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample and refine a feature map.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        return self.res(self.proj(x))


class DeepSORTAppearanceNet(nn.Module):
    """CNN descriptor head for DeepSORT cosine appearance matching."""

    def __init__(self, embedding_dim: int = 128, classes: int = 64) -> None:
        """Create a compact appearance descriptor network.

        Parameters
        ----------
        embedding_dim
            Descriptor dimension used for cosine matching.
        classes
            Training classifier output count retained for trace visibility.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            DownBlock(16, 32),
            DownBlock(32, 64),
            DownBlock(64, 96),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(96, embedding_dim)
        self.classifier_scale = nn.Parameter(torch.tensor(10.0))
        self.classifier_weight = nn.Parameter(torch.randn(classes, embedding_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized descriptor and cosine-softmax logits.

        Parameters
        ----------
        x
            Pedestrian crop batch ``(B, 3, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            L2-normalized embedding and cosine classifier logits.
        """
        feat = torch.flatten(self.pool(self.features(x)), 1)
        emb = F.normalize(self.embedding(feat), dim=-1)
        weight = F.normalize(self.classifier_weight, dim=-1)
        logits = self.classifier_scale * torch.matmul(emb, weight.t())
        return emb, logits


def build() -> nn.Module:
    """Build the compact DeepSORT AppearanceNet.

    Returns
    -------
    nn.Module
        Random-init appearance descriptor network.
    """
    return DeepSORTAppearanceNet().eval()


def example_input() -> Tensor:
    """Return a pedestrian crop input.

    Returns
    -------
    Tensor
        Example crop batch.
    """
    return torch.randn(1, 3, 64, 32)


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = [
    ("DeepSORT-AppearanceNet", "build", "example_input", "2017", "E7"),
]
