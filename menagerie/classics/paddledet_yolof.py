"""YOLOF compact single-level detector with a dilated encoder.

Paper: "You Only Look One-level Feature" (Chen et al., 2021).

The faithful primitive is YOLOF's deliberately single-in/single-out detector:
one C5-like feature map is processed by stacked residual dilated convolutions
instead of an FPN, then separate classification and box towers predict dense
anchors from that one level.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DilatedResidual(nn.Module):
    """Residual block with a dilated convolution."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        channels:
            Feature channel count.
        dilation:
            Dilation rate.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated residual encoding.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Encoded feature tensor.
        """

        return x + self.act(self.conv2(self.act(self.conv1(x))))


class YOLOFCompact(nn.Module):
    """Compact YOLOF using only one feature level."""

    def __init__(self, channels: int = 32, num_classes: int = 5) -> None:
        """Initialize YOLOF.

        Parameters
        ----------
        channels:
            Internal feature width.
        num_classes:
            Number of dense class logits.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.encoder = nn.Sequential(
            DilatedResidual(channels, 2),
            DilatedResidual(channels, 4),
            DilatedResidual(channels, 6),
        )
        self.cls_tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 3, padding=1),
        )
        self.box_tower = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 4, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict dense classes and boxes from one encoded feature level.

        Parameters
        ----------
        x:
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class logits and positive box distances.
        """

        feat = self.encoder(self.backbone(x))
        return self.cls_tower(feat), torch.relu(self.box_tower(feat))


def build_paddledet_yolof() -> nn.Module:
    """Build compact random-init YOLOF.

    Returns
    -------
    nn.Module
        YOLOF compact model.
    """

    return YOLOFCompact().eval()


def example_input() -> torch.Tensor:
    """Create a small RGB image.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddledet_yolof", "build_paddledet_yolof", "example_input", "2021", "DC"),
]
