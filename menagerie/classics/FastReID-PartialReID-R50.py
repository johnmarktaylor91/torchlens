"""FastReID PartialReID R50 compact reconstruction.

Deep Spatial Reconstruction and later visibility-aware partial person ReID systems pair
a ResNet-style person backbone with stripe/part features and visibility-aware matching.
This compact model keeps the R50-style bottleneck hierarchy, horizontal part pooling,
visibility logits, and ID embedding head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Compact ResNet bottleneck block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize bottleneck layers.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        stride:
            Spatial stride.
        """

        super().__init__()
        mid = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.down = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a bottleneck residual block.

        Parameters
        ----------
        x:
            Image feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return F.relu(y + self.down(x))


class CompactPartialReIDR50(nn.Module):
    """Compact FastReID-style partial person ReID model."""

    def __init__(self, embedding_dim: int = 64, parts: int = 4, identities: int = 32) -> None:
        """Initialize compact ReID model.

        Parameters
        ----------
        embedding_dim:
            Output embedding dimension.
        parts:
            Horizontal body-part stripes.
        identities:
            Compact training identity count.
        """

        super().__init__()
        self.parts = parts
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(Bottleneck(16, 32), Bottleneck(32, 32))
        self.layer2 = nn.Sequential(Bottleneck(32, 64, stride=2), Bottleneck(64, 64))
        self.layer3 = nn.Sequential(Bottleneck(64, 96, stride=2), Bottleneck(96, 96))
        self.global_head = nn.Linear(96, embedding_dim)
        self.part_head = nn.Linear(96 * parts, embedding_dim)
        self.visibility = nn.Linear(96, 1)
        self.classifier = nn.Linear(embedding_dim, identities)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract global/part ReID features and visibility logits.

        Parameters
        ----------
        image:
            Person image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Embedding, identity logits, and per-part visibility logits.
        """

        feat = self.layer3(self.layer2(self.layer1(self.stem(image))))
        global_feat = feat.mean(dim=(2, 3))
        stripes = torch.chunk(feat, self.parts, dim=2)
        part_feats = torch.stack([stripe.mean(dim=(2, 3)) for stripe in stripes], dim=1)
        visibility = self.visibility(part_feats).squeeze(-1)
        fused = self.global_head(global_feat) + self.part_head(part_feats.flatten(1))
        embedding = F.normalize(fused, dim=-1)
        return embedding, self.classifier(embedding), visibility


def build_FastReID_PartialReID_R50() -> nn.Module:
    """Build compact FastReID PartialReID R50.

    Returns
    -------
    nn.Module
        Random-init compact PartialReID model.
    """

    return CompactPartialReIDR50()


def example_input() -> torch.Tensor:
    """Create compact person image input.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(1, 3, 64, 32)``.
    """

    return torch.randn(1, 3, 64, 32)


build = build_FastReID_PartialReID_R50

MENAGERIE_ENTRIES = [
    ("FastReID-PartialReID-R50", "build_FastReID_PartialReID_R50", "example_input", "2018", "E5"),
]
