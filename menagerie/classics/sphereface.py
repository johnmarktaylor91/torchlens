"""SphereFace: Deep Hypersphere Embedding for Face Recognition.

Liu et al., CVPR 2017.
Paper: https://arxiv.org/abs/1704.08063
Source: https://github.com/wy1iu/sphereface

SphereFace introduces the angular softmax (A-Softmax) loss that constrains
face embeddings to lie on a hypersphere and optimises angular margins between
classes.  The network is a variant of the ResNet family, but trained with
A-Softmax instead of standard softmax.

Architecture (20-layer 'A' variant):
  - stem: 3x3 conv -> BN -> PReLU
  - 4 residual stages with blocks using 1x1 -> 3x3 -> 1x1 (bottleneck-like)
    with shortcut convolutions (NOT identity shortcut -- SphereFace uses a
    plain 3x3 residual conv shortcut throughout).
  - Depths: [1, 2, 4, 1] for the 20-layer version.
  - A global average pool then a fully-connected embedding layer (512-d).
  - The embedding is L2-normalised before the angular-margin linear layer.

Angular-margin linear (A-Softmax proxy):
  The final classification layer has weight W with L2-normalised columns.
  The logit for class k is: cos(theta_k) = x_norm . W_k_norm
  This gives the distinctive angular-margin geometry.  In this reimpl the
  head is a simple NormedLinear (normalises both x and W) producing cosine
  logits, which faithfully shows the geometry in the graph.

Compact config: 4 stages, channels [16,32,64,128], no_classes=100,
  embed_dim=128, input (1,3,64,64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual block (SphereFace-style: 3x3 conv shortcut)
# ---------------------------------------------------------------------------


class SphereBlock(nn.Module):
    """SphereFace residual block: two 3x3 convs + 3x3 shortcut + PReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu1 = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu2 = nn.PReLU(out_ch)

        # Shortcut always a conv (SphereFace architecture)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.prelu2(out + residual)


# ---------------------------------------------------------------------------
# NormedLinear (angular-margin head proxy)
# ---------------------------------------------------------------------------


class NormedLinear(nn.Module):
    """Linear layer with L2-normalised weights: computes cosine similarity logits.

    This is the key component that gives SphereFace its angular-margin geometry:
      logit_k = x_norm . W_k_norm = cos(theta_k)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm)


# ---------------------------------------------------------------------------
# SphereFace-20 network
# ---------------------------------------------------------------------------


class SphereFace20(nn.Module):
    """SphereFace 20-layer face recognition CNN with angular-margin head.

    Stages: stem + 4 residual stages (depths [1,2,4,1]) + GAP + FC + NormedLinear.
    """

    def __init__(
        self,
        in_chans: int = 3,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        depths: tuple[int, ...] = (1, 2, 4, 1),
        embed_dim: int = 128,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.PReLU(channels[0]),
        )

        # Residual stages (stride-2 first block, stride-1 subsequent)
        stages = []
        in_ch = channels[0]
        for i, (out_ch, depth) in enumerate(zip(channels, depths)):
            # First block strides down (except stage 0 which already has small input)
            stride = 2 if i > 0 else 1
            stage = [SphereBlock(in_ch, out_ch, stride=stride)]
            for _ in range(depth - 1):
                stage.append(SphereBlock(out_ch, out_ch, stride=1))
            stages.append(nn.Sequential(*stage))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], embed_dim)
        self.bn_fc = nn.BatchNorm1d(embed_dim)

        # Angular-margin head: NormedLinear (cosine logits)
        self.classifier = NormedLinear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x).flatten(1)  # (B, C)
        x = self.bn_fc(self.fc(x))  # (B, embed_dim)
        logits = self.classifier(x)  # (B, num_classes) -- cosine logits
        return logits


# ---------------------------------------------------------------------------
# Builder + example
# ---------------------------------------------------------------------------


def build_sphereface20a() -> nn.Module:
    """Build compact SphereFace-20A face recognition model."""
    return SphereFace20(
        in_chans=3,
        channels=(16, 32, 64, 128),
        depths=(1, 2, 4, 1),
        embed_dim=128,
        num_classes=100,
    )


def example_input_sphereface() -> torch.Tensor:
    """(1, 3, 64, 64) face image input."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SphereFace-20A (A-Softmax: angular-margin NormedLinear head, hypersphere face embedding)",
        "build_sphereface20a",
        "example_input_sphereface",
        "2017",
        "DC",
    ),
]
