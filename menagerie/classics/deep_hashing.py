"""Deep Hashing -- GreedyHash and HashNet for Image Retrieval.

GreedyHash: Su et al., NeurIPS 2018.
  Paper: https://arxiv.org/abs/1806.05760
HashNet: Cao et al., ICCV 2017.
  Paper: https://arxiv.org/abs/1702.00758

Both networks share a compact AlexNet-like backbone trained on 32x32 images,
followed by a hashing layer that maps features to binary (or near-binary) codes.

GreedyHash uses a straight-through estimator: the backward pass flows through
the continuous hash code, but the forward pass applies sign(h) for binary codes.
HashNet uses a scaled tanh relaxation during training, annealing to sign at test.
This is a faithful compact random-init reimplementation of both.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HashBackbone(nn.Module):
    """Compact AlexNet-style backbone for 32x32 image input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


class GreedyHash(nn.Module):
    """GreedyHash: straight-through deep hashing (Su et al., NeurIPS 2018).

    Forward: sign(h) for binary codes.
    Backward: straight-through through the continuous h.
    """

    def __init__(self, bits: int = 48) -> None:
        super().__init__()
        self.backbone = HashBackbone()
        self.hash_fc = nn.Linear(256, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        h = self.hash_fc(feat)
        # Straight-through estimator: forward uses sign, grad flows through h.
        return h + (torch.sign(h) - h).detach()


class HashNet(nn.Module):
    """HashNet: tanh-relaxation deep hashing (Cao et al., ICCV 2017).

    Maps features to (-1, 1) range via scaled tanh; anneals to binary at test.
    """

    def __init__(self, bits: int = 48) -> None:
        super().__init__()
        self.backbone = HashBackbone()
        self.hash_fc = nn.Linear(256, bits)
        self.beta: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        h = self.hash_fc(feat)
        return torch.tanh(self.beta * h)


def build_greedyhash_alexnet_hash() -> nn.Module:
    """Build GreedyHash model with 48-bit codes."""
    return GreedyHash(bits=48)


def build_hashnet_alexnet_hash() -> nn.Module:
    """Build HashNet model with 48-bit codes."""
    return HashNet(bits=48)


def example_input() -> torch.Tensor:
    """Example 32x32 image tensor ``(1, 3, 32, 32)``."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "GreedyHash (straight-through deep hashing)",
        "build_greedyhash_alexnet_hash",
        "example_input",
        "2018",
        "DC",
    ),
    (
        "HashNet (tanh-relaxation deep hashing)",
        "build_hashnet_alexnet_hash",
        "example_input",
        "2017",
        "DC",
    ),
]
