"""Deep-hashing image-retrieval family: AlexNet backbone + k-bit hash head.

Source: https://github.com/swuxyj/DeepHash-pytorch  (DeepHash-pytorch reference impl)
These four methods all share the SAME forward architecture -- an AlexNet convolutional
backbone whose final classifier layer is replaced by a fully-connected **hash layer**
producing a continuous k-bit code (later binarized by sign()).  They differ ONLY in the
TRAINING LOSS, not in the forward graph:

  - DPSH  -- Deep Pairwise-Supervised Hashing (Li et al., IJCAI 2016,
             https://arxiv.org/abs/1511.03855): pairwise maximum-likelihood loss on
             inner products of hash codes.
  - DSH   -- Deep Supervised Hashing (Liu et al., CVPR 2016): contrastive margin loss
             pulling same-class codes together / pushing different-class apart, with a
             quantization regularizer toward {-1,+1}.
  - DTSH  -- Deep Triplet-Supervised Hashing (Wang et al., ACCV 2016): triplet ranking
             likelihood loss over (anchor, positive, negative) hash codes.
  - DSDH  -- Deep Supervised Discrete Hashing (Li et al., NeurIPS 2017,
             https://arxiv.org/abs/1705.10999): jointly learns the binary codes and a
             linear classifier with a DISCRETE (sign-constrained) cyclic-coordinate-descent
             objective; the network forward is still backbone + hash-fc.

This faithful compact reimplementation reproduces the shared forward primitive: an
AlexNet-lite feature extractor + a linear hash head emitting a `hash_bit`-dimensional code
with a tanh squashing toward the hypercube.  The four registered entries point at the same
builder; the loss difference is documented but is training-only and not part of the forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNetHash(nn.Module):
    """AlexNet-lite backbone + linear hash head producing a k-bit continuous code."""

    def __init__(self, hash_bit: int = 48, in_ch: int = 3) -> None:
        super().__init__()
        # AlexNet feature extractor (5 conv stages with ReLU + maxpool).
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # AlexNet fc1/fc2 classifier head, with the final layer -> hash code.
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        # The DISTINCTIVE deep-hashing head: a linear layer to a k-bit code.
        self.hash_layer = nn.Linear(4096, hash_bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        code = self.hash_layer(x)
        # tanh squashes toward the {-1,+1} hypercube (relaxed binary code).
        return torch.tanh(code)


def _build(hash_bit: int = 48) -> nn.Module:
    return AlexNetHash(hash_bit=hash_bit).eval()


def example_input() -> torch.Tensor:
    """Example RGB image ``(1, 3, 224, 224)`` for the AlexNet hashing backbone."""
    return torch.randn(1, 3, 224, 224)


def build_dpsh() -> nn.Module:
    """DPSH: AlexNet + 48-bit hash head (pairwise-likelihood loss, training-only)."""
    return _build(48)


def build_dsh() -> nn.Module:
    """DSH: AlexNet + 48-bit hash head (contrastive margin loss, training-only)."""
    return _build(48)


def build_dtsh() -> nn.Module:
    """DTSH: AlexNet + 48-bit hash head (triplet ranking loss, training-only)."""
    return _build(48)


def build_dsdh() -> nn.Module:
    """DSDH: AlexNet + 48-bit hash head (discrete supervised hashing, training-only)."""
    return _build(48)


MENAGERIE_ENTRIES = [
    (
        "DPSH (AlexNet + pairwise-likelihood deep-hash head)",
        "build_dpsh",
        "example_input",
        "2016",
        "DC",
    ),
    (
        "DSH (AlexNet + contrastive-margin deep-hash head)",
        "build_dsh",
        "example_input",
        "2016",
        "DC",
    ),
    (
        "DTSH (AlexNet + triplet-supervised deep-hash head)",
        "build_dtsh",
        "example_input",
        "2016",
        "DC",
    ),
    (
        "DSDH (AlexNet + discrete supervised deep-hash head)",
        "build_dsdh",
        "example_input",
        "2017",
        "DC",
    ),
]
