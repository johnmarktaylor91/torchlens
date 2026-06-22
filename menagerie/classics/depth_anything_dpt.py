"""Compact DPT / Depth Anything dense prediction transformers.

Ranftl et al., "Vision Transformers for Dense Prediction", ICCV 2021, assemble
intermediate ViT tokens into image-like multi-scale features and progressively
fuse them with a convolutional decoder.  Depth Anything V1/V2 use DINOv2 ViT
encoders with DPT depth heads; V2 keeps the same architecture family while using
intermediate features rather than the last four layers and a stronger data recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPTCompact(nn.Module):
    """Small dense prediction transformer with optional hybrid convolutional stem."""

    def __init__(self, dim: int = 64, depth: int = 4, hybrid: bool = False) -> None:
        """Initialize ViT backbone and DPT fusion decoder.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Number of transformer layers.
        hybrid:
            Whether to use a convolutional pre-stem before patchification.
        """

        super().__init__()
        self.hybrid = hybrid
        self.stem = (
            nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 3, 3, padding=1))
            if hybrid
            else nn.Identity()
        )
        self.patch = nn.Conv2d(3, dim, 8, stride=8)
        self.pos = nn.Parameter(torch.randn(1, 17, dim) * 0.02)
        self.readout = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True, activation="gelu")
                for _ in range(depth)
            ]
        )
        taps = 4
        self.projects = nn.ModuleList([nn.Conv2d(dim, 32, 1) for _ in range(taps)])
        self.fuse = nn.ModuleList([nn.Conv2d(32, 32, 3, padding=1) for _ in range(taps)])
        self.head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict dense depth from an RGB image.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Dense depth logits at input resolution.
        """

        bsz = x.shape[0]
        x = self.stem(x)
        patches = self.patch(x)
        height, width = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.readout.expand(bsz, -1, -1), tokens], dim=1) + self.pos
        feats = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index >= len(self.blocks) - 4:
                grid = tokens[:, 1:].transpose(1, 2).reshape(bsz, -1, height, width)
                feats.append(grid)
        fused = torch.zeros(bsz, 32, height, width, device=x.device)
        for feat, project, fuse in zip(feats, self.projects, self.fuse, strict=False):
            fused = fuse(fused + project(feat))
        return self.head(
            F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
        )


def build_vits() -> nn.Module:
    """Build a compact ViT-small DPT model.

    Returns
    -------
    nn.Module
        Compact DPT model.
    """

    return DPTCompact(dim=48, depth=4, hybrid=False)


def build_vitb() -> nn.Module:
    """Build a compact ViT-base DPT model.

    Returns
    -------
    nn.Module
        Compact DPT model.
    """

    return DPTCompact(dim=64, depth=4, hybrid=False)


def build_vitl() -> nn.Module:
    """Build a compact ViT-large DPT model.

    Returns
    -------
    nn.Module
        Compact DPT model.
    """

    return DPTCompact(dim=72, depth=4, hybrid=False)


def build_vitg() -> nn.Module:
    """Build a compact ViT-giant DPT model.

    Returns
    -------
    nn.Module
        Compact DPT model.
    """

    return DPTCompact(dim=80, depth=4, hybrid=False)


def build_hybrid() -> nn.Module:
    """Build compact DPT-Hybrid with convolutional stem.

    Returns
    -------
    nn.Module
        Compact DPT-Hybrid model.
    """

    return DPTCompact(dim=64, depth=4, hybrid=True)


def example_input() -> torch.Tensor:
    """Create a small RGB dense-prediction image.

    Returns
    -------
    torch.Tensor
        Example tensor ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "depth_anything_v1_vitb (DINOv2-DPT monocular depth)",
        "build_vitb",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v1_vitl (DINOv2-DPT monocular depth)",
        "build_vitl",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v1_vits (DINOv2-DPT monocular depth)",
        "build_vits",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v1:vitb (DINOv2-DPT monocular depth)",
        "build_vitb",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v1:vitl (DINOv2-DPT monocular depth)",
        "build_vitl",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v1:vits (DINOv2-DPT monocular depth)",
        "build_vits",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2_vitb (DINOv2-DPT monocular depth)",
        "build_vitb",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2_vitg (DINOv2-DPT monocular depth)",
        "build_vitg",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2_vitl (DINOv2-DPT monocular depth)",
        "build_vitl",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2_vits (DINOv2-DPT monocular depth)",
        "build_vits",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2:vitb (DINOv2-DPT monocular depth)",
        "build_vitb",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2:vitg (DINOv2-DPT monocular depth)",
        "build_vitg",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2:vitl (DINOv2-DPT monocular depth)",
        "build_vitl",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "depth_anything_v2:vits (DINOv2-DPT monocular depth)",
        "build_vits",
        "example_input",
        "2024",
        "DC",
    ),
    (
        "dpt_original:hybrid (Dense Prediction Transformer hybrid)",
        "build_hybrid",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "dpt_original:large (Dense Prediction Transformer large)",
        "build_vitl",
        "example_input",
        "2021",
        "DC",
    ),
]
