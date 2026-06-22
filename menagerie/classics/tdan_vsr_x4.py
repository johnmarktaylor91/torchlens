"""TDAN video super-resolution compact random-init reconstruction.

Paper: TDAN: Temporally-Deformable Alignment Network for Video
Super-Resolution (Tian, Zhang, Fu, Xu, CVPR 2020).

TDAN aligns neighboring frames to a reference frame at the feature level by
predicting deformable offsets from reference/support features, then reconstructs
and pixel-shuffles to high resolution.  This compact version implements the
offset prediction and differentiable feature sampling with ``grid_sample``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DeformAlign(nn.Module):
    """Feature-level temporal deformable alignment block."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize offset and feature projection layers."""

        super().__init__()
        self.offset = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.mix = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, ref: Tensor, support: Tensor) -> Tensor:
        """Align a support-frame feature map to a reference feature map."""

        bsz, _, height, width = ref.shape
        delta = torch.tanh(self.offset(torch.cat([ref, support], dim=1))).permute(0, 2, 3, 1)
        ys = torch.linspace(-1.0, 1.0, height, device=ref.device)
        xs = torch.linspace(-1.0, 1.0, width, device=ref.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)
        aligned = F.grid_sample(support, grid + 0.15 * delta, align_corners=False)
        return self.mix(aligned)


class TDANVSR(nn.Module):
    """Compact TDAN x4 video super-resolution network."""

    def __init__(self, channels: int = 24, frames: int = 3) -> None:
        """Initialize feature extractor, alignment, reconstruction, and upsampler."""

        super().__init__()
        self.frames = frames
        self.feat = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.align = DeformAlign(channels)
        self.recon = nn.Sequential(
            nn.Conv2d(channels * frames, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, 3 * 16, 3, padding=1),
        )

    def forward(self, frames: Tensor) -> Tensor:
        """Super-resolve the middle reference frame from neighboring frames."""

        feats = [self.feat(frames[:, i]) for i in range(self.frames)]
        ref = feats[self.frames // 2]
        aligned = [
            self.align(ref, feat) if i != self.frames // 2 else ref for i, feat in enumerate(feats)
        ]
        lr_residual = self.recon(torch.cat(aligned, dim=1))
        return F.pixel_shuffle(lr_residual, 4)


def build() -> nn.Module:
    """Build a compact random-init TDAN x4 VSR model."""

    return TDANVSR().eval()


def example_input() -> Tensor:
    """Return three low-resolution RGB frames."""

    return torch.randn(1, 3, 3, 12, 12)


MENAGERIE_ENTRIES = [
    ("tdan_vsr_x4", "build", "example_input", "2020", "DC"),
]
