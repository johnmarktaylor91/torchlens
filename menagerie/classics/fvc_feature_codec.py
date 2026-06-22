"""FVC feature-space video codec compact random-init reconstruction.

Paper: FVC: A New Framework Towards Deep Video Compression in Feature Space
(Hu, Lu, Xu, CVPR 2021).

The load-bearing mechanism is that motion estimation, motion compression,
motion compensation, and residual compression operate on learned features, with
deformable compensation and non-local fusion from reference features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FeatureCodec(nn.Module):
    """Small autoencoder used for motion and residual feature compression."""

    def __init__(self, channels: int) -> None:
        """Initialize analysis and synthesis transforms."""

        super().__init__()
        self.analysis = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU()
        )
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compress and reconstruct a feature tensor."""

        return self.synthesis(torch.round(self.analysis(x) * 8.0) / 8.0)


class FVCFeatureCodec(nn.Module):
    """Compact feature-space video compression network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize feature extraction, codecs, compensation, fusion, and decoder."""

        super().__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.motion = nn.Conv2d(channels * 2, 2, 3, padding=1)
        self.motion_codec = FeatureCodec(2)
        self.res_codec = FeatureCodec(channels)
        self.fuse_q = nn.Conv2d(channels, channels, 1)
        self.fuse_k = nn.Conv2d(channels, channels, 1)
        self.decode = nn.Conv2d(channels, 3, 3, padding=1)

    def _warp(self, feat: Tensor, offset: Tensor) -> Tensor:
        """Apply feature-space deformable compensation with a dense offset field."""

        bsz, _, height, width = feat.shape
        ys = torch.linspace(-1.0, 1.0, height, device=feat.device)
        xs = torch.linspace(-1.0, 1.0, width, device=feat.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)
        return F.grid_sample(
            feat, grid + 0.2 * torch.tanh(offset).permute(0, 2, 3, 1), align_corners=False
        )

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode current/reference frames and reconstruct the current frame."""

        current, reference = inputs
        cur_feat = self.extract(current)
        ref_feat = self.extract(reference)
        motion = self.motion_codec(self.motion(torch.cat([cur_feat, ref_feat], dim=1)))
        pred_feat = self._warp(ref_feat, motion)
        residual_hat = self.res_codec(cur_feat - pred_feat)
        query = self.fuse_q(cur_feat).flatten(2).transpose(1, 2)
        key = self.fuse_k(ref_feat).flatten(2)
        attn = torch.softmax(torch.matmul(query, key) / (cur_feat.shape[1] ** 0.5), dim=-1)
        ref_context = torch.matmul(attn, ref_feat.flatten(2).transpose(1, 2)).transpose(1, 2)
        ref_context = ref_context.view_as(cur_feat)
        recon_feat = pred_feat + residual_hat + 0.25 * ref_context
        return self.decode(recon_feat), motion, residual_hat


def build() -> nn.Module:
    """Build a compact random-init FVC feature codec."""

    return FVCFeatureCodec().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return current and reference RGB frames."""

    return (torch.randn(1, 3, 24, 24), torch.randn(1, 3, 24, 24))


MENAGERIE_ENTRIES = [
    ("fvc_feature_codec", "build", "example_input", "2021", "DC"),
]
