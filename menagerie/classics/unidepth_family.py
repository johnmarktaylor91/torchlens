"""UniDepth compact metric monocular depth family.

Paper: Piccinelli et al., 2024, "UniDepth: Universal Monocular Metric Depth
Estimation"; UniDepthV2, 2025.

UniDepth predicts metric 3D from one image by estimating a dense camera
representation that conditions depth features and by using a pseudo-spherical
output representation.  The V2 variants add edge-guided local scale/shift
refinement and confidence output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class UniDepthCompact(nn.Module):
    """Compact UniDepth/UniDepthV2 metric depth estimator."""

    def __init__(self, width: int = 48, vit: bool = True, v2: bool = False) -> None:
        """Initialize image encoder, camera prompt, and spherical depth heads."""

        super().__init__()
        self.v2 = v2
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.patch = nn.Conv2d(width, width, 1 if vit else 3, padding=0 if vit else 1)
        self.camera = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 6))
        self.cam_proj = nn.Linear(6, width)
        layer = nn.TransformerEncoderLayer(
            width, 4, dim_feedforward=width * 2, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.spherical = nn.Conv2d(width, 3, 1)
        self.refine = nn.Conv2d(width + 1, width, 3, padding=1)
        self.confidence = nn.Conv2d(width, 1, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict pseudo-spherical points, metric depth, and confidence."""

        feat = self.patch(self.stem(image))
        camera = self.camera(feat)
        tokens = feat.flatten(2).transpose(1, 2) + self.cam_proj(camera).unsqueeze(1)
        tokens = self.encoder(tokens)
        feat = tokens.transpose(1, 2).reshape_as(feat)
        spherical = self.spherical(feat)
        rays = F.normalize(spherical, dim=1)
        depth = F.softplus(spherical[:, :1]) + 0.1
        if self.v2:
            edge = image.mean(dim=1, keepdim=True)
            edge = F.interpolate(edge, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            depth = depth + 0.1 * torch.tanh(
                self.confidence(F.gelu(self.refine(torch.cat([feat, edge], dim=1))))
            )
        points = rays * depth
        confidence = torch.sigmoid(self.confidence(feat))
        return points, depth, confidence


def build_v1_cnvnxtl() -> nn.Module:
    """Build compact UniDepth V1 ConvNeXt-L style model."""

    return UniDepthCompact(width=48, vit=False, v2=False).eval()


def build_v1_vitl14() -> nn.Module:
    """Build compact UniDepth V1 ViT-L/14 style model."""

    return UniDepthCompact(width=56, vit=True, v2=False).eval()


def build_v2_vitb14() -> nn.Module:
    """Build compact UniDepth V2 ViT-B/14 style model."""

    return UniDepthCompact(width=40, vit=True, v2=True).eval()


def build_v2_vitl14() -> nn.Module:
    """Build compact UniDepth V2 ViT-L/14 style model."""

    return UniDepthCompact(width=56, vit=True, v2=True).eval()


def build_v2_vits14() -> nn.Module:
    """Build compact UniDepth V2 ViT-S/14 style model."""

    return UniDepthCompact(width=32, vit=True, v2=True).eval()


def build_v2old_vitl14() -> nn.Module:
    """Build compact UniDepth V2-old/V1-compatible ViT-L/14 model."""

    return UniDepthCompact(width=56, vit=True, v2=False).eval()


def example_input() -> Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 48, 64)


MENAGERIE_ENTRIES = [
    ("unidepth:v1_cnvnxtl", "build_v1_cnvnxtl", "example_input", "2024", "DEPTH"),
    ("unidepth:v1_vitl14", "build_v1_vitl14", "example_input", "2024", "DEPTH"),
    ("unidepth:v2_vitb14", "build_v2_vitb14", "example_input", "2025", "DEPTH"),
    ("unidepth:v2_vitl14", "build_v2_vitl14", "example_input", "2025", "DEPTH"),
    ("unidepth:v2_vits14", "build_v2_vits14", "example_input", "2025", "DEPTH"),
    ("unidepth:v2old_vitl14", "build_v2old_vitl14", "example_input", "2025", "DEPTH"),
]
