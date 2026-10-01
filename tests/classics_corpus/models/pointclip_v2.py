"""PointCLIP V2: Prompting CLIP and GPT for Powerful 3D Open-world Learning.

Zhu, Ma, Li, Liu, Tang & You (PKU / UCL), ICCV 2023, arXiv:2211.11682.
Source: https://github.com/yangyangyang127/PointCLIP_V2

PointCLIP V2 performs zero/few-shot 3D classification by:
  1. Projecting a point cloud onto multiple 2D DEPTH MAP views (orthographic
     projections along X, Y, Z axes, and diagonal views).
  2. Running a shared (CLIP-like) image encoder on each projected depth map.
  3. Aggregating view features (average pooling across views).
  4. Comparing aggregated features to text/class embeddings.

Distinctive primitives reproduced:
  - Point-cloud -> multi-view depth map projection (scatter + normalize)
  - Shared convolutional image encoder per view (lightweight CLIP-like)
  - View-level feature aggregation

We use a compact conv encoder (no actual CLIP weights -- random init), 3 views
(front, side, top), and a tiny image size (16x16) for tractable tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Multi-view depth map projection                                             #
# --------------------------------------------------------------------------- #


def project_to_depth_map(xyz: torch.Tensor, view_idx: int, img_size: int = 16) -> torch.Tensor:
    """Project point cloud to a 2D depth map via orthographic projection.

    view_idx: 0=front (XY plane, depth=Z),
              1=side  (YZ plane, depth=X),
              2=top   (XZ plane, depth=Y)

    Returns depth map (B, 1, H, W) in [0, 1].
    """
    B, N, _ = xyz.shape

    if view_idx == 0:  # front: project onto XY, depth = Z
        u, v, d = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    elif view_idx == 1:  # side: project onto YZ, depth = X
        u, v, d = xyz[:, :, 1], xyz[:, :, 2], xyz[:, :, 0]
    else:  # top: project onto XZ, depth = Y
        u, v, d = xyz[:, :, 0], xyz[:, :, 2], xyz[:, :, 1]

    # Normalize u, v to [0, img_size-1] pixel coords
    u_min = u.min(dim=1, keepdim=True)[0]
    u_max = u.max(dim=1, keepdim=True)[0]
    v_min = v.min(dim=1, keepdim=True)[0]
    v_max = v.max(dim=1, keepdim=True)[0]
    u_norm = (u - u_min) / (u_max - u_min + 1e-6) * (img_size - 1)
    v_norm = (v - v_min) / (v_max - v_min + 1e-6) * (img_size - 1)

    # Normalize depth to [0, 1]
    d_min = d.min(dim=1, keepdim=True)[0]
    d_max = d.max(dim=1, keepdim=True)[0]
    d_norm = (d - d_min) / (d_max - d_min + 1e-6)

    # Scatter depth values onto image grid (simple argmax -- take max depth per pixel)
    depth_map = torch.zeros(B, img_size * img_size, device=xyz.device)
    px = u_norm.long().clamp(0, img_size - 1) * img_size + v_norm.long().clamp(0, img_size - 1)

    # scatter_reduce: for each pixel take max depth
    depth_map = depth_map.scatter_reduce(1, px, d_norm, reduce="amax", include_self=True)
    depth_map = depth_map.reshape(B, 1, img_size, img_size)
    return depth_map


# --------------------------------------------------------------------------- #
#  Lightweight per-view image encoder (CLIP image encoder stand-in)           #
# --------------------------------------------------------------------------- #


class ViewEncoder(nn.Module):
    """Small conv encoder: processes a (B, 1, H, W) depth map -> (B, D) feature."""

    def __init__(self, img_size: int = 16, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        # Global average pool after
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        return self.pool(self.net(x)).flatten(1)  # (B, out_dim)


# --------------------------------------------------------------------------- #
#  PointCLIP V2 model                                                          #
# --------------------------------------------------------------------------- #


class PointCLIPV2(nn.Module):
    """PointCLIP V2: multi-view depth projection + shared vision encoder."""

    def __init__(
        self, num_views: int = 3, img_size: int = 16, feat_dim: int = 64, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.img_size = img_size

        # Shared view encoder (mimics CLIP's image encoder; here lightweight + random init)
        self.view_encoder = ViewEncoder(img_size=img_size, out_dim=feat_dim)

        # Classification head on aggregated multi-view features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        view_feats = []
        for v in range(self.num_views):
            depth_map = project_to_depth_map(xyz, v, self.img_size)  # (B, 1, H, W)
            feat = self.view_encoder(depth_map)  # (B, feat_dim)
            view_feats.append(feat)

        # Average across views
        multi_view_feat = torch.stack(view_feats, dim=1).mean(dim=1)  # (B, feat_dim)
        return self.head(multi_view_feat)


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointCLIPV2(num_views=3, img_size=16, feat_dim=64, num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_pointclip_v2_model() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointCLIP V2 (multi-view depth projection + shared vision encoder)",
        "build_pointclip_v2_model",
        "example_input",
        "2023",
        "DC",
    ),
]
