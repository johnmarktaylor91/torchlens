"""pixelNeRF: image-conditioned NeRF with per-pixel feature sampling.

Yu et al. (2021), "pixelNeRF: Neural Radiance Fields from One or Few Images".
CVPR 2021.  arXiv:2012.02190.
Source: https://github.com/sxyu/pixel-nerf

Distinctive primitives:
  1. IMAGE ENCODER (ResNet-34 here, compact ResNet-like below): encodes the
     reference image into a dense feature grid F (B, C, H, W).
  2. FEATURE SAMPLING: for each 3D query point p, project to 2D pixel coords via
     camera intrinsics/extrinsics, then bilinearly sample F -> feature vector f(p).
  3. POSITIONAL ENCODING: encode 3D point x and view direction d with sin/cos PE.
  4. NeRF MLP: takes [PE(x), f(p)] -> (RGB, density).  The image features F are
     the conditioning signal -- the same MLP generalises to novel scenes by
     sampling different F.

The camera projection is simplified here: we use a known identity-like camera
(orthographic projection) so the atlas can run without external camera matrices.

Compact config: n_points=8, d_feat=16, d_hidden=32, n_resblocks=2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Compact ResNet image encoder
# ==============================================================


class ResBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + x)


class TinyResNetEncoder(nn.Module):
    """Compact image encoder: (B, 3, H, W) -> feature grid (B, d_feat, H//4, W//4)."""

    def __init__(self, d_feat: int = 16, n_blocks: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, d_feat, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feat),
            nn.ReLU(),
            nn.Conv2d(d_feat, d_feat, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_feat),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(d_feat) for _ in range(n_blocks)])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) -> (B, d_feat, H//4, W//4)"""
        return self.blocks(self.stem(img))


# ==============================================================
# Positional encoding
# ==============================================================


def posenc(x: torch.Tensor, n_freqs: int = 4) -> torch.Tensor:
    """Positional encoding: [x, sin(2^k x), cos(2^k x)] for k=0..n_freqs-1."""
    out = [x]
    for k in range(n_freqs):
        freq = 2.0**k
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


# ==============================================================
# NeRF MLP
# ==============================================================


class NeRFMLP(nn.Module):
    """NeRF field MLP: [PE(x, view_dir), image_feature] -> (RGB, density)."""

    def __init__(self, d_pe: int, d_feat: int = 16, d_hidden: int = 32) -> None:
        super().__init__()
        d_in = d_pe + d_feat
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 4),  # (R, G, B, density)
        )

    def forward(self, pe: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """pe: (B, N, d_pe), feat: (B, N, d_feat) -> (B, N, 4)"""
        inp = torch.cat([pe, feat], dim=-1)
        out = self.net(inp)
        return out


# ==============================================================
# pixelNeRF model
# ==============================================================


class PixelNeRFResnet34(nn.Module):
    """pixelNeRF: ResNet encoder + bilinear feature sampling + NeRF MLP.

    Input tensor: (B, 3+n_points*3, H, W) packed:
      - First 3 channels: reference image (B, 3, H, W)
      - Remaining n_points*3 channels: 3D query points flattened into spatial dims
        as (B, n_points*3, H, W) -> extracted via mean/reshape trick

    Actually simpler: we pack as a flat (B, 3*H*W + n_points*3) float tensor.
    But to support clean tracing, we wrap into a single tensor input.

    Simplified camera: orthographic projection (xy of 3D point maps to H/2 +- x*scale).
    """

    def __init__(
        self,
        d_feat: int = 16,
        n_resblocks: int = 2,
        n_points: int = 8,
        n_freqs: int = 2,
        d_hidden: int = 32,
        img_h: int = 16,
        img_w: int = 16,
    ) -> None:
        super().__init__()
        self.d_feat = d_feat
        self.n_points = n_points
        self.img_h = img_h
        self.img_w = img_w
        self.encoder = TinyResNetEncoder(d_feat, n_resblocks)
        # PE dim: (3+3) directions x (1 + 2*n_freqs)
        d_pe = 6 * (1 + 2 * n_freqs)
        self.n_freqs = n_freqs
        self.nerf_mlp = NeRFMLP(d_pe, d_feat, d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3*img_h*img_w + n_points*6) packed float.

        First 3*H*W = flattened reference image.
        Next n_points*6 = (x,y,z, dx,dy,dz) per query point.
        Output: (B, n_points, 4) = (RGB, density) per point.
        """
        B = x.shape[0]
        H, W = self.img_h, self.img_w
        n_img = 3 * H * W
        # Split
        img_flat = x[:, :n_img]
        pts_flat = x[:, n_img:]  # (B, n_points*6)

        # Decode image
        img = img_flat.view(B, 3, H, W)
        feat_grid = self.encoder(img)  # (B, d_feat, H//4, W//4)
        fH, fW = feat_grid.shape[2], feat_grid.shape[3]

        # Query points: (B, n_points, 6) = xyz + view_dir
        pts = pts_flat.view(B, self.n_points, 6)
        xyz = pts[:, :, :3]  # (B, n_points, 3)
        viewdir = pts[:, :, 3:]  # (B, n_points, 3)

        # Simplified orthographic projection: xy in [-1,1] -> grid_sample coords
        # Use tanh to keep xy in valid range
        grid_x = torch.tanh(xyz[:, :, 0])  # (B, n_points) in (-1,1)
        grid_y = torch.tanh(xyz[:, :, 1])
        # grid_sample expects (B, H_out, W_out, 2), output (B, C, H_out, W_out)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, n_points, 1, 2)
        # Sample features from grid
        sampled = F.grid_sample(
            feat_grid, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B, d_feat, n_points, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, n_points, d_feat)

        # PE
        pe_pts = posenc(torch.cat([xyz, viewdir], dim=-1), self.n_freqs)  # (B, n_points, d_pe)

        # NeRF MLP
        out = self.nerf_mlp(pe_pts, sampled)  # (B, n_points, 4)
        return out.reshape(B, -1)  # (B, n_points*4)


def build_pixelnerf_resnet34() -> nn.Module:
    model = PixelNeRFResnet34(
        d_feat=16, n_resblocks=2, n_points=8, n_freqs=2, d_hidden=32, img_h=16, img_w=16
    )
    return model.eval()


def example_input() -> torch.Tensor:
    """(1, 3*256 + 8*6) = (1, 816) packed: 16x16 image + 8 query points."""
    return torch.randn(1, 3 * 16 * 16 + 8 * 6)


MENAGERIE_ENTRIES = [
    (
        "pixelNeRF ResNet34 (image-conditioned NeRF: ResNet encoder + bilinear feature sampling + NeRF MLP)",
        "build_pixelnerf_resnet34",
        "example_input",
        "2021",
        "DC",
    ),
]
