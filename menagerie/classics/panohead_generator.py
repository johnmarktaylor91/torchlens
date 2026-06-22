"""PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360 Degrees.

An et al. (Microsoft Research) CVPR 2023.  arXiv:2303.13071.
Source: https://github.com/SizheAn/PanoHead

PanoHead's distinctive primitives:
  - **360-degree full head generation**: unlike EG3D which focuses on the frontal
    hemisphere, PanoHead extends tri-plane GAN synthesis to the full sphere by using
    a **hybrid tri-grid** representation: XY, XZ, YZ planes but with an expanded
    angular range and a learned background sphere.
  - **Tri-plane synthesis**: same as EG3D -- a StyleGAN2 backbone generates three
    2D feature planes; 3D points are queried by projecting onto all three planes and
    summing features.
  - **360 camera sampling**: rays from any azimuth angle in [0, 2pi] are supported,
    not just the frontal hemisphere.  Implemented via appropriate camera-to-world
    transforms (reproduced here as a random pose vector).
  - **Dual discriminator**: a face discriminator + a full-head discriminator (not
    reproduced here).
  - **NeRF volume rendering** (points sampled along camera rays, features decoded to
    RGB + density, accumulated via alpha compositing) -- simplified here to a single
    query point for compact tracing.

Here we reproduce:
  - StyleGAN2 mapping + tri-plane generator (identical primitive to GET3D).
  - 360-aware ray sampler stub (random pose -> single 3D query point).
  - NeRF MLP decoder: tri-plane features -> RGB + density -> single colour output.

Random init, CPU, small channels for compact tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 32, w_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StyleBlock(nn.Module):
    """StyleGAN2 synthesis block: upsample + modulated conv."""

    def __init__(self, in_c: int, out_c: int, w_dim: int) -> None:
        super().__init__()
        self.style = nn.Linear(w_dim, in_c)
        self.weight = nn.Parameter(torch.randn(out_c, in_c, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.noise_w = nn.Parameter(torch.zeros(1, out_c, 1, 1))
        self.in_c, self.out_c = in_c, out_c

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        s = self.style(w) + 1.0
        wt = self.weight.unsqueeze(0) * s.view(B, 1, self.in_c, 1, 1)
        denom = wt.pow(2).sum([2, 3, 4], keepdim=True).add(1e-8).sqrt()
        wt = (wt / denom).view(B * self.out_c, self.in_c, 3, 3)
        x_flat = x.reshape(1, B * self.in_c, x.shape[2], x.shape[3])
        out = F.conv2d(x_flat, wt, padding=1, groups=B).view(B, self.out_c, x.shape[2], x.shape[3])
        noise = torch.randn(B, 1, out.shape[2], out.shape[3], device=x.device)
        return F.leaky_relu(out + self.noise_w * noise + self.bias.view(1, -1, 1, 1), 0.2)


class TriPlaneGenerator(nn.Module):
    """StyleGAN2-style tri-plane generator: w -> XY, XZ, YZ feature planes."""

    def __init__(self, w_dim: int = 64, nf: int = 8) -> None:
        super().__init__()
        # Three independent const feature maps (4x4) for XY, XZ, YZ planes
        self.const_xy = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        self.const_xz = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        self.const_yz = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        # Two upsample blocks per plane: 4->8->16
        self.xy1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.xy2 = StyleBlock(nf * 2, nf, w_dim)
        self.xz1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.xz2 = StyleBlock(nf * 2, nf, w_dim)
        self.yz1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.yz2 = StyleBlock(nf * 2, nf, w_dim)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = w.shape[0]
        xy = self.xy2(self.xy1(self.const_xy.expand(B, -1, -1, -1), w), w)
        xz = self.xz2(self.xz1(self.const_xz.expand(B, -1, -1, -1), w), w)
        yz = self.yz2(self.yz1(self.const_yz.expand(B, -1, -1, -1), w), w)
        return xy, xz, yz


def _sample_plane(plane: torch.Tensor, coords2d: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample features at 2D coordinates from a feature plane."""
    # plane: (B, C, H, W),  coords2d: (B, N, 2) in [-1,1]
    grid = coords2d.unsqueeze(1)  # (B, 1, N, 2)
    return F.grid_sample(plane, grid, align_corners=True, mode="bilinear").squeeze(2)


class NeRFDecoder(nn.Module):
    """Tiny NeRF MLP: tri-plane features -> RGB + density."""

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
        )
        self.rgb_head = nn.Linear(hidden, 3)
        self.sigma_head = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.net(feat)
        rgb = torch.sigmoid(self.rgb_head(h))
        sigma = F.softplus(self.sigma_head(h))
        return torch.cat([rgb, sigma], dim=-1)  # (B, 4)


class PanoHeadTriplaneGenerator(nn.Module):
    """PanoHead: StyleGAN2 tri-plane synthesis + 360 ray + NeRF MLP decoder."""

    def __init__(self, z_dim: int = 32, w_dim: int = 64, nf: int = 8) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.triplane = TriPlaneGenerator(w_dim, nf)
        # NeRF decoder: input = sum of 3 plane features
        self.nerf_mlp = NeRFDecoder(feat_dim=nf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, z_dim + 3)  -- last 3 are 3D query point (full 360 viewpoint)
        z = x[:, : self.z_dim]
        pts = x[:, self.z_dim :]  # (B, 3) in [-1,1]
        pts_q = pts.unsqueeze(1)  # (B, 1, 3) single query point
        w = self.mapping(z)
        xy, xz, yz = self.triplane(w)
        # Tri-plane sampling: project onto XY, XZ, YZ planes
        feat_xy = _sample_plane(xy, pts_q[..., :2])  # (B, nf, 1)
        feat_xz = _sample_plane(xz, pts_q[..., [0, 2]])
        feat_yz = _sample_plane(yz, pts_q[..., 1:])
        feat = (feat_xy + feat_xz + feat_yz).squeeze(-1)  # (B, nf)
        return self.nerf_mlp(feat)  # (B, 4)  RGB+sigma


def build_panohead_triplane_generator() -> nn.Module:
    return PanoHeadTriplaneGenerator()


def example_input() -> torch.Tensor:
    # z (32) + 3D query point (3)
    return torch.randn(1, 32 + 3)


MENAGERIE_ENTRIES = [
    (
        "PanoHead Triplane Generator (360-degree full-head tri-plane GAN + NeRF decoder)",
        "build_panohead_triplane_generator",
        "example_input",
        "2023",
        "DC",
    ),
]
