"""GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images.

Gao et al. (NVIDIA) NeurIPS 2022.  arXiv:2209.11163.
Source: https://github.com/nv-tlabs/GET3D

GET3D's distinctive primitives:
  - **Dual generator**: a shape generator (z_shape -> SDF/mesh geometry) and a
    texture generator (z_tex -> tri-plane feature volumes) work in parallel.
  - **Tri-plane feature representation**: three axis-aligned 2D feature planes
    (XY, XZ, YZ) are synthesized by the texture generator.  At any 3D point, the
    feature vector is the sum of bilinear-sampled projections onto the three planes.
  - **StyleGAN2-style synthesis**: both generators share a mapping network (z -> w)
    and use modulated-conv synthesis blocks (noise injection + AdaIN-style
    per-channel affine conditioning from w).
  - **DMTet integration**: differentiable mesh tessellation from a predicted SDF
    (not reproduced here -- tracing operates on the feature volumes only).

Here we reproduce:
  1. Shared mapping network z -> w.
  2. Tri-plane texture generator: w -> three HxW feature planes via StyleGAN2-style
     modulated-conv blocks.
  3. Tri-plane sampler: given a 3D query point, projects onto each plane and sums.
  4. SDF shape head: w -> small MLP SDF (geometry branch).

The full model also has a DMTet renderer; we wrap everything to return the summed
tri-plane feature at a single query point for tracing.

Random init, CPU, tiny spatial/channel for clean tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    """Maps latent z -> style code w (shared by shape + texture branches)."""

    def __init__(self, z_dim: int = 32, w_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated conv2d: weight is modulated by a per-channel style vector."""

    def __init__(self, in_c: int, out_c: int, k: int, w_dim: int, pad: int = 1) -> None:
        super().__init__()
        self.out_c = out_c
        self.in_c = in_c
        self.k = k
        self.pad = pad
        self.weight = nn.Parameter(torch.randn(out_c, in_c, k, k))
        self.mod = nn.Linear(w_dim, in_c)  # style -> per-channel scale

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Per-sample style modulation
        style = self.mod(w) + 1.0  # (B, in_c)  -- +1 for identity init
        # Weight modulation: w_mod[b,o,i,h,v] = w[o,i,h,v] * style[b,i]
        wt = self.weight.unsqueeze(0) * style.view(B, 1, self.in_c, 1, 1)
        # Demodulate (normalize per output channel)
        denom = wt.pow(2).sum([2, 3, 4], keepdim=True).add(1e-8).sqrt()
        wt = wt / denom  # (B, out_c, in_c, k, k)
        # Fold batch into group conv
        x = x.reshape(1, B * self.in_c, x.shape[2], x.shape[3])
        wt = wt.view(B * self.out_c, self.in_c, self.k, self.k)
        out = F.conv2d(x, wt, padding=self.pad, groups=B)
        return out.view(B, self.out_c, out.shape[2], out.shape[3])


class StyleBlock(nn.Module):
    """Single StyleGAN2 synthesis block: modulated conv + upsample + noise injection."""

    def __init__(self, in_c: int, out_c: int, w_dim: int) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_c, out_c, 3, w_dim)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_c, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv(x, w)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + self.noise_weight * noise + self.bias
        return F.leaky_relu(x, 0.2)


class TriPlaneGenerator(nn.Module):
    """Tri-plane texture generator: w -> (XY plane, XZ plane, YZ plane).

    Three separate 2D synthesis stacks produce feature planes.
    """

    def __init__(self, w_dim: int = 64, nf: int = 8, plane_res: int = 16) -> None:
        super().__init__()
        self.plane_res = plane_res
        # Shared const starting feature map for all three planes
        self.const_xy = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        self.const_xz = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        self.const_yz = nn.Parameter(torch.randn(1, nf * 4, 4, 4))
        # Synthesis blocks for each plane (4->8->16)
        self.xy_blk1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.xy_blk2 = StyleBlock(nf * 2, nf, w_dim)
        self.xz_blk1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.xz_blk2 = StyleBlock(nf * 2, nf, w_dim)
        self.yz_blk1 = StyleBlock(nf * 4, nf * 2, w_dim)
        self.yz_blk2 = StyleBlock(nf * 2, nf, w_dim)

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = w.shape[0]
        xy = self.const_xy.expand(B, -1, -1, -1)
        xy = self.xy_blk2(self.xy_blk1(xy, w), w)  # (B, nf, 16, 16)
        xz = self.const_xz.expand(B, -1, -1, -1)
        xz = self.xz_blk2(self.xz_blk1(xz, w), w)
        yz = self.const_yz.expand(B, -1, -1, -1)
        yz = self.yz_blk2(self.yz_blk1(yz, w), w)
        return xy, xz, yz


class TriPlaneSampler(nn.Module):
    """Sample tri-plane features at 3D query points by projecting onto each plane."""

    def forward(
        self,
        xy: torch.Tensor,
        xz: torch.Tensor,
        yz: torch.Tensor,
        pts: torch.Tensor,  # (B, N, 3) in [-1, 1]
    ) -> torch.Tensor:
        def _sample_plane(plane: torch.Tensor, coords2d: torch.Tensor) -> torch.Tensor:
            # plane: (B, C, H, W), coords2d: (B, N, 2)
            grid = coords2d.unsqueeze(1)  # (B, 1, N, 2)
            return F.grid_sample(plane, grid, align_corners=True, mode="bilinear").squeeze(2)
            # output: (B, C, N)

        xy_feat = _sample_plane(xy, pts[..., :2])  # project onto XY plane
        xz_feat = _sample_plane(xz, pts[..., [0, 2]])  # XZ plane
        yz_feat = _sample_plane(yz, pts[..., 1:])  # YZ plane
        return xy_feat + xz_feat + yz_feat  # (B, C, N)


class SDFHead(nn.Module):
    """Shape branch: w -> small MLP producing SDF values (geometry)."""

    def __init__(self, w_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.net(w)


class GET3DGenerator(nn.Module):
    """GET3D dual generator: tri-plane texture + SDF shape branch.

    Returns concatenation of: tri-plane feature at query points (B, C) + SDF scalar.
    Wrapped to return a single flat tensor for clean tracing.
    """

    def __init__(self, z_dim: int = 32, w_dim: int = 64, nf: int = 8) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.triplane = TriPlaneGenerator(w_dim, nf)
        self.sampler = TriPlaneSampler()
        self.sdf_head = SDFHead(w_dim)
        self.nf = nf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, z_dim + 3)  last 3 dims = query 3D point
        z = x[:, : self.z_dim]
        pts = x[:, self.z_dim :]  # (B, 3)
        pts_norm = pts.unsqueeze(1)  # (B, 1, 3) -- single query point
        w = self.mapping(z)
        xy, xz, yz = self.triplane(w)
        feat = self.sampler(xy, xz, yz, pts_norm)  # (B, nf, 1)
        feat = feat.squeeze(-1)  # (B, nf)
        sdf = self.sdf_head(w)  # (B, 1)
        return torch.cat([feat, sdf], dim=-1)  # (B, nf+1)


def build_get3d_generator() -> nn.Module:
    return GET3DGenerator()


def example_input() -> torch.Tensor:
    # z (32) + query point (3)
    return torch.randn(1, 32 + 3)


MENAGERIE_ENTRIES = [
    (
        "GET3D Generator (tri-plane texture synthesis + SDF shape branch, dual GAN)",
        "build_get3d_generator",
        "example_input",
        "2022",
        "DC",
    ),
]
