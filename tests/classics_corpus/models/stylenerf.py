"""StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis.

Gu et al., ICLR 2022.  arXiv:2110.08985.
Source: https://github.com/facebookresearch/StyleNeRF

StyleNeRF integrates Neural Radiance Fields (NeRF) volume rendering with StyleGAN2-style
modulation for 3D-aware high-resolution image generation.

Distinctive primitives:
  - **NeRF MLP with style modulation**: a compact MLP maps (x, y, z, view_dir) -> (RGB, sigma)
    where EACH linear layer is modulated by the style code w (weight-demodulated modulated
    linear, analogous to StyleGAN2's modulated conv2d but for 1D feature vectors).
  - **Volume rendering**: along each camera ray, sample N points -> query NeRF MLP ->
    accumulate (RGB, sigma) via alpha compositing to produce a low-res feature map.
  - **2D upsampler**: the low-res volume-rendered feature map is upsampled to high-res
    using a StyleGAN2-style synthesis network.
  - **Camera conditioning**: camera extrinsics (azimuth, elevation) parameterize ray origins
    and directions. Here we use random rays for compact verification.

Compact: z_dim=64, w_dim=64, n_samples=4 (ray samples), volume 4x4, upsample to 16x16.

Also: stylenerf_encoder -- an inversion encoder that maps an image to w latent space
(from GAN inversion; common companion to StyleNeRF for real-image editing).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Mapping network
# ============================================================


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 64, w_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.pixel_norm(z))


# ============================================================
# Style-modulated linear (NeRF MLP layers)
# ============================================================


class ModulatedLinear(nn.Module):
    """Style-modulated fully-connected layer for NeRF MLP.

    Analogous to StyleGAN2's modulated conv2d but for 1D feature vectors:
    weight_ij *= style_i / sqrt(sum_i style_i^2 * w_ij^2).
    Applied per-sample to the NeRF point features.
    """

    def __init__(self, in_dim: int, out_dim: int, w_dim: int, demodulate: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.demodulate = demodulate
        self.affine = nn.Linear(w_dim, in_dim, bias=True)
        nn.init.ones_(self.affine.bias)
        self.weight = nn.Parameter(torch.randn(1, out_dim, in_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)  w: (B, w_dim)
        B, N, _ = x.shape
        s = self.affine(w)  # (B, in_dim)
        weight = self.weight * s.view(B, 1, self.in_dim)  # (B, out_dim, in_dim)
        if self.demodulate:
            d = weight.pow(2).sum(dim=-1, keepdim=True).add(1e-8).sqrt()
            weight = weight / d
        # Batched matmul: (B, N, in_dim) x (B, in_dim, out_dim) -> (B, N, out_dim)
        out = torch.bmm(x, weight.transpose(1, 2))
        return out + self.bias.view(1, 1, -1)


# ============================================================
# NeRF MLP with style modulation (StyleNeRF core)
# ============================================================


class StyleNeRFMLP(nn.Module):
    """StyleNeRF radiance field MLP.

    Input: (x, y, z, viewdir) positional-encoded -> (B, N, pos_dim)
    Conditioned on style w from the mapping network.
    Output: (B, N, 4) -- RGB (3) + density sigma (1).

    Each linear layer is a ModulatedLinear (style-conditioned weight demodulation).
    This is the key contribution: 3D-aware synthesis where density/color are style-controlled.
    """

    def __init__(
        self,
        pos_dim: int = 16,  # positional encoding output dim
        hidden_dim: int = 32,
        w_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        # Positional encoding projection (Fourier embedding of 3D coords)
        self.pos_emb = nn.Linear(6, pos_dim)  # 3D xyz + 3D view_dir -> pos_dim
        # Style-modulated MLP layers
        dims = [pos_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [ModulatedLinear(dims[i], dims[i + 1], w_dim) for i in range(n_layers)]
        )
        self.acts = nn.ModuleList([nn.Softplus() for _ in range(n_layers)])
        # Output: RGB + sigma
        self.to_rgb = nn.Linear(hidden_dim, 3)
        self.to_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, pts: torch.Tensor, dirs: torch.Tensor, w: torch.Tensor) -> tuple:
        # pts: (B, N, 3) ray sample positions
        # dirs: (B, N, 3) view directions
        # w: (B, w_dim) style vector
        x = torch.cat([pts, dirs], dim=-1)  # (B, N, 6)
        # Simple Fourier-like embedding via sine
        x = torch.sin(self.pos_emb(x))  # (B, N, pos_dim)
        for layer, act in zip(self.layers, self.acts):
            x = act(layer(x, w))
        rgb = torch.sigmoid(self.to_rgb(x))  # (B, N, 3)
        sigma = F.softplus(self.to_sigma(x))  # (B, N, 1)
        return rgb, sigma


# ============================================================
# Volume rendering
# ============================================================


def volume_render(
    rgb: torch.Tensor,  # (B, N, 3)
    sigma: torch.Tensor,  # (B, N, 1)
    z_vals: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """Alpha compositing along rays to produce per-pixel color.

    Returns (B, 3) accumulated color per ray.
    """
    # Deltas between adjacent sample depths
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # (B, N-1)
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (B, N)
    # Alpha values
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (B, N)
    # Transmittance
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1] + 1e-10], dim=-1),
        dim=-1,
    )  # (B, N)
    weights = alpha * T  # (B, N)
    # Accumulated color
    color = (weights.unsqueeze(-1) * rgb).sum(dim=1)  # (B, 3)
    return color


# ============================================================
# 2D upsampler for StyleNeRF (StyleGAN2-style synthesis)
# ============================================================


class ModulatedConv2d(nn.Module):
    """Weight-demodulated modulated conv2d (StyleGAN2 op) for 2D upsampler."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int, upsample: bool = False) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        k = 3
        self.affine = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.affine.bias)
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, k, k) / math.sqrt(in_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.weight.size(-1)
        s = self.affine(w)
        weight = self.weight * s.view(B, 1, self.in_ch, 1, 1)
        d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
        weight = weight / d
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x.reshape(1, B * C, x.size(2), x.size(3))
        weight = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x, weight, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.size(2), out.size(3))
        return out + self.bias.view(1, -1, 1, 1)


class StyleNeRF2DUpsampler(nn.Module):
    """StyleGAN2-style 2D upsampling network for the volume-rendered feature map."""

    def __init__(self, in_ch: int, base_ch: int, w_dim: int) -> None:
        super().__init__()
        self.conv0 = ModulatedConv2d(in_ch, base_ch, w_dim, upsample=True)
        self.act0 = nn.LeakyReLU(0.2)
        self.conv1 = ModulatedConv2d(base_ch, base_ch, w_dim, upsample=True)
        self.act1 = nn.LeakyReLU(0.2)
        self.to_rgb = nn.Conv2d(base_ch, 3, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.conv0(x, w))
        x = self.act1(self.conv1(x, w))
        return torch.tanh(self.to_rgb(x))


# ============================================================
# StyleNeRF Generator
# ============================================================


class StyleNeRFGenerator(nn.Module):
    """StyleNeRF 3D-aware generator (compact).

    z -> mapping -> w ->
    [NeRF MLP (style-modulated)] -> volume rendering on 4x4 ray grid ->
    [2D StyleGAN upsampler] -> 16x16 RGB output.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        n_samples: int = 4,  # samples per ray (compact; paper uses 64)
        base_ch: int = 16,
        grid_size: int = 4,  # spatial resolution of volume render
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.nerf_mlp = StyleNeRFMLP(pos_dim=16, hidden_dim=32, w_dim=w_dim, n_layers=2)
        # Volume render outputs 3-channel color image at grid_size x grid_size
        # Then 2D upsampler brings it to 16x16
        self.upsampler = StyleNeRF2DUpsampler(in_ch=3, base_ch=base_ch, w_dim=w_dim)

    def _generate_rays(self, B: int, H: int, W: int, device: torch.device):
        """Generate simple orthographic camera rays for a HxW grid."""
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Ray origins: pixel positions on image plane
        origins = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), torch.zeros(H * W, device=device)], dim=-1
        )  # (H*W, 3)
        # Ray directions: all pointing into the scene (z+)
        dirs = torch.zeros_like(origins)
        dirs[..., 2] = 1.0
        origins = origins.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 3)
        dirs = dirs.unsqueeze(0).expand(B, -1, -1)
        return origins, dirs

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        device = z.device
        w = self.mapping(z)
        H = W = self.grid_size
        N_rays = H * W
        # Generate rays
        origins, dirs = self._generate_rays(B, H, W, device)
        # Sample points along each ray
        z_near, z_far = 2.0, 6.0
        z_vals = torch.linspace(z_near, z_far, self.n_samples, device=device)
        z_vals = z_vals.view(1, 1, self.n_samples).expand(B, N_rays, -1)  # (B, N_rays, n_samples)
        # Points: origins + t * dirs for each sample
        pts = origins.unsqueeze(2) + z_vals.unsqueeze(-1) * dirs.unsqueeze(2)
        # (B, N_rays, n_samples, 3)
        pts_flat = pts.view(B, N_rays * self.n_samples, 3)
        dirs_flat = dirs.unsqueeze(2).expand_as(pts).reshape(B, N_rays * self.n_samples, 3)
        # NeRF MLP query
        rgb_flat, sigma_flat = self.nerf_mlp(pts_flat, dirs_flat, w)
        rgb = rgb_flat.view(B, N_rays, self.n_samples, 3)
        sigma = sigma_flat.view(B, N_rays, self.n_samples, 1)
        # Volume render per ray
        z_for_render = z_vals  # (B, N_rays, n_samples)
        rgb_rays, sigma_rays = (
            rgb.reshape(B * N_rays, self.n_samples, 3),
            sigma.reshape(B * N_rays, self.n_samples, 1),
        )
        z_render = z_for_render.reshape(B * N_rays, self.n_samples)
        color = volume_render(rgb_rays, sigma_rays, z_render)  # (B*N_rays, 3)
        color = color.view(B, N_rays, 3).permute(0, 2, 1).view(B, 3, H, W)  # (B, 3, H, W)
        # 2D upsampler
        return self.upsampler(color, w)


def build_stylenerf_generator() -> nn.Module:
    return StyleNeRFGenerator()


def example_input_stylenerf_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# StyleNeRF Encoder (GAN inversion)
# ============================================================


class StyleNeRFEncoder(nn.Module):
    """StyleNeRF inversion encoder: image -> w latent code.

    Used for projecting real images into the StyleNeRF latent space
    (GAN inversion / real image editing). Architecture: a convolutional
    encoder that maps an RGB image to a w-code.
    """

    def __init__(self, base_ch: int = 32, w_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),  # 2x2
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(base_ch * 4, w_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, H, W)  -> w: (B, w_dim)
        feats = self.encoder(img)
        return self.fc(feats)


def build_stylenerf_encoder() -> nn.Module:
    return StyleNeRFEncoder()


def example_input_stylenerf_encoder() -> torch.Tensor:
    return torch.randn(1, 3, 16, 16)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylenerf_generator",
        "build_stylenerf_generator",
        "example_input_stylenerf_generator",
        "2022",
        "DC",
    ),
    (
        "stylenerf_encoder",
        "build_stylenerf_encoder",
        "example_input_stylenerf_encoder",
        "2022",
        "DC",
    ),
]
