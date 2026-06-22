"""StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis.

Sauer et al., ICML 2023.  arXiv:2301.09515.
Source: https://github.com/autonomousvision/stylegan-t

StyleGAN-T is a text-conditioned GAN (fast alternative to diffusion) based on StyleGAN3
with several key modifications for large-scale T2I:

Distinctive primitives:
  - **Text conditioning via CLIP-like embedding**: a text embedding (or random vector
    in our rand-init compact version) feeds both the mapping network and direct
    cross-attention layers in the synthesis blocks.
  - **Projected discriminator**: patches of features from a DINO-pretrained backbone
    (here approximated as a lightweight random conv feature extractor) are projected
    and classified, rather than training discriminator from scratch on pixel space.
  - **Stacked synthesis stages**: generator has more upsampling stages than StyleGAN3-T;
    text cross-attention is injected at multiple resolutions.
  - **Alias-free filtered lrelu** inherited from StyleGAN3.

Compact: z_dim=64, w_dim=64, text_dim=64 (random embedding vector), output 32x32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared utilities
# ============================================================


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetwork(nn.Module):
    """Mapping network conditioned on text embedding (concatenated to z)."""

    def __init__(
        self, z_dim: int = 64, w_dim: int = 64, text_dim: int = 64, n_layers: int = 4
    ) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        in_dim = z_dim + text_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        z = self.pixel_norm(z)
        return self.net(torch.cat([z, text_emb], dim=1))


# ============================================================
# Alias-free filtered activation (from StyleGAN3)
# ============================================================


class FilteredLReLU(nn.Module):
    """Upsample -> leaky_relu -> low-pass filter (alias-free activation)."""

    def __init__(self, channels: int, upsample: bool = True) -> None:
        super().__init__()
        self.do_upsample = upsample
        k = 5
        sigma = 1.5
        g = torch.tensor(
            [math.exp(-((i - k // 2) ** 2) / (2 * sigma**2)) for i in range(k)],
            dtype=torch.float32,
        )
        kernel = g[:, None] * g[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("lpf", kernel.view(1, 1, k, k).expand(channels, 1, k, k).clone())
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(x, 0.2)
        x = F.conv2d(x, self.lpf, padding=2, groups=self.channels)
        if self.do_upsample:
            x = F.avg_pool2d(x, 2)
        return x


# ============================================================
# Modulated synthesis layer with text cross-attention
# ============================================================


class TextCrossAttention(nn.Module):
    """Cross-attention: spatial tokens (Q) attend to text tokens (K/V)."""

    def __init__(self, feat_dim: int, text_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(feat_dim, feat_dim)
        self.k = nn.Linear(text_dim, feat_dim)
        self.v = nn.Linear(text_dim, feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim**-0.5

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  text: (B, T, text_dim)
        B, C, H, W = x.shape
        # flatten spatial
        xf = x.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        q = self.q(xf)
        k = self.k(text)
        v = self.v(text)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out(out)
        return out.permute(0, 2, 1).view(B, C, H, W)


class StyleGANTSynthesisLayer(nn.Module):
    """StyleGAN-T synthesis layer: modulated conv + filtered lrelu + optional text attn."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        text_dim: int = 64,
        upsample: bool = True,
        use_text_attn: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = upsample
        self.affine = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.affine.bias)
        k = 3
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) / math.sqrt(in_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.filtered_act = FilteredLReLU(out_ch, upsample=upsample)
        self.use_text_attn = use_text_attn
        if use_text_attn:
            self.text_attn = TextCrossAttention(out_ch, text_dim)
            self.attn_norm = nn.GroupNorm(min(4, out_ch), out_ch)

    def forward(self, x: torch.Tensor, w: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        s = self.affine(w)
        k = self.weight.size(-1)
        weight = self.weight.unsqueeze(0) * s.view(B, 1, self.in_ch, 1, 1)
        d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
        weight = weight / d
        x_in = x.reshape(1, B * self.in_ch, x.size(2), x.size(3))
        w_flat = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x_in, w_flat, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.size(2), out.size(3))
        out = out + self.bias.view(1, -1, 1, 1)
        out = self.filtered_act(out)
        if self.use_text_attn:
            out = out + self.text_attn(self.attn_norm(out), text)
        return out


# ============================================================
# StyleGAN-T Generator
# ============================================================


class StyleGANTGenerator(nn.Module):
    """StyleGAN-T text-conditioned generator (compact).

    Inputs: z (noise latent) + text_emb (CLIP text embedding, here random).
    Architecture: mapping (z+text -> w) -> constant 4x4 input ->
    3 alias-free upsampling synthesis layers with text cross-attention -> to-RGB.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        text_dim: int = 64,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.mapping = MappingNetwork(z_dim, w_dim, text_dim, n_layers=4)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        ch = [base_ch * 4, base_ch * 2, base_ch, base_ch]
        self.layers = nn.ModuleList(
            [
                StyleGANTSynthesisLayer(
                    ch[0], ch[1], w_dim, text_dim, upsample=True, use_text_attn=True
                ),
                StyleGANTSynthesisLayer(
                    ch[1], ch[2], w_dim, text_dim, upsample=True, use_text_attn=True
                ),
                StyleGANTSynthesisLayer(
                    ch[2], ch[3], w_dim, text_dim, upsample=True, use_text_attn=False
                ),
            ]
        )
        self.to_rgb = nn.Conv2d(ch[-1], 3, 1)

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # text_emb: (B, text_dim) -- single vector; expand to token sequence (B, 1, text_dim)
        B = z.size(0)
        text_tokens = text_emb.unsqueeze(1)  # (B, 1, text_dim)
        w = self.mapping(z, text_emb)
        x = self.const.expand(B, -1, -1, -1)
        for layer in self.layers:
            x = layer(x, w, text_tokens)
        return torch.tanh(self.to_rgb(x))


def build_stylegan_t_generator() -> nn.Module:
    return StyleGANTGenerator()


def example_input_stylegan_t_generator() -> tuple:
    """Returns (z, text_emb) where text_emb is a random CLIP-like embedding."""
    z = torch.randn(1, 64)
    text_emb = torch.randn(1, 64)
    return (z, text_emb)


# ============================================================
# StyleGAN-T Text-to-Image Generator (same arch, separate entry)
# ============================================================


def build_stylegan_t_text_to_image_generator() -> nn.Module:
    """StyleGAN-T text-to-image generator (same architecture as stylegan_t_generator)."""
    return StyleGANTGenerator()


def example_input_stylegan_t_text_to_image_generator() -> tuple:
    """(z, random_text_embedding) -- text_emb stands in for CLIP text encoder output."""
    z = torch.randn(1, 64)
    text_emb = torch.randn(1, 64)
    return (z, text_emb)


# ============================================================
# StyleGAN-T Discriminator (projected discriminator)
# ============================================================


class ProjectedDiscriminator(nn.Module):
    """StyleGAN-T projected discriminator.

    Uses feature projections from a lightweight conv backbone (approximating
    DINOv2/CLIP backbone used in paper). Features at multiple scales are projected
    and classified independently.
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        # Lightweight backbone (approximates frozen pretrained in paper)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),  # 4x4
            nn.LeakyReLU(0.2),
        )
        # Feature projection and classification head
        self.proj = nn.Conv2d(base_ch * 4, base_ch, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img)
        feats = self.proj(feats)
        return self.classifier(feats)


def build_stylegan_t_discriminator() -> nn.Module:
    return ProjectedDiscriminator()


def example_input_stylegan_t_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan_t_generator",
        "build_stylegan_t_generator",
        "example_input_stylegan_t_generator",
        "2023",
        "DC",
    ),
    (
        "stylegan_t_text_to_image_generator",
        "build_stylegan_t_text_to_image_generator",
        "example_input_stylegan_t_text_to_image_generator",
        "2023",
        "DC",
    ),
    (
        "stylegan_t_discriminator",
        "build_stylegan_t_discriminator",
        "example_input_stylegan_t_discriminator",
        "2023",
        "DC",
    ),
]
