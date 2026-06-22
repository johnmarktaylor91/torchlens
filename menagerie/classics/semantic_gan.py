"""SemanticGAN: StyleGAN-based model jointly generating images AND segmentation maps.

Li et al., "Semantic Segmentation with Generative Models: Semi-Supervised Learning
and Strong Out-of-Domain Generalization", CVPR 2021. arXiv:2104.05650.
Source: https://github.com/nv-tlabs/semanticGAN_code

SemanticGAN key contribution:
  DUAL-BRANCH GENERATOR: extends StyleGAN to jointly generate:
    1. A photorealistic image (standard StyleGAN synthesis)
    2. A semantic segmentation label map (separate synthesis branch)
  Both branches share the same mapping network and style code w.
  The image branch uses modulated convolutions (StyleGAN2-style weight demodulation).
  The label branch uses separate convolutions (less modulation, more structural).
  This enables semi-supervised training: use paired data for supervised loss,
  unpaired data for GAN loss on both image and label distributions.

Compact random-init CPU model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# StyleGAN2 modulated conv (reused from literature, simplified)
# ============================================================


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated convolution with weight demodulation.

    Per-sample weight modulation: style affine -> per-input-channel scale,
    multiply conv weights by scale, then demodulate (L2 normalize per output channel).
    """

    def __init__(self, in_ch: int, out_ch: int, style_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.02)
        self.style_mod = nn.Linear(style_dim, in_ch)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        style = self.style_mod(w) + 1.0  # (B, in_ch)
        weight = self.weight.unsqueeze(0) * style.view(B, 1, C, 1, 1)  # (B, out_ch, in_ch, k, k)
        # Demodulation
        demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod
        x_flat = x.view(1, B * C, H, W)
        w_flat = weight.view(B * self.out_ch, C, self.kernel_size, self.kernel_size)
        pad = self.kernel_size // 2
        out = F.conv2d(x_flat, w_flat, padding=pad, groups=B)
        return out.view(B, self.out_ch, H, W)


class MappingNetwork(nn.Module):
    """StyleGAN mapping network: z -> w (4-layer MLP)."""

    def __init__(self, z_dim: int, w_dim: int, n_layers: int = 4) -> None:
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================
# MODULE 25: semantic_gan_generator
# ============================================================


class SemanticGANGenerator(nn.Module):
    """SemanticGAN Dual-Branch Generator.

    Shared mapping network maps z -> w style code.
    Image branch: modulated convolutions (StyleGAN2-style) synthesize photorealistic image.
    Label branch: standard convolutions from same feature map synthesize semantic labels.
    Both outputs generated in a single forward pass from the same w.

    Signature: shared mapping network + two synthesis branches from one style code.
    Input: z (1, z_dim).
    Outputs: (image (1,3,H,W), label_map (1,num_classes,H,W)).
    """

    def __init__(
        self,
        z_dim: int = 32,
        w_dim: int = 16,
        ch: int = 16,
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        # Shared mapping network
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=4)
        # Learned constant (shared initial feature map)
        self.const = nn.Parameter(torch.randn(1, ch * 4, 4, 4))
        # Image branch: modulated convolutions (StyleGAN2 signature)
        self.img_mod_conv1 = ModulatedConv2d(ch * 4, ch * 2, w_dim)
        self.img_mod_conv2 = ModulatedConv2d(ch * 2, ch, w_dim)
        self.img_to_rgb = nn.Conv2d(ch, 3, 1)
        # Label branch: standard convolutions (structural synthesis)
        self.lbl_conv1 = nn.Conv2d(ch * 4, ch * 2, 3, 1, 1)
        self.lbl_bn1 = nn.BatchNorm2d(ch * 2)
        self.lbl_conv2 = nn.Conv2d(ch * 2, ch, 3, 1, 1)
        self.lbl_bn2 = nn.BatchNorm2d(ch)
        self.lbl_to_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, z: torch.Tensor) -> tuple:
        B = z.shape[0]
        # Shared mapping: z -> w style code
        w = self.mapping(z)  # (B, w_dim)
        # Initial feature map from learned const
        x = self.const.expand(B, -1, -1, -1)
        # Image branch (modulated conv synthesis)
        img_feat = F.leaky_relu(self.img_mod_conv1(x, w), 0.2)
        img_feat = F.interpolate(img_feat, scale_factor=2, mode="bilinear", align_corners=False)
        img_feat = F.leaky_relu(self.img_mod_conv2(img_feat, w), 0.2)
        image = torch.tanh(self.img_to_rgb(img_feat))
        # Label branch (standard conv synthesis from same initial feature map)
        lbl_feat = F.relu(self.lbl_bn1(self.lbl_conv1(x)))
        lbl_feat = F.interpolate(lbl_feat, scale_factor=2, mode="bilinear", align_corners=False)
        lbl_feat = F.relu(self.lbl_bn2(self.lbl_conv2(lbl_feat)))
        label_map = self.lbl_to_seg(lbl_feat)  # (B, num_classes, H, W) -- logits
        return image, label_map


def build_semantic_gan_generator() -> nn.Module:
    return SemanticGANGenerator(z_dim=32, w_dim=16, ch=16, num_classes=8)


def example_semantic_gan_generator() -> torch.Tensor:
    return torch.randn(1, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "semantic_gan_generator",
        "build_semantic_gan_generator",
        "example_semantic_gan_generator",
        "2021",
        "DC",
    ),
]
